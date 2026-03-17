from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import mir_eval
from my_mt3.dataset import AMTDataset
from my_mt3.model import MT3Mini
from my_mt3.tokenizer import Vocab, build_vocab
from my_mt3.train import make_collate
from my_mt3.audio import load_audio_mono
from my_mt3.dataset import LogMelCfg, LogMelExtractor
from my_mt3.eval import extract_notes_in_range
from my_mt3.infer import greedy_decode_batch_with_logprobs


def collect_maestro_pairs(
    root: str | Path,
    *,
    split: str = "validation",
    max_songs: int = 0,
    program_id: int = 0,
) -> list[tuple[str, str, int]]:
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    rows = df[df["split"] == split]
    pairs: list[tuple[str, str, int]] = []
    for _, row in rows.iterrows():
        wav = root / str(row["audio_filename"])
        mid = root / str(row["midi_filename"])
        if not wav.exists() or not mid.exists():
            continue
        pairs.append((str(wav), str(mid), int(program_id)))
        if max_songs > 0 and len(pairs) >= max_songs:
            break
    return pairs


def _load_model(ckpt_path: str | Path, *, vocab_size: int, device: torch.device) -> MT3Mini:
    model = MT3Mini(vocab_size=vocab_size).to(device)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _time_hist_from_tokens(token_ids_2d: torch.Tensor, vocab: Vocab) -> np.ndarray:
    # token_ids_2d: [B, S]
    num_time = len(vocab.time)
    hist = np.zeros((num_time,), dtype=np.int64)
    for tid, t in ((tid, t) for t, tid in vocab.time.items()):
        c = int((token_ids_2d == int(tid)).sum().item())
        hist[int(t)] = c
    return hist


@torch.no_grad()
def collect_time_hist(
    model: MT3Mini | None,
    dl: DataLoader,
    *,
    vocab: Vocab,
    device: torch.device,
    mode: str,
    max_batches: int = 0,
) -> np.ndarray:
    # mode: "target" or "pred"
    total = np.zeros((len(vocab.time),), dtype=np.int64)
    n_batches = 0

    for mels, y_in, y_tg in dl:
        mels = mels.to(device, non_blocking=True)
        y_in = y_in.to(device, non_blocking=True)
        y_tg = y_tg.to(device, non_blocking=True)

        if mode == "target":
            toks = y_tg
        elif mode == "pred":
            if model is None:
                raise ValueError("model is required for pred mode")
            logits = model(mels, y_in)
            toks = logits.argmax(dim=-1)
        else:
            raise ValueError(f"unknown mode: {mode}")

        total += _time_hist_from_tokens(toks, vocab)
        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break

    return total


def _plot_hist(
    by_label: dict[str, np.ndarray],
    *,
    out_png: Path,
    top_k: int = 80,
) -> None:
    totals = {k: int(v.sum()) for k, v in by_label.items()}
    n_time = max(len(v) for v in by_label.values()) if by_label else 0
    x = np.arange(n_time, dtype=np.int32)

    # 可視性のため、合計頻度上位top_kのみ表示
    score = np.zeros((n_time,), dtype=np.float64)
    for v in by_label.values():
        score += v.astype(np.float64)
    if n_time > top_k:
        idx = np.argsort(score)[-top_k:]
        idx = np.sort(idx)
    else:
        idx = x

    plt.figure(figsize=(12, 6))
    for label, hist in by_label.items():
        denom = max(totals[label], 1)
        y = hist[idx].astype(np.float64) / float(denom)
        plt.plot(idx, y, marker="o", linewidth=1.2, markersize=2.5, label=label)
    plt.xlabel("TIM_k (time token index)")
    plt.ylabel("Relative frequency")
    plt.title("Time-shift token distribution")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def _parse_time_groups(token_ids, lps, vocab: Vocab, step_ms: int):
    eos_id = int(vocab.eos)
    id2on = {tid: p for p, tid in vocab.note_on.items()}
    id2time = {tid: t for t, tid in vocab.time.items()}

    groups = []
    onsets = []
    cur_group_idx = None

    for tok_i, tid in enumerate(token_ids):
        tid = int(tid)
        if tid == eos_id:
            break
        if tid in id2time:
            t_step = int(id2time[tid])
            groups.append({
                "group_idx": len(groups),
                "token_idx": tok_i,
                "time_step": t_step,
                "time_sec": (t_step * int(step_ms)) / 1000.0,
                "time_logp": float(lps[tok_i]) if tok_i < len(lps) else np.nan,
            })
            cur_group_idx = len(groups) - 1
            continue
        if cur_group_idx is None:
            continue
        if tid in id2on:
            onsets.append({
                "group_idx": cur_group_idx,
                "token_idx": tok_i,
                "pitch": int(id2on[tid]),
                "time_sec": groups[cur_group_idx]["time_sec"],
                "logp": float(lps[tok_i]) if tok_i < len(lps) else np.nan,
            })
    return groups, onsets


def _matched_pred_indices(ref_times, pred_times, tol: float) -> set[int]:
    if len(ref_times) == 0 or len(pred_times) == 0:
        return set()
    matched = mir_eval.util.match_events(ref_times, pred_times, window=tol)
    if isinstance(matched, tuple) and len(matched) == 2:
        return {int(i) for i in matched[1]}
    return {int(j) for _, j in matched}


def _onset_correct_flags_mir_eval(
    onset_events: list[dict],
    ref_int: np.ndarray,
    ref_pitch: np.ndarray,
    *,
    onset_tol: float,
) -> list[bool]:
    if len(onset_events) == 0:
        return []
    if len(ref_int) == 0:
        return [False] * len(onset_events)

    flags = np.zeros((len(onset_events),), dtype=bool)
    pred_pitches = np.array([e["pitch"] for e in onset_events], dtype=int)
    pred_times = np.array([e["time_sec"] for e in onset_events], dtype=float)
    ref_on = ref_int[:, 0]

    for p in np.unique(pred_pitches):
        pred_mask = pred_pitches == p
        pred_times_p = pred_times[pred_mask]
        pred_local_indices = np.where(pred_mask)[0]
        ref_times_p = ref_on[ref_pitch == p]
        matched_pred = _matched_pred_indices(ref_times_p, pred_times_p, tol=onset_tol)
        for j in matched_pred:
            flags[pred_local_indices[j]] = True
    return [bool(v) for v in flags]


def _plot_onset_burst_probability(df: pd.DataFrame, out_dir: Path, min_burst: int) -> None:
    if df.empty:
        print("No onset burst data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    p_correct = df.loc[df["correct"], "prob"].values
    p_incorrect = df.loc[~df["correct"], "prob"].values
    bp = axes[0].boxplot(
        [p_correct, p_incorrect],
        labels=[f"Correct (n={len(p_correct)})", f"Incorrect (n={len(p_incorrect)})"],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    for patch, color in zip(bp["boxes"], ["#4c94d6", "#e06060"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Onset burst token probability by correctness")
    axes[0].set_ylabel("P(token)")
    axes[0].grid(axis="y", alpha=0.3)

    x = df["burst_size"].values.astype(np.int32)
    y = df["prob"].values.astype(np.float64)
    c = df["correct"].values
    axes[1].scatter(x[c], y[c], s=8, alpha=0.25, edgecolors="none", label="Correct")
    axes[1].scatter(x[~c], y[~c], s=8, alpha=0.25, edgecolors="none", label="Incorrect")
    axes[1].set_title("Onset burst size vs token probability")
    axes[1].set_xlabel("Burst size at same TIME")
    axes[1].set_ylabel("P(token)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(f"Onset bursts (min size={min_burst})", fontsize=12)
    fig.tight_layout()
    out_png = out_dir / "onset_burst_probability_correctness.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG:  {out_png}")


@torch.no_grad()
def collect_onset_burst_rows(
    model: MT3Mini,
    pairs: list[tuple[str, str, int]],
    *,
    vocab: Vocab,
    device: torch.device,
    input_frames: int,
    step_ms: int,
    batch_size: int,
    max_len: int,
    onset_tol: float,
    min_burst: int,
    max_batches: int = 0,
) -> list[dict]:
    feat = LogMelExtractor(LogMelCfg(sr=16000, n_fft=2048, hop=256, n_mels=256))
    need_samples = (input_frames - 1) * 256 + 2048
    chunk_sec = need_samples / 16000.0

    rows: list[dict] = []
    global_batch_count = 0
    for song_idx, (audio_path, midi_path, pid) in enumerate(tqdm(pairs, desc="songs(burst)")):
        stem = Path(audio_path).stem
        print(f"[burst] start song {song_idx+1}/{len(pairs)}: {stem}")
        ref_pm = pretty_midi.PrettyMIDI(midi_path)
        y, _ = load_audio_mono(audio_path, sr=16000)
        total_samples = len(y)
        stride_samples = input_frames * 256

        starts = list(range(0, max(0, total_samples - need_samples) + 1, stride_samples))
        if not starts:
            starts = [0]

        mel_list = []
        for ss in starts:
            y_seg = y[ss:ss + need_samples]
            if len(y_seg) < need_samples:
                y_seg = np.pad(y_seg, (0, need_samples - len(y_seg)))
            mel = feat(y_seg)
            if mel.shape[0] > input_frames:
                mel = mel[:input_frames]
            elif mel.shape[0] < input_frames:
                mel = np.pad(mel, ((0, input_frames - mel.shape[0]), (0, 0)))
            mel_list.append(mel.astype(np.float32, copy=False))

        total_batches = (len(starts) + batch_size - 1) // batch_size
        for b0 in tqdm(
            range(0, len(starts), batch_size),
            desc=f"chunks({stem})",
            total=total_batches,
            leave=False,
        ):
            b1 = min(len(starts), b0 + batch_size)
            mels_bt = torch.from_numpy(np.stack(mel_list[b0:b1])).to(device=device, dtype=torch.float32)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tok_batch, lp_batch = greedy_decode_batch_with_logprobs(
                        model, mels_bt, max_len=max_len, device=device.type, program_id=int(pid), vocab=vocab
                    )
            else:
                tok_batch, lp_batch = greedy_decode_batch_with_logprobs(
                    model, mels_bt, max_len=max_len, device=device.type, program_id=int(pid), vocab=vocab
                )

            for local_i in range(len(tok_batch)):
                chunk_idx = b0 + local_i
                ss = starts[chunk_idx]
                t0 = ss / 16000.0
                t1 = t0 + chunk_sec

                token_ids = tok_batch[local_i]
                lps = lp_batch[local_i]
                groups, onsets = _parse_time_groups(token_ids, lps, vocab, step_ms=step_ms)
                if not groups or not onsets:
                    continue

                ref_int, ref_pitch, _ = extract_notes_in_range(ref_pm, t0, t1, program=int(pid))
                flags = _onset_correct_flags_mir_eval(onsets, ref_int, ref_pitch, onset_tol=onset_tol)

                group_count: dict[int, int] = {}
                for ev in onsets:
                    gi = int(ev["group_idx"])
                    group_count[gi] = group_count.get(gi, 0) + 1

                for ev, corr in zip(onsets, flags):
                    burst_size = group_count.get(int(ev["group_idx"]), 0)
                    if burst_size < int(min_burst):
                        continue
                    lp = float(ev["logp"])
                    rows.append({
                        "stem": stem,
                        "song_idx": song_idx,
                        "chunk_idx": chunk_idx,
                        "t0": float(t0),
                        "t1": float(t1),
                        "time_sec": float(ev["time_sec"]),
                        "pitch": int(ev["pitch"]),
                        "logp": lp,
                        "prob": float(np.exp(lp)),
                        "correct": bool(corr),
                        "burst_size": int(burst_size),
                    })
            global_batch_count += 1
            if max_batches > 0 and global_batch_count >= max_batches:
                print(f"[burst] reached max_batches={max_batches}, early stop.")
                return rows
        print(f"[burst] done song {song_idx+1}/{len(pairs)}: {stem}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze TIM_* token distribution on MAESTRO split")
    ap.add_argument("--ckpt", action="append", default=None, help="checkpoint path (repeatable)")
    ap.add_argument("--label", action="append", default=None, help="label per checkpoint")
    ap.add_argument(
        "--include_target",
        action="store_true",
        help="also include GT target(y_tg) TIM distribution as reference",
    )
    ap.add_argument("--root", type=str, default="dataset/maestro-v3.0.0")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--input_frames", type=int, default=256)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_songs", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="outputs/time_shift_token_dist")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--analyze_onset_burst", action="store_true", help="analyze onset burst token probability vs correctness")
    ap.add_argument("--min_burst", type=int, default=2, help="minimum onset count at the same TIME to define burst")
    ap.add_argument("--step_ms", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--onset_tol", type=float, default=0.05, help="onset matching tolerance [sec]")
    args = ap.parse_args()

    ckpts = list(args.ckpt or [])
    labels = list(args.label or [])
    if ckpts:
        if labels and len(labels) != len(ckpts):
            raise SystemExit("len(--label) must match len(--ckpt)")
        if not labels:
            labels = [Path(p).stem for p in ckpts]
    elif not args.include_target:
        raise SystemExit("Specify at least one --ckpt or --include_target")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    vocab = build_vocab(input_frames=int(args.input_frames), instrument_type="piano", include_note_off=True)
    pairs = collect_maestro_pairs(
        args.root, split=args.split, max_songs=int(args.max_songs), program_id=0
    )
    if not pairs:
        raise SystemExit("No pairs found")

    by_label: dict[str, np.ndarray] = {}

    if args.analyze_onset_burst:
        if not ckpts:
            raise SystemExit("--analyze_onset_burst requires at least one --ckpt")
        all_rows = []
        for ckpt, label in zip(ckpts, labels):
            print(f"\n=== onset burst analysis: {label} ===")
            model = _load_model(ckpt, vocab_size=len(vocab.itos), device=device)
            rows = collect_onset_burst_rows(
                model,
                pairs,
                vocab=vocab,
                device=device,
                input_frames=int(args.input_frames),
                step_ms=int(args.step_ms),
                batch_size=int(args.bs),
                max_len=int(args.max_len),
                onset_tol=float(args.onset_tol),
                min_burst=int(args.min_burst),
                max_batches=int(args.max_batches),
            )
            for r in rows:
                r["label"] = label
            all_rows.extend(rows)
            print(f"burst onset tokens={len(rows)}")

        out_csv = out_dir / "onset_burst_probability_correctness.csv"
        df = pd.DataFrame(all_rows)
        df.to_csv(out_csv, index=False)
        print(f"Saved CSV:  {out_csv}")

        if not df.empty:
            if len(df["label"].unique()) == 1:
                _plot_onset_burst_probability(df, out_dir, min_burst=int(args.min_burst))
            else:
                # Multi-ckpt summary figure by label
                fig, ax = plt.subplots(figsize=(8, 5))
                labels_u = list(df["label"].dropna().unique())
                data = [df.loc[(df["label"] == lb) & (df["correct"] == True), "prob"].values for lb in labels_u]  # noqa: E712
                ax.boxplot(
                    data,
                    labels=[f"{lb}\n(correct)" for lb in labels_u],
                    patch_artist=True,
                    widths=0.5,
                    showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                )
                ax.set_title("Onset burst token probability (correct only) by checkpoint")
                ax.set_ylabel("P(token)")
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                out_png = out_dir / "onset_burst_probability_by_label.png"
                fig.savefig(out_png, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved PNG:  {out_png}")

            p_c = df.loc[df["correct"], "prob"].values
            p_i = df.loc[~df["correct"], "prob"].values
            print("\n=== onset burst probability summary ===")
            print(f"correct n={len(p_c)} mean={np.mean(p_c):.4f} median={np.median(p_c):.4f}" if len(p_c) else "correct n=0")
            print(f"incorrect n={len(p_i)} mean={np.mean(p_i):.4f} median={np.median(p_i):.4f}" if len(p_i) else "incorrect n=0")
        else:
            print("No onset burst tokens found. Try lowering --min_burst or changing split/max_songs.")
        return

    ds = AMTDataset(
        pairs,
        mode="validation",
        sr=16000,
        input_frames=int(args.input_frames),
        vocab=vocab,
    )
    dl = DataLoader(
        ds,
        batch_size=int(args.bs),
        shuffle=False,
        collate_fn=make_collate(vocab),
        num_workers=int(args.num_workers),
        pin_memory=False,
    )

    if args.include_target:
        print("\n=== target ===")
        h_tg = collect_time_hist(
            None, dl, vocab=vocab, device=device, mode="target", max_batches=int(args.max_batches)
        )
        by_label["target"] = h_tg
        print(f"tokens={int(h_tg.sum())}")

    for ckpt, label in zip(ckpts, labels):
        print(f"\n=== {label} ===")
        model = _load_model(ckpt, vocab_size=len(vocab.itos), device=device)
        h_pr = collect_time_hist(
            model, dl, vocab=vocab, device=device, mode="pred", max_batches=int(args.max_batches)
        )
        by_label[label] = h_pr
        print(f"tokens={int(h_pr.sum())}")

    # 保存
    out_json = out_dir / "time_shift_token_distribution.json"
    serial = {}
    for label, hist in by_label.items():
        total = int(hist.sum())
        rel = (hist.astype(np.float64) / float(max(total, 1))).tolist()
        serial[label] = {
            "total_time_tokens": total,
            "hist_counts": hist.tolist(),
            "hist_rel": rel,
            "argmax_time_idx": int(np.argmax(hist)) if total > 0 else None,
        }
    out_json.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")

    out_png = out_dir / "time_shift_token_distribution.png"
    _plot_hist(by_label, out_png=out_png, top_k=80)
    print(f"\nSaved JSON: {out_json}")
    print(f"Saved PNG:  {out_png}")


if __name__ == "__main__":
    main()
