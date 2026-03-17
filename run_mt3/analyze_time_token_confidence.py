"""
Analyze reliability of TIME tokens against note on/off correctness.

Example:
  uv run run_mt3/analyze_time_token_confidence.py \
    --ckpt checkpoints/model.pt \
    --root dataset/maestro-v3.0.0 \
    --split validation \
    --out_dir outputs/time_token_confidence
"""
from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import torch
from tqdm import tqdm
import mir_eval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from my_mt3.tokenizer import VOCAB, INPUT_FRAMES, build_vocab
from my_mt3.infer import greedy_decode_batch_with_logprobs
from my_mt3.eval import extract_notes_in_range
from infer_maestro import collect_pairs_maestro


def _parse_time_groups(token_ids, lps, vocab, step_ms: int):
    """Parse token stream into TIME groups and event list tied to each TIME token."""
    eos_id = int(vocab.eos)
    id2on = {tid: p for p, tid in vocab.note_on.items()}
    id2off = {}
    if vocab.note_off is not None:
        id2off = {tid: p for p, tid in vocab.note_off.items()}
    id2time = {tid: t for t, tid in vocab.time.items()}

    groups = []
    events = []
    current_group_idx = None

    for tok_i, tid in enumerate(token_ids):
        tid = int(tid)
        if tid == eos_id:
            break

        if tid in id2time:
            time_step = int(id2time[tid])
            group = {
                "group_idx": len(groups),
                "token_idx": tok_i,
                "time_step": time_step,
                "time_sec": (time_step * int(step_ms)) / 1000.0,
                "time_logp": float(lps[tok_i]) if tok_i < len(lps) else np.nan,
            }
            groups.append(group)
            current_group_idx = group["group_idx"]
            continue

        if current_group_idx is None:
            continue

        if tid in id2on:
            events.append({
                "group_idx": current_group_idx,
                "token_idx": tok_i,
                "type": "on",
                "pitch": int(id2on[tid]),
                "time_sec": groups[current_group_idx]["time_sec"],
                "event_logp": float(lps[tok_i]) if tok_i < len(lps) else np.nan,
            })
        elif tid in id2off:
            events.append({
                "group_idx": current_group_idx,
                "token_idx": tok_i,
                "type": "off",
                "pitch": int(id2off[tid]),
                "time_sec": groups[current_group_idx]["time_sec"],
                "event_logp": float(lps[tok_i]) if tok_i < len(lps) else np.nan,
            })

    return groups, events


def _matched_pred_indices(ref_times, pred_times, tol: float) -> set[int]:
    """Return matched predicted indices using mir_eval one-to-one event matching."""
    if len(ref_times) == 0 or len(pred_times) == 0:
        return set()
    matched = mir_eval.util.match_events(ref_times, pred_times, window=tol)
    if isinstance(matched, tuple) and len(matched) == 2:
        return {int(i) for i in matched[1]}
    return {int(j) for _, j in matched}


def _event_correct_flags_mir_eval_style(
    events: list[dict],
    ref_int: np.ndarray,
    ref_pitch: np.ndarray,
    *,
    onset_tol: float,
    offset_tol: float,
) -> list[bool]:
    """
    Determine correctness for each predicted on/off event using mir_eval-style
    one-to-one matching per pitch.
    """
    if len(events) == 0:
        return []
    if len(ref_int) == 0:
        return [False] * len(events)

    flags = [False] * len(events)
    ref_on = ref_int[:, 0]
    ref_off = ref_int[:, 1]

    for ev_type, tol, ref_base in (("on", onset_tol, ref_on), ("off", offset_tol, ref_off)):
        idxs = [i for i, e in enumerate(events) if e["type"] == ev_type]
        if not idxs:
            continue
        type_events = [events[i] for i in idxs]
        pred_pitches = np.array([e["pitch"] for e in type_events], dtype=int)
        pred_times = np.array([e["time_sec"] for e in type_events], dtype=float)
        local_flags = np.zeros((len(type_events),), dtype=bool)

        for p in np.unique(pred_pitches):
            pred_mask = pred_pitches == p
            pred_times_p = pred_times[pred_mask]
            pred_local_indices = np.where(pred_mask)[0]

            ref_times_p = ref_base[ref_pitch == p]
            matched_pred = _matched_pred_indices(ref_times_p, pred_times_p, tol=tol)
            for j in matched_pred:
                local_flags[pred_local_indices[j]] = True

        for local_i, global_i in enumerate(idxs):
            flags[global_i] = bool(local_flags[local_i])

    return flags


def _summarize_groups(
    groups: list[dict],
    events: list[dict],
    flags: list[bool],
):
    """Aggregate event correctness into per-time-token rows."""
    group2events: dict[int, list[int]] = {}
    for i, ev in enumerate(events):
        group2events.setdefault(int(ev["group_idx"]), []).append(i)

    rows = []
    for g in groups:
        gi = int(g["group_idx"])
        eidx = group2events.get(gi, [])
        n_events = len(eidx)

        n_on = 0
        n_off = 0
        n_correct = 0
        n_on_correct = 0
        n_off_correct = 0
        for i in eidx:
            if events[i]["type"] == "on":
                n_on += 1
                if flags[i]:
                    n_on_correct += 1
            else:
                n_off += 1
                if flags[i]:
                    n_off_correct += 1
            if flags[i]:
                n_correct += 1

        event_acc = (n_correct / n_events) if n_events > 0 else np.nan
        on_acc = (n_on_correct / n_on) if n_on > 0 else np.nan
        off_acc = (n_off_correct / n_off) if n_off > 0 else np.nan

        rows.append({
            **g,
            "n_events": n_events,
            "n_on": n_on,
            "n_off": n_off,
            "n_correct_events": n_correct,
            "n_on_correct": n_on_correct,
            "n_off_correct": n_off_correct,
            "event_acc": event_acc,
            "on_acc": on_acc,
            "off_acc": off_acc,
            "valid_any": bool(n_correct > 0) if n_events > 0 else np.nan,
            "valid_all": bool(n_correct == n_events) if n_events > 0 else np.nan,
        })
    return rows


def _plot_and_report(df_time: pd.DataFrame, out_dir: Path) -> None:
    """Create plots and print simple correlation/stat summary."""
    if df_time.empty:
        print("No time-token rows found.")
        return

    # Use rows with at least one note event at the time token.
    d = df_time[df_time["n_events"] > 0].copy()
    if d.empty:
        print("No TIME tokens have note on/off events; nothing to analyze.")
        return

    d["valid_any_int"] = d["valid_any"].astype(int)
    d["valid_all_int"] = d["valid_all"].astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = d["time_logp"].values
    y = d["event_acc"].values
    mask = np.isfinite(x) & np.isfinite(y)
    axes[0].scatter(x[mask], y[mask], s=8, alpha=0.2, edgecolors="none")
    if mask.sum() >= 3:
        rp, _ = stats.pearsonr(x[mask], y[mask])
        rs, _ = stats.spearmanr(x[mask], y[mask])
        title = f"time_logp vs event_acc\nPearson={rp:+.3f}, Spearman={rs:+.3f}"
    else:
        title = "time_logp vs event_acc"
    axes[0].set_title(title)
    axes[0].set_xlabel("TIME token logp")
    axes[0].set_ylabel("event_acc at same TIME")
    axes[0].set_ylim(-0.05, 1.05)

    for ax_i, key in enumerate(("valid_any", "valid_all"), start=1):
        g_true = d.loc[d[key] == True, "time_logp"].values  # noqa: E712
        g_false = d.loc[d[key] == False, "time_logp"].values  # noqa: E712
        data = [g_true, g_false]
        labels = [f"{key}=True (n={len(g_true)})", f"{key}=False (n={len(g_false)})"]
        bp = axes[ax_i].boxplot(
            data,
            labels=labels,
            patch_artist=True,
            widths=0.5,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        for patch, color in zip(bp["boxes"], ["#4c94d6", "#e06060"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[ax_i].set_title(f"TIME token logp grouped by {key}")
        axes[ax_i].set_ylabel("TIME token logp")
        axes[ax_i].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "time_token_confidence_analysis.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {fig_path}")

    print("\n=== Time Token Correlation Summary ===")
    pairs = [
        ("event_acc", "event_acc"),
        ("valid_any_int", "valid_any"),
        ("valid_all_int", "valid_all"),
    ]
    for col, name in pairs:
        xv = d["time_logp"].values
        yv = d[col].values
        m = np.isfinite(xv) & np.isfinite(yv)
        if m.sum() >= 3:
            rp, _ = stats.pearsonr(xv[m], yv[m])
            rs, _ = stats.spearmanr(xv[m], yv[m])
        else:
            rp = rs = float("nan")
        print(f"  time_logp vs {name:10s}  Pearson={rp:+.4f}  Spearman={rs:+.4f}")

    # Reliability table by quantile bins
    try:
        d["time_logp_bin"] = pd.qcut(d["time_logp"], q=10, duplicates="drop")
        rel = d.groupby("time_logp_bin", observed=False).agg(
            n=("event_acc", "size"),
            mean_event_acc=("event_acc", "mean"),
            mean_valid_any=("valid_any_int", "mean"),
            mean_valid_all=("valid_all_int", "mean"),
            mean_logp=("time_logp", "mean"),
        ).reset_index()
        rel_path = out_dir / "time_token_reliability_bins.csv"
        rel.to_csv(rel_path, index=False)
        print(f"Reliability bins CSV -> {rel_path}")
    except Exception:
        pass


def run(args):
    from my_mt3.model import MT3Mini
    from my_mt3.audio import load_audio_mono
    from my_mt3.dataset import LogMelCfg, LogMelExtractor

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_frames = INPUT_FRAMES

    vocab = build_vocab(input_frames=input_frames, instrument_type="piano", include_note_off=True)

    model = MT3Mini(vocab_size=len(VOCAB.itos)).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    pairs = collect_pairs_maestro(args.root, split=args.split, program_id=args.program_id)
    if args.max_songs > 0:
        pairs = pairs[:args.max_songs]
    print(f"Songs: {len(pairs)}")

    feat = LogMelExtractor(LogMelCfg(sr=args.sr, n_fft=args.n_fft, hop=args.hop, n_mels=args.n_mels))
    need_samples = (input_frames - 1) * args.hop + args.n_fft
    chunk_sec = need_samples / float(args.sr)

    time_rows = []
    event_rows = []

    for song_idx, (audio_path, midi_path, pid) in enumerate(tqdm(pairs, desc="songs")):
        stem = Path(audio_path).stem
        ref_pm = pretty_midi.PrettyMIDI(midi_path)
        y, _ = load_audio_mono(audio_path, sr=args.sr)
        total_samples = len(y)
        stride_samples = input_frames * args.hop

        starts = list(range(0, max(0, total_samples - need_samples) + 1, stride_samples))
        if len(starts) == 0:
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

        for b0 in range(0, len(starts), args.batch_size):
            b1 = min(len(starts), b0 + args.batch_size)
            mels_bt = torch.from_numpy(np.stack(mel_list[b0:b1])).to(
                device=args.device, dtype=torch.float32
            )

            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        tok_batch, lp_batch = greedy_decode_batch_with_logprobs(
                            model, mels_bt, max_len=args.max_len,
                            device=args.device, program_id=int(pid), vocab=vocab,
                        )
                else:
                    tok_batch, lp_batch = greedy_decode_batch_with_logprobs(
                        model, mels_bt, max_len=args.max_len,
                        device=args.device, program_id=int(pid), vocab=vocab,
                    )

            for local_i in range(len(tok_batch)):
                chunk_idx = b0 + local_i
                ss = starts[chunk_idx]
                t0 = ss / float(args.sr)
                t1 = t0 + chunk_sec

                token_ids = tok_batch[local_i]
                lps = lp_batch[local_i]
                groups, events = _parse_time_groups(token_ids, lps, vocab, step_ms=args.step_ms)
                if not groups:
                    continue

                ref_int, ref_pitch, _ = extract_notes_in_range(
                    ref_pm, t0, t1, program=int(pid),
                )
                flags = _event_correct_flags_mir_eval_style(
                    events,
                    ref_int,
                    ref_pitch,
                    onset_tol=args.onset_tol,
                    offset_tol=args.offset_tol,
                )
                g_rows = _summarize_groups(groups, events, flags)

                for row in g_rows:
                    time_rows.append({
                        "stem": stem,
                        "song_idx": song_idx,
                        "chunk_idx": chunk_idx,
                        "t0": t0,
                        "t1": t1,
                        **row,
                    })

                for ev, c in zip(events, flags):
                    event_rows.append({
                        "stem": stem,
                        "song_idx": song_idx,
                        "chunk_idx": chunk_idx,
                        "t0": t0,
                        "t1": t1,
                        **ev,
                        "correct": bool(c),
                    })

    df_time = pd.DataFrame(time_rows)
    df_event = pd.DataFrame(event_rows)

    time_csv = out_dir / "time_token_confidence.csv"
    event_csv = out_dir / "time_token_events.csv"
    df_time.to_csv(time_csv, index=False)
    df_event.to_csv(event_csv, index=False)
    print(f"Saved -> {time_csv}  ({len(df_time)} rows)")
    print(f"Saved -> {event_csv} ({len(df_event)} rows)")

    _plot_and_report(df_time, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="model checkpoint")
    ap.add_argument("--root", type=str, required=True, help="MAESTRO root dir")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--out_dir", type=str, default="outputs/time_token_confidence")
    ap.add_argument("--program_id", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--step_ms", type=int, default=10)
    ap.add_argument("--max_songs", type=int, default=0, help="limit number of songs (0=all)")

    ap.add_argument("--onset_tol", type=float, default=0.05, help="onset tolerance [sec]")
    ap.add_argument("--offset_tol", type=float, default=0.05, help="offset tolerance [sec]")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
