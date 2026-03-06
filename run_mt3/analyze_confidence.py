"""
Chunk-level confidence vs evaluation metric correlation analysis.

Mode 1 — from scratch (requires model):
  uv run run/analyze_confidence.py \
    --ckpt checkpoints/model.pt \
    --root dataset/maestro-v3.0.0 \
    --split validation \
    --out_dir outputs/confidence_analysis \
    --batch_size 32

Mode 2 — from pre-saved confidence CSV (no model needed):
  uv run run/analyze_confidence.py \
    --confidence_csv outputs/maestro_val_pred/chunk_confidence.csv \
    --pred_dir outputs/maestro_val_pred \
    --root dataset/maestro-v3.0.0 \
    --split validation \
    --out_dir outputs/confidence_analysis
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from my_mt3.tokenizer import VOCAB, INPUT_FRAMES, build_vocab
from my_mt3.infer import to_midi_from_tokens_piano, greedy_decode_batch_with_logprobs
from my_mt3.eval import extract_notes_in_range, evaluate_notes_direct

from infer_maestro import collect_pairs_maestro


def pm_to_arrays(pm: pretty_midi.PrettyMIDI):
    """PrettyMIDI -> (intervals [N,2], pitches [N], velocities [N])."""
    intervals, pitches, velocities = [], [], []
    for inst in pm.instruments:
        for n in inst.notes:
            intervals.append([n.start, n.end])
            pitches.append(n.pitch)
            velocities.append(n.velocity)
    if not intervals:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
    intervals = np.asarray(intervals, dtype=float)
    pitches = np.asarray(pitches, dtype=int)
    velocities = np.asarray(velocities, dtype=int)
    order = np.argsort(intervals[:, 0])
    return intervals[order], pitches[order], velocities[order]


def plot_and_summarize(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate 6 scatter plots + correlation summary from a DataFrame with confidence & metric columns."""
    conf_keys = [("log_pyx", "log P(Y|X)"), ("log_pyx_norm", "log P(Y|X) / T")]
    metric_keys = [("onset_f", "Onset F1"), ("onset_pitch_f", "Onset+Pitch F1"),
                   ("note_f", "Note F1"), ("note_vel_f", "Note+Vel F1")]

    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(2, n_metrics, figsize=(6 * n_metrics, 10))

    for row_i, (conf_col, conf_label) in enumerate(conf_keys):
        if conf_col not in df.columns:
            continue
        for col_i, (met_col, met_label) in enumerate(metric_keys):
            if met_col not in df.columns:
                continue
            ax = axes[row_i, col_i]
            x = df[conf_col].values
            y = df[met_col].values

            mask = np.isfinite(x) & np.isfinite(y)
            x_valid, y_valid = x[mask], y[mask]

            ax.scatter(x_valid, y_valid, alpha=0.15, s=8, edgecolors="none")

            if len(x_valid) >= 3:
                r_pearson, _ = stats.pearsonr(x_valid, y_valid)
                r_spearman, _ = stats.spearmanr(x_valid, y_valid)
            else:
                r_pearson = r_spearman = float("nan")

            ax.set_xlabel(conf_label)
            ax.set_ylabel(met_label)
            ax.set_title(
                f"{conf_label}  vs  {met_label}\n"
                f"Pearson r={r_pearson:.3f}  Spearman rho={r_spearman:.3f}"
            )

    fig.suptitle("Decoder Confidence vs Evaluation Metrics (per chunk)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig_path = out_dir / "confidence_vs_metrics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {fig_path}")

    print("\n=== Correlation Summary ===")
    for conf_col, conf_label in conf_keys:
        if conf_col not in df.columns:
            continue
        for met_col, met_label in metric_keys:
            if met_col not in df.columns:
                continue
            x = df[conf_col].values
            y = df[met_col].values
            mask = np.isfinite(x) & np.isfinite(y)
            x_v, y_v = x[mask], y[mask]
            if len(x_v) >= 3:
                r_p, _ = stats.pearsonr(x_v, y_v)
                r_s, _ = stats.spearmanr(x_v, y_v)
            else:
                r_p = r_s = float("nan")
            print(f"  {conf_label:20s} vs {met_label:15s}  Pearson={r_p:+.4f}  Spearman={r_s:+.4f}")
    print()


def plot_chunk_piano_roll(
    ref_int: np.ndarray, ref_pitch: np.ndarray,
    est_int: np.ndarray, est_pitch: np.ndarray,
    *,
    title: str,
    save_path: Path,
    chunk_sec: float = 0.0,
) -> None:
    """Overlay GT (blue) and Pred (red) piano roll for one chunk."""
    fig, ax = plt.subplots(figsize=(12, 5))

    note_h = 0.8

    for i in range(len(ref_int)):
        onset, offset = ref_int[i]
        p = ref_pitch[i]
        ax.add_patch(Rectangle(
            (onset, p - note_h / 2), offset - onset, note_h,
            facecolor="royalblue", edgecolor="navy", alpha=0.5, linewidth=0.5,
        ))

    for i in range(len(est_int)):
        onset, offset = est_int[i]
        p = est_pitch[i]
        ax.add_patch(Rectangle(
            (onset, p - note_h / 2), offset - onset, note_h,
            facecolor="crimson", edgecolor="darkred", alpha=0.5, linewidth=0.5,
        ))

    all_pitches = np.concatenate([ref_pitch, est_pitch]) if (len(ref_pitch) + len(est_pitch)) > 0 else np.array([60])
    pmin, pmax = int(all_pitches.min()) - 2, int(all_pitches.max()) + 2

    ax.set_xlim(0, chunk_sec if chunk_sec > 0 else max(0.1, ax.get_xlim()[1]))
    ax.set_ylim(pmin, pmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(title, fontsize=10)

    legend_elems = [
        Line2D([0], [0], color="royalblue", lw=6, alpha=0.5, label=f"GT ({len(ref_pitch)} notes)"),
        Line2D([0], [0], color="crimson", lw=6, alpha=0.5, label=f"Pred ({len(est_pitch)} notes)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def generate_piano_rolls(
    df: pd.DataFrame,
    *,
    pred_dir: Path,
    root: str,
    split: str,
    program_id: int,
    out_dir: Path,
    n_samples: int = 0,
) -> None:
    """
    Generate piano roll PNGs.
    n_samples=0: all chunks, n_samples>0: sample N spanning worst->best by note_f.
    """
    pr_dir = out_dir / "piano_rolls"
    pr_dir.mkdir(parents=True, exist_ok=True)

    if n_samples > 0 and len(df) > n_samples:
        df_sorted = df.sort_values("note_f").reset_index(drop=True)
        indices = np.linspace(0, len(df_sorted) - 1, n_samples, dtype=int)
        df_sample = df_sorted.iloc[indices].copy()
    else:
        df_sample = df.copy()

    pairs = collect_pairs_maestro(root, split=split, program_id=program_id)
    stem2midi = {Path(ap).stem: mp for ap, mp, *_ in pairs}

    pm_cache: dict[str, pretty_midi.PrettyMIDI] = {}

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="piano rolls"):
        stem = row["stem"]
        t0, t1 = float(row["t0"]), float(row["t1"])
        chunk_sec = t1 - t0

        ref_midi_path = stem2midi.get(stem)
        if ref_midi_path is None:
            continue
        pred_midi_path = pred_dir / f"{stem}.pred.mid"
        if not pred_midi_path.exists():
            continue

        cache_key_ref = f"__ref__{stem}"
        if cache_key_ref not in pm_cache:
            try:
                pm_cache[cache_key_ref] = pretty_midi.PrettyMIDI(ref_midi_path)
            except Exception:
                continue
        if stem not in pm_cache:
            try:
                pm_cache[stem] = pretty_midi.PrettyMIDI(str(pred_midi_path))
            except Exception:
                continue

        ref_int, ref_pitch, _ = extract_notes_in_range(pm_cache[cache_key_ref], t0, t1, program=program_id)
        est_int, est_pitch, _ = extract_notes_in_range(pm_cache[stem], t0, t1, program=program_id)

        note_f = row.get("note_f", float("nan"))
        onset_f = row.get("onset_f", float("nan"))
        log_norm = row.get("log_pyx_norm", float("nan"))
        chunk_idx = int(row.get("chunk_idx", 0))

        title = (
            f"{stem}  chunk={chunk_idx}  [{t0:.1f}s–{t1:.1f}s]\n"
            f"onset_f={onset_f:.3f}  note_f={note_f:.3f}  "
            f"log P/T={log_norm:.2f}  tokens={int(row.get('n_tokens', 0))}"
        )

        fname = f"{stem}__c{chunk_idx:04d}_nf{note_f:.3f}.png"
        save_path = pr_dir / fname

        plot_chunk_piano_roll(
            ref_int, ref_pitch, est_int, est_pitch,
            title=title, save_path=save_path, chunk_sec=chunk_sec,
        )

    print(f"Piano rolls saved -> {pr_dir}  ({len(df_sample)} images)")


# ──────────────────────────────────────────────────────────────
# Token-level confidence analysis
# ──────────────────────────────────────────────────────────────

def _parse_token_events(token_ids, lps, vocab, step_ms: int):
    """
    Walk through generated tokens and extract per-token info for NON/NOF.

    Returns list of dicts:
      {"type": "on"|"off", "pitch": int, "time_sec": float, "logp": float}
    """
    eos_id = int(vocab.eos)
    id2on = {tid: p for p, tid in vocab.note_on.items()}
    id2off = {}
    if vocab.note_off is not None:
        id2off = {tid: p for p, tid in vocab.note_off.items()}
    id2time = {tid: t for t, tid in vocab.time.items()}

    cur_ms = 0
    events = []
    for i, tid in enumerate(token_ids):
        tid = int(tid)
        if tid == eos_id:
            break
        if tid in id2time:
            cur_ms = id2time[tid] * int(step_ms)
        elif tid in id2on:
            lp = lps[i] if i < len(lps) else 0.0
            events.append({"type": "on", "pitch": id2on[tid],
                           "time_sec": cur_ms / 1000.0, "logp": float(lp)})
        elif tid in id2off:
            lp = lps[i] if i < len(lps) else 0.0
            events.append({"type": "off", "pitch": id2off[tid],
                           "time_sec": cur_ms / 1000.0, "logp": float(lp)})
    return events


def _match_events_to_ref(
    token_events,
    ref_int, ref_pitch,
    *,
    onset_tol: float = 0.05,
    offset_tol: float = 0.05,
):
    """
    For each token event, determine if it is "correct" (matches a reference note).
      - NON token: correct if ref has same pitch with onset within onset_tol
      - NOF token: correct if ref has same pitch with offset within offset_tol

    Returns list of booleans parallel to token_events.
    """
    if len(ref_int) == 0:
        return [False] * len(token_events)

    ref_onsets = ref_int[:, 0]
    ref_offsets = ref_int[:, 1]

    correct = []
    for ev in token_events:
        p = ev["pitch"]
        t = ev["time_sec"]
        pitch_mask = ref_pitch == p
        if not pitch_mask.any():
            correct.append(False)
            continue
        if ev["type"] == "on":
            dists = np.abs(ref_onsets[pitch_mask] - t)
            correct.append(bool(dists.min() <= onset_tol))
        else:
            dists = np.abs(ref_offsets[pitch_mask] - t)
            correct.append(bool(dists.min() <= offset_tol))
    return correct


def plot_token_confidence_boxplots(token_rows: list, out_dir: Path) -> None:
    """
    Create 3 box-plot figures (on / off / on+off), each split by correct vs incorrect.
    """
    df = pd.DataFrame(token_rows)
    if df.empty:
        print("No token-level data for box plots.")
        return

    configs = [
        ("on", "Note-On tokens", df[df["type"] == "on"]),
        ("off", "Note-Off tokens", df[df["type"] == "off"]),
        ("on_off", "Note-On + Note-Off tokens", df),
    ]

    for suffix, title, sub in configs:
        if sub.empty:
            continue

        groups = [
            sub.loc[sub["correct"], "logp"].values,
            sub.loc[~sub["correct"], "logp"].values,
        ]
        labels = [
            f"Correct (n={len(groups[0])})",
            f"Incorrect (n={len(groups[1])})",
        ]
        colors = ["#4c94d6", "#e06060"]

        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                        widths=0.5, showfliers=True,
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.set_ylabel("log P(y_t | y_{<t}, X)")
        ax.set_title(f"Token-level Confidence: {title}")
        ax.grid(axis="y", alpha=0.3)

        medians_str = "  ".join(
            f"{l}: med={np.median(g):.3f}" for l, g in zip(labels, groups) if len(g)
        )
        ax.text(0.02, 0.02, medians_str, transform=ax.transAxes,
                fontsize=8, verticalalignment="bottom", color="gray")

        fig.tight_layout()
        fig_path = out_dir / f"token_confidence_boxplot_{suffix}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Token box plot saved -> {fig_path}")

    csv_path = out_dir / "token_confidence.csv"
    df.to_csv(csv_path, index=False)
    print(f"Token confidence CSV -> {csv_path}  ({len(df)} tokens)")


# ──────────────────────────────────────────────────────────────
# Mode 2: offline analysis from pre-saved confidence CSV
# ──────────────────────────────────────────────────────────────
def run_offline(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_conf = pd.read_csv(args.confidence_csv)
    print(f"Loaded confidence CSV: {len(df_conf)} chunks")

    pred_dir = Path(args.pred_dir)
    pairs = collect_pairs_maestro(args.root, split=args.split, program_id=args.program_id)
    stem2midi = {Path(ap).stem: mp for ap, mp, *_ in pairs}

    # cache predicted PrettyMIDI per stem
    pred_pm_cache: dict[str, pretty_midi.PrettyMIDI] = {}

    rows = []
    for _, row in tqdm(df_conf.iterrows(), total=len(df_conf), desc="eval chunks"):
        stem = row["stem"]
        t0, t1 = float(row["t0"]), float(row["t1"])

        ref_midi_path = stem2midi.get(stem)
        if ref_midi_path is None:
            continue
        pred_midi_path = pred_dir / f"{stem}.pred.mid"
        if not pred_midi_path.exists():
            continue

        if stem not in pred_pm_cache:
            try:
                pred_pm_cache[stem] = pretty_midi.PrettyMIDI(str(pred_midi_path))
            except Exception:
                continue
        pred_pm = pred_pm_cache[stem]

        ref_pm_key = f"__ref__{stem}"
        if ref_pm_key not in pred_pm_cache:
            try:
                pred_pm_cache[ref_pm_key] = pretty_midi.PrettyMIDI(ref_midi_path)
            except Exception:
                continue
        ref_pm = pred_pm_cache[ref_pm_key]

        ref_int, ref_pitch, ref_vel = extract_notes_in_range(
            ref_pm, t0, t1, program=args.program_id,
        )
        est_int, est_pitch, est_vel = extract_notes_in_range(
            pred_pm, t0, t1, program=args.program_id,
        )

        try:
            m = evaluate_notes_direct(ref_int, ref_pitch, ref_vel, est_int, est_pitch, est_vel)
        except Exception:
            continue

        rows.append({
            "stem": stem,
            "chunk_idx": int(row["chunk_idx"]),
            "t0": t0,
            "t1": t1,
            "n_tokens": int(row["n_tokens"]),
            "log_pyx": float(row["log_pyx"]),
            "log_pyx_norm": float(row["log_pyx_norm"]),
            **m,
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "chunk_confidence_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved -> {csv_path}  ({len(df)} chunks)")

    if len(df) < 2:
        print("Not enough data points for correlation analysis.")
        return

    plot_and_summarize(df, out_dir)

    if args.piano_roll_n != 0:
        generate_piano_rolls(
            df,
            pred_dir=pred_dir,
            root=args.root,
            split=args.split,
            program_id=args.program_id,
            out_dir=out_dir,
            n_samples=max(0, args.piano_roll_n),
        )


# ──────────────────────────────────────────────────────────────
# Mode 1: from scratch (with model)
# ──────────────────────────────────────────────────────────────
def run_from_scratch(args):
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

    rows = []
    token_rows = []

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

                n_tokens = len(lps)
                if n_tokens == 0:
                    continue

                log_pyx = sum(lps)
                log_pyx_norm = log_pyx / n_tokens

                res = to_midi_from_tokens_piano(
                    token_ids, program_id=int(pid), step_ms=args.step_ms, vocab=vocab,
                )
                est_int, est_pitch, est_vel = pm_to_arrays(res.pm)
                ref_int, ref_pitch, ref_vel = extract_notes_in_range(
                    ref_pm, t0, t1, program=int(pid),
                )

                try:
                    m = evaluate_notes_direct(
                        ref_int, ref_pitch, ref_vel,
                        est_int, est_pitch, est_vel,
                    )
                except Exception:
                    continue

                rows.append({
                    "stem": stem,
                    "song_idx": song_idx,
                    "chunk_idx": chunk_idx,
                    "t0": t0,
                    "t1": t1,
                    "n_tokens": n_tokens,
                    "log_pyx": log_pyx,
                    "log_pyx_norm": log_pyx_norm,
                    **m,
                })

                tok_evts = _parse_token_events(token_ids, lps, vocab, step_ms=args.step_ms)
                tok_correct = _match_events_to_ref(
                    tok_evts, ref_int, ref_pitch,
                    onset_tol=0.05, offset_tol=0.05,
                )
                for ev, c in zip(tok_evts, tok_correct):
                    token_rows.append({
                        "stem": stem,
                        "chunk_idx": chunk_idx,
                        "type": ev["type"],
                        "pitch": ev["pitch"],
                        "time_sec": ev["time_sec"],
                        "logp": ev["logp"],
                        "correct": c,
                    })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "chunk_confidence_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved -> {csv_path}  ({len(df)} chunks)")

    if len(df) < 2:
        print("Not enough data points for correlation analysis.")
        return

    plot_and_summarize(df, out_dir)

    if token_rows:
        plot_token_confidence_boxplots(token_rows, out_dir)

    if args.piano_roll_n != 0 and args.pred_dir:
        generate_piano_rolls(
            df,
            pred_dir=Path(args.pred_dir),
            root=args.root,
            split=args.split,
            program_id=args.program_id,
            out_dir=out_dir,
            n_samples=max(0, args.piano_roll_n),
        )
    elif args.piano_roll_n != 0 and not args.pred_dir:
        print("WARNING: --piano_roll_n requires --pred_dir in from-scratch mode, skipping piano rolls")


def main():
    ap = argparse.ArgumentParser()
    # common
    ap.add_argument("--root", type=str, required=True, help="MAESTRO root dir")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--out_dir", type=str, default="outputs/confidence_analysis")
    ap.add_argument("--program_id", type=int, default=0)

    # Mode 2: offline from pre-saved CSV
    ap.add_argument("--confidence_csv", type=str, default=None,
                     help="pre-saved chunk_confidence.csv from infer_maestro --save_confidence")
    ap.add_argument("--pred_dir", type=str, default=None,
                     help="directory containing .pred.mid files (required with --confidence_csv)")

    # Mode 1: from scratch
    ap.add_argument("--ckpt", type=str, default=None, help="model checkpoint (required for from-scratch mode)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--step_ms", type=int, default=10)
    ap.add_argument("--max_songs", type=int, default=0, help="limit number of songs (0=all)")
    ap.add_argument("--piano_roll_n", type=int, default=0,
                     help="generate piano roll PNGs: 0=none, -1=all, N=sample N chunks spanning worst->best")
    args = ap.parse_args()

    if args.confidence_csv:
        if not args.pred_dir:
            ap.error("--pred_dir is required when using --confidence_csv")
        run_offline(args)
    else:
        if not args.ckpt:
            ap.error("--ckpt is required when not using --confidence_csv")
        run_from_scratch(args)


if __name__ == "__main__":
    main()
