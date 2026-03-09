"""
Token-level mask effect analysis for Note-On / Note-Off events.

What this script does:
  1) Greedy-decodes each MAESTRO chunk.
  2) Finds ON/OFF token events and labels each event as correct/incorrect
     using the same pitch+time tolerance style as analyze_confidence.py.
  3) Masks the input mel around each event time and re-scores the SAME token
     sequence with teacher forcing.
  4) Computes delta log-probability (masked - base) per token event.
  5) Plots boxplots of delta log-probability split by correct/incorrect.
  6) Plots chunk-level scatter (mean delta vs note metrics).
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
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from my_mt3.tokenizer import INPUT_FRAMES, build_vocab
from my_mt3.infer import greedy_decode_batch_with_logprobs, to_midi_from_tokens_piano
from my_mt3.eval import extract_notes_in_range, evaluate_notes_direct
from my_mt3.analysis_attribution import apply_source_mask_band

from infer_maestro import collect_pairs_maestro


def pm_to_arrays(pm: pretty_midi.PrettyMIDI):
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


def _parse_token_events(token_ids, lps, vocab, step_ms: int):
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
            lp = lps[i] if i < len(lps) else np.nan
            events.append(
                {
                    "token_pos": int(i),
                    "type": "on",
                    "pitch": int(id2on[tid]),
                    "time_sec": float(cur_ms / 1000.0),
                    "logp_decode": float(lp),
                }
            )
        elif tid in id2off:
            lp = lps[i] if i < len(lps) else np.nan
            events.append(
                {
                    "token_pos": int(i),
                    "type": "off",
                    "pitch": int(id2off[tid]),
                    "time_sec": float(cur_ms / 1000.0),
                    "logp_decode": float(lp),
                }
            )
    return events


def _match_events_to_ref(
    token_events,
    ref_int,
    ref_pitch,
    *,
    onset_tol: float = 0.05,
    offset_tol: float = 0.05,
):
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


@torch.no_grad()
def _logp_for_fixed_sequence(
    model,
    mel_1: torch.Tensor,        # [1,T,F]
    token_ids: list[int],       # generated sequence (without PRG)
    *,
    program_id: int,
    vocab,
) -> torch.Tensor:
    if len(token_ids) == 0:
        return torch.empty((0,), dtype=torch.float32, device=mel_1.device)
    prg_id = int(vocab.instrument_type[f"PRG_{int(program_id)}"])
    y_in = torch.tensor([prg_id] + [int(x) for x in token_ids[:-1]], dtype=torch.long, device=mel_1.device).unsqueeze(0)
    y_tg = torch.tensor([int(x) for x in token_ids], dtype=torch.long, device=mel_1.device).unsqueeze(0)
    mem = model.enc(mel_1)
    logits = model.dec(y_in, mem)[0]  # [S,V]
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, y_tg[0].unsqueeze(-1)).squeeze(-1)  # [S]


def plot_delta_boxplots(df_token: pd.DataFrame, out_dir: Path) -> None:
    if df_token.empty:
        print("No token rows for boxplots.")
        return
    configs = [
        ("on", "Note-On tokens", df_token[df_token["type"] == "on"]),
        ("off", "Note-Off tokens", df_token[df_token["type"] == "off"]),
        ("on_off", "Note-On + Note-Off tokens", df_token),
    ]
    for suffix, title, sub in configs:
        if sub.empty:
            continue
        groups = [
            sub.loc[sub["correct"], "delta_logp"].values,
            sub.loc[~sub["correct"], "delta_logp"].values,
        ]
        labels = [
            f"Correct (n={len(groups[0])})",
            f"Incorrect (n={len(groups[1])})",
        ]
        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            widths=0.5,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        colors = ["#4c94d6", "#e06060"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_ylabel("delta logP = logP(masked) - logP(base)")
        ax.set_title(f"Mask Effect on Token Confidence: {title}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir / f"token_mask_delta_boxplot_{suffix}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Token delta boxplot -> {fig_path}")


def plot_abs_delta_boxplots(df_token: pd.DataFrame, out_dir: Path) -> None:
    if df_token.empty:
        print("No token rows for abs-delta boxplots.")
        return
    if "abs_delta_logp" not in df_token.columns:
        print("abs_delta_logp column not found, skipping abs-delta boxplots.")
        return
    configs = [
        ("on", "Note-On tokens", df_token[df_token["type"] == "on"]),
        ("off", "Note-Off tokens", df_token[df_token["type"] == "off"]),
        ("on_off", "Note-On + Note-Off tokens", df_token),
    ]
    for suffix, title, sub in configs:
        if sub.empty:
            continue
        groups = [
            sub.loc[sub["correct"], "abs_delta_logp"].values,
            sub.loc[~sub["correct"], "abs_delta_logp"].values,
        ]
        labels = [
            f"Correct (n={len(groups[0])})",
            f"Incorrect (n={len(groups[1])})",
        ]
        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            widths=0.5,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        colors = ["#4c94d6", "#e06060"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)
        ax.set_ylabel("|delta logP| = |logP(masked) - logP(base)|")
        ax.set_title(f"Mask Effect Abs-Delta: {title}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir / f"token_mask_abs_delta_boxplot_{suffix}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Token abs-delta boxplot -> {fig_path}")


def plot_abs_delta_plus_logp_boxplots(df_token: pd.DataFrame, out_dir: Path) -> None:
    if df_token.empty:
        print("No token rows for abs-delta+logp boxplots.")
        return
    col = "abs_delta_plus_logp_base"
    if col not in df_token.columns:
        print(f"{col} column not found, skipping abs-delta+logp boxplots.")
        return
    configs = [
        ("on", "Note-On tokens", df_token[df_token["type"] == "on"]),
        ("off", "Note-Off tokens", df_token[df_token["type"] == "off"]),
        ("on_off", "Note-On + Note-Off tokens", df_token),
    ]
    for suffix, title, sub in configs:
        if sub.empty:
            continue
        groups = [
            sub.loc[sub["correct"], col].values,
            sub.loc[~sub["correct"], col].values,
        ]
        labels = [
            f"Correct (n={len(groups[0])})",
            f"Incorrect (n={len(groups[1])})",
        ]
        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            widths=0.5,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        colors = ["#4c94d6", "#e06060"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)
        ax.set_ylabel("|delta logP| + logP(base)")
        ax.set_title(f"Mask Effect Combined Score: {title}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir / f"token_mask_abs_delta_plus_logp_boxplot_{suffix}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Token abs-delta+logp boxplot -> {fig_path}")


def plot_chunk_scatter(df_chunk: pd.DataFrame, out_dir: Path) -> None:
    if df_chunk.empty:
        return
    if "mean_delta_logp_all" not in df_chunk.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [("note_f", "Note F1"), ("onset_f", "Onset F1")]
    for ax, (col, label) in zip(axes, pairs):
        if col not in df_chunk.columns:
            continue
        x = df_chunk["mean_delta_logp_all"].to_numpy()
        y = df_chunk[col].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=10, alpha=0.25, edgecolors="none")
        ax.set_xlabel("mean delta logP (all on/off tokens)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if m.sum() >= 3:
            corr = np.corrcoef(x[m], y[m])[0, 1]
            ax.set_title(f"{label} vs mean delta logP (r={corr:.3f})")
        else:
            ax.set_title(f"{label} vs mean delta logP")
    fig.tight_layout()
    fig_path = out_dir / "mask_delta_vs_note_metrics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chunk scatter -> {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--out_dir", type=str, default="outputs/mask_token_effect")
    ap.add_argument("--program_id", type=int, default=0)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--step_ms", type=int, default=10)
    ap.add_argument("--max_songs", type=int, default=0)

    ap.add_argument("--mask_width_ratio", type=float, default=0.2)
    ap.add_argument("--mask_fill", type=str, default="zero", choices=["zero", "mean"])
    ap.add_argument("--onset_tol", type=float, default=0.05)
    ap.add_argument("--offset_tol", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from my_mt3.model import MT3Mini
    from my_mt3.audio import load_audio_mono
    from my_mt3.dataset import LogMelCfg, LogMelExtractor

    vocab = build_vocab(input_frames=INPUT_FRAMES, instrument_type="piano", include_note_off=True)
    model = MT3Mini(vocab_size=len(vocab.itos), n_mels=args.n_mels).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    pairs = collect_pairs_maestro(args.root, split=args.split, program_id=args.program_id)
    if args.max_songs > 0:
        pairs = pairs[:args.max_songs]
    print(f"Songs: {len(pairs)}")

    feat = LogMelExtractor(LogMelCfg(sr=args.sr, n_fft=args.n_fft, hop=args.hop, n_mels=args.n_mels))
    need_samples = (INPUT_FRAMES - 1) * args.hop + args.n_fft
    chunk_sec = need_samples / float(args.sr)

    token_rows = []
    chunk_rows = []

    for song_idx, (audio_path, midi_path, pid) in enumerate(tqdm(pairs, desc="songs")):
        stem = Path(audio_path).stem
        ref_pm = pretty_midi.PrettyMIDI(midi_path)
        y, _ = load_audio_mono(audio_path, sr=args.sr)
        total_samples = len(y)
        stride_samples = INPUT_FRAMES * args.hop

        starts = list(range(0, max(0, total_samples - need_samples) + 1, stride_samples))
        if len(starts) == 0:
            starts = [0]

        mel_list = []
        for ss in starts:
            y_seg = y[ss:ss + need_samples]
            if len(y_seg) < need_samples:
                y_seg = np.pad(y_seg, (0, need_samples - len(y_seg)))
            mel = feat(y_seg)
            if mel.shape[0] > INPUT_FRAMES:
                mel = mel[:INPUT_FRAMES]
            elif mel.shape[0] < INPUT_FRAMES:
                mel = np.pad(mel, ((0, INPUT_FRAMES - mel.shape[0]), (0, 0)))
            mel_list.append(mel.astype(np.float32, copy=False))

        for b0 in range(0, len(starts), args.batch_size):
            b1 = min(len(starts), b0 + args.batch_size)
            mel_np = np.stack(mel_list[b0:b1])
            mels_bt = torch.from_numpy(mel_np).to(device=args.device, dtype=torch.float32)

            with torch.no_grad():
                tok_batch, lp_batch = greedy_decode_batch_with_logprobs(
                    model, mels_bt, max_len=args.max_len, device=args.device, program_id=int(pid), vocab=vocab
                )

            for local_i in range(len(tok_batch)):
                chunk_idx = b0 + local_i
                ss = starts[chunk_idx]
                t0 = ss / float(args.sr)
                t1 = t0 + chunk_sec
                token_ids = tok_batch[local_i]
                decode_lps = lp_batch[local_i]
                if len(token_ids) == 0:
                    continue

                res = to_midi_from_tokens_piano(token_ids, program_id=int(pid), step_ms=args.step_ms, vocab=vocab)
                est_int, est_pitch, est_vel = pm_to_arrays(res.pm)
                ref_int, ref_pitch, ref_vel = extract_notes_in_range(ref_pm, t0, t1, program=int(pid))
                try:
                    m = evaluate_notes_direct(ref_int, ref_pitch, ref_vel, est_int, est_pitch, est_vel)
                except Exception:
                    continue

                mel_1 = mels_bt[local_i:local_i + 1]
                base_tf_logp = _logp_for_fixed_sequence(
                    model, mel_1, token_ids, program_id=int(pid), vocab=vocab
                )
                token_events = _parse_token_events(token_ids, decode_lps, vocab, step_ms=args.step_ms)
                tok_correct = _match_events_to_ref(
                    token_events,
                    ref_int,
                    ref_pitch,
                    onset_tol=float(args.onset_tol),
                    offset_tol=float(args.offset_tol),
                )
                if len(token_events) == 0:
                    continue

                frame2logp = {}
                event_delta_vals = []
                event_combined_vals = []
                for ev, is_correct in zip(token_events, tok_correct):
                    token_pos = int(ev["token_pos"])
                    if token_pos >= int(base_tf_logp.numel()):
                        continue
                    frame = int(round(float(ev["time_sec"]) * float(args.sr) / float(args.hop)))
                    frame = max(0, min(frame, INPUT_FRAMES - 1))
                    if frame not in frame2logp:
                        mel_mask = apply_source_mask_band(
                            mel_1,
                            center_frame=frame,
                            width_ratio=float(args.mask_width_ratio),
                            fill="mean" if args.mask_fill == "mean" else "zero",
                        )
                        frame2logp[frame] = _logp_for_fixed_sequence(
                            model, mel_mask, token_ids, program_id=int(pid), vocab=vocab
                        )
                    masked_tf_logp = frame2logp[frame]
                    if token_pos >= int(masked_tf_logp.numel()):
                        continue

                    logp_base = float(base_tf_logp[token_pos].item())
                    logp_mask = float(masked_tf_logp[token_pos].item())
                    delta = float(logp_mask - logp_base)
                    abs_delta = float(abs(delta))
                    combined = float(abs_delta + logp_base)
                    event_delta_vals.append(delta)
                    event_combined_vals.append(combined)
                    token_rows.append(
                        {
                            "stem": stem,
                            "song_idx": int(song_idx),
                            "chunk_idx": int(chunk_idx),
                            "t0": float(t0),
                            "t1": float(t1),
                            "type": ev["type"],
                            "pitch": int(ev["pitch"]),
                            "time_sec": float(ev["time_sec"]),
                            "token_pos": token_pos,
                            "correct": bool(is_correct),
                            "logp_base": logp_base,
                            "logp_masked": logp_mask,
                            "delta_logp": delta,
                            "abs_delta_logp": abs_delta,
                            "abs_delta_plus_logp_base": combined,
                            "logp_decode": float(ev["logp_decode"]),
                            "mask_width_ratio": float(args.mask_width_ratio),
                            "mask_fill": str(args.mask_fill),
                            "onset_f": float(m.get("onset_f", np.nan)),
                            "note_f": float(m.get("note_f", np.nan)),
                            "onset_pitch_f": float(m.get("onset_pitch_f", np.nan)),
                            "note_vel_f": float(m.get("note_vel_f", np.nan)),
                        }
                    )

                if event_delta_vals:
                    vals = np.asarray(event_delta_vals, dtype=float)
                    vals_combined = np.asarray(event_combined_vals, dtype=float)
                    chunk_rows.append(
                        {
                            "stem": stem,
                            "song_idx": int(song_idx),
                            "chunk_idx": int(chunk_idx),
                            "t0": float(t0),
                            "t1": float(t1),
                            "n_events": int(len(vals)),
                            "mean_delta_logp_all": float(np.mean(vals)),
                            "median_delta_logp_all": float(np.median(vals)),
                            "mean_abs_delta_plus_logp_base_all": float(np.mean(vals_combined)),
                            "onset_f": float(m.get("onset_f", np.nan)),
                            "note_f": float(m.get("note_f", np.nan)),
                            "onset_pitch_f": float(m.get("onset_pitch_f", np.nan)),
                            "note_vel_f": float(m.get("note_vel_f", np.nan)),
                        }
                    )

    df_token = pd.DataFrame(token_rows)
    tok_csv = out_dir / "token_mask_effect.csv"
    df_token.to_csv(tok_csv, index=False)
    print(f"Token CSV -> {tok_csv}  ({len(df_token)} rows)")

    df_chunk = pd.DataFrame(chunk_rows)
    chunk_csv = out_dir / "chunk_mask_effect_metrics.csv"
    df_chunk.to_csv(chunk_csv, index=False)
    print(f"Chunk CSV -> {chunk_csv}  ({len(df_chunk)} chunks)")

    if not df_token.empty:
        plot_delta_boxplots(df_token, out_dir)
        plot_abs_delta_boxplots(df_token, out_dir)
        plot_abs_delta_plus_logp_boxplots(df_token, out_dir)
    if not df_chunk.empty:
        plot_chunk_scatter(df_chunk, out_dir)


if __name__ == "__main__":
    main()

