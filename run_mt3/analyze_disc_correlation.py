#!/usr/bin/env python3
"""
Discriminator confidence vs evaluation metrics correlation analysis.

Given:
  - chunk_confidence_metrics.csv (from analyze_confidence.py)
  - predicted MIDI files (*.pred.mid)
  - a trained Discriminator checkpoint

For each chunk, extract the predicted notes in [t0, t1], tokenize them with
the Discriminator's tokenizer, score with the Discriminator, then correlate
with evaluation metrics and plot scatter charts.

Usage:
  python run/analyze_disc_correlation.py \
    --metrics_csv outputs/confidence_analysis/chunk_confidence_metrics.csv \
    --pred_dir outputs/maestro_val_pred_giantMIDI \
    --disc_ckpt Discriminator/ckpt_conf_giantmidi/conf_step1500.pt \
    --out_dir outputs/disc_correlation
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "Discriminator"))

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import pretty_midi

from midi_tokenizer import MidiTokCfg, midi_notes_to_tokens
from conf_data import pad_and_add_cls
from conf_clf_model import ConfClfCfg, TransformerConfidenceClf


def load_disc_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]

    tok_cfg = MidiTokCfg(
        time_step_sec=float(cfg.get("time_step_sec", 0.01)),
        max_shift_steps=int(cfg.get("max_shift_steps", 100)),
    )
    vocab_size = int(cfg.get("vocab_size", tok_cfg.vocab_size()))

    mcfg = ConfClfCfg(
        vocab_size=vocab_size,
        max_len=int(cfg.get("max_len", 512)),
        d_model=int(cfg.get("d_model", 256)),
        n_layers=int(cfg.get("n_layers", 6)),
        n_heads=int(cfg.get("n_heads", 8)),
        d_ff=int(cfg.get("d_ff", 1024)),
        dropout=0.0,
        pad_id=tok_cfg.pad_id,
        cls_id=tok_cfg.cls_id,
    )
    model = TransformerConfidenceClf(mcfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, tok_cfg


def extract_notes_in_range(pm: pretty_midi.PrettyMIDI, t0: float, t1: float) -> List[pretty_midi.Note]:
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= t0 and n.start < t1:
                notes.append(pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=n.start - t0,
                    end=min(n.end, t1) - t0,
                ))
    return notes


@torch.no_grad()
def score_chunk_midi(
    model: TransformerConfidenceClf,
    tok_cfg: MidiTokCfg,
    notes: List[pretty_midi.Note],
    device: str,
) -> float:
    if len(notes) == 0:
        return 0.0

    seq = midi_notes_to_tokens(notes, tok_cfg)
    if seq.numel() == 0:
        return 0.0

    max_len = model.cfg.max_len
    seq_trunc = seq[:max_len - 1]

    tokens, attn = pad_and_add_cls(
        seq_trunc, max_len=max_len, pad_id=tok_cfg.pad_id, cls_id=tok_cfg.cls_id
    )
    tokens = tokens.unsqueeze(0).to(device)
    attn = attn.unsqueeze(0).to(device)
    return float(model.score(tokens, attn_mask=attn)[0].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, required=True,
                     help="chunk_confidence_metrics.csv with evaluation columns")
    ap.add_argument("--pred_dir", type=str, required=True,
                     help="directory with *.pred.mid files")
    ap.add_argument("--disc_ckpt", type=str, required=True,
                     help="Discriminator checkpoint (.pt)")
    ap.add_argument("--out_dir", type=str, default="outputs/disc_correlation")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading discriminator: {args.disc_ckpt}")
    model, tok_cfg = load_disc_model(args.disc_ckpt, args.device)

    print(f"Loading metrics: {args.metrics_csv}")
    df = pd.read_csv(args.metrics_csv)
    print(f"  {len(df)} chunks, columns: {list(df.columns)}")

    pred_dir = Path(args.pred_dir)
    midi_cache = {}

    disc_scores = []
    for idx, row in df.iterrows():
        stem = row["stem"]
        t0, t1 = float(row["t0"]), float(row["t1"])

        midi_path = pred_dir / f"{stem}.pred.mid"
        if not midi_path.exists():
            disc_scores.append(np.nan)
            continue

        if stem not in midi_cache:
            try:
                midi_cache[stem] = pretty_midi.PrettyMIDI(str(midi_path))
            except Exception:
                disc_scores.append(np.nan)
                continue

        pm = midi_cache[stem]
        notes = extract_notes_in_range(pm, t0, t1)
        score = score_chunk_midi(model, tok_cfg, notes, args.device)
        disc_scores.append(score)

        if (idx + 1) % 2000 == 0:
            print(f"  scored {idx+1}/{len(df)} chunks...")

    df["disc_score"] = disc_scores
    df_valid = df.dropna(subset=["disc_score"])

    csv_out = out_dir / "disc_metrics.csv"
    df_valid.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out} ({len(df_valid)} rows)")

    # Correlation & scatter plots
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_keys = [k for k in ["onset_f", "onset_pitch_f", "note_f", "note_vel_f"] if k in df_valid.columns]
    conf_keys = ["disc_score"]
    if "log_pyx_norm" in df_valid.columns:
        conf_keys.append("log_pyx_norm")

    summary_rows = []

    for ck in conf_keys:
        for mk in metric_keys:
            x = df_valid[ck].values
            y = df_valid[mk].values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) < 10:
                continue

            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)
            summary_rows.append({
                "confidence": ck, "metric": mk,
                "pearson_r": pearson_r, "pearson_p": pearson_p,
                "spearman_r": spearman_r, "spearman_p": spearman_p,
                "n": len(x),
            })

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(x, y, alpha=0.15, s=8, edgecolors="none")
            ax.set_xlabel(ck)
            ax.set_ylabel(mk)
            ax.set_title(f"{ck} vs {mk}\nPearson={pearson_r:.3f}  Spearman={spearman_r:.3f}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = out_dir / f"scatter_{ck}_vs_{mk}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "correlation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCorrelation summary -> {summary_path}")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()
