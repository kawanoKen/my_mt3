"""
Standalone evaluation script for MAESTRO piano transcription.

Usage:
  uv run run/eval_maestro.py --pred_dir outputs/maestro_validation_pred --root maestro-v3.0.0 --split validation
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from my_mt3.eval import evaluate_midi_pair


def collect_ref_midi_maestro(
    root: str | Path,
    split: str = "validation",
) -> dict[str, str]:
    """
    Returns {stem: midi_path} for the given split from maestro-v3.0.0.csv.
    """
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    subset = df[df["split"] == split]
    out: dict[str, str] = {}
    for midi_rel in subset["midi_filename"]:
        midi_path = root / str(midi_rel)
        if midi_path.exists():
            out[midi_path.stem] = str(midi_path)
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate predicted MIDIs against MAESTRO ground truth")
    ap.add_argument("--pred_dir", type=str, required=True, help="directory containing .pred.mid files")
    ap.add_argument("--root", type=str, required=True, help="MAESTRO v3.0.0 root")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--program", type=int, default=0, help="MIDI program to evaluate (0=piano)")
    ap.add_argument("--onset_tolerance", type=float, default=0.05)
    ap.add_argument("--offset_ratio", type=float, default=0.2)
    ap.add_argument("--offset_min_tolerance", type=float, default=0.05)
    ap.add_argument("--velocity_tolerance", type=float, default=0.1)
    ap.add_argument("--out_csv", type=str, default=None, help="output CSV path (default: pred_dir/eval_metrics.csv)")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        raise SystemExit(f"pred_dir not found: {pred_dir}")

    ref_map = collect_ref_midi_maestro(args.root, split=args.split)
    if not ref_map:
        raise SystemExit(f"No reference MIDIs found for split={args.split}")

    pred_files = sorted(pred_dir.glob("*.pred.mid"))
    if not pred_files:
        raise SystemExit(f"No .pred.mid files in {pred_dir}")

    rows = []
    matched = 0
    for pf in tqdm(pred_files, desc="evaluating", unit="file"):
        stem = pf.stem.replace(".pred", "")
        if stem not in ref_map:
            continue
        matched += 1
        ref_path = ref_map[stem]
        try:
            m = evaluate_midi_pair(
                ref_path, str(pf),
                onset_tolerance=args.onset_tolerance,
                offset_ratio=args.offset_ratio,
                offset_min_tolerance=args.offset_min_tolerance,
                velocity_tolerance=args.velocity_tolerance,
                program=args.program,
            )
            m["stem"] = stem
            rows.append(m)
        except Exception as e:
            print(f"[skip] {stem}: {e}")

    print(f"\nMatched {matched}/{len(pred_files)} prediction files to references")

    if not rows:
        print("No files evaluated.")
        return

    df = pd.DataFrame(rows)
    csv_path = Path(args.out_csv) if args.out_csv else pred_dir / "eval_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"CSV -> {csv_path}")

    metric_cols = [c for c in df.columns if c != "stem"]
    summary = df[metric_cols].mean()
    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    print()


if __name__ == "__main__":
    main()
