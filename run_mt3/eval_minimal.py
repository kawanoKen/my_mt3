"""
Minimal evaluation script.
Core logic is in my_mt3/eval.py — this is a thin CLI wrapper.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from my_mt3.eval import evaluate_midi_pair


def collect_ref_midi_maps(
    maps_csv: str | Path,
    split: str = "validation",
) -> dict[str, str]:
    """Returns {stem: midi_path} for a MAPS scenario CSV split."""
    df = pd.read_csv(maps_csv)
    subset = df[df["split"] == split]
    out: dict[str, str] = {}
    for _, row in subset.iterrows():
        midi_path = Path(str(row["midi_path"]))
        if midi_path.exists():
            out[midi_path.stem] = str(midi_path)
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate a single MIDI pair or a MAPS split")
    # single-pair mode
    ap.add_argument("--ref", type=str, default=None, help="reference MIDI path")
    ap.add_argument("--est", type=str, default=None, help="estimated MIDI path")
    # dataset mode (MAPS)
    ap.add_argument("--pred_dir", type=str, default=None, help="directory containing *.pred.mid files")
    ap.add_argument("--maps_csv", type=str, default=None, help="MAPS_*_scenario.csv path")
    ap.add_argument("--maps_split", type=str, default="validation", choices=["train", "validation"])
    ap.add_argument("--out_csv", type=str, default=None, help="output CSV path (default: pred_dir/eval_metrics.csv)")
    ap.add_argument("--onset_tolerance", type=float, default=0.05)
    ap.add_argument("--offset_ratio", type=float, default=0.2)
    ap.add_argument("--offset_min_tolerance", type=float, default=0.05)
    ap.add_argument("--velocity_tolerance", type=float, default=0.1)
    ap.add_argument("--program", type=int, default=None)
    ap.add_argument("--drums", action="store_true", help="evaluate drum tracks only")
    args = ap.parse_args()

    # ---- single pair mode ----
    if args.ref and args.est:
        metrics = evaluate_midi_pair(
            args.ref,
            args.est,
            onset_tolerance=args.onset_tolerance,
            offset_ratio=args.offset_ratio,
            offset_min_tolerance=args.offset_min_tolerance,
            velocity_tolerance=args.velocity_tolerance,
            use_drums_only=args.drums,
            program=args.program,
        )
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        return

    # ---- MAPS CSV batch mode ----
    if not args.pred_dir or not args.maps_csv:
        raise SystemExit("Specify either (--ref and --est) or (--pred_dir and --maps_csv).")

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        raise SystemExit(f"pred_dir not found: {pred_dir}")

    ref_map = collect_ref_midi_maps(args.maps_csv, split=args.maps_split)
    if not ref_map:
        raise SystemExit(f"No reference MIDIs found in maps_csv for split={args.maps_split}")

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
        try:
            m = evaluate_midi_pair(
                ref_map[stem],
                str(pf),
                onset_tolerance=args.onset_tolerance,
                offset_ratio=args.offset_ratio,
                offset_min_tolerance=args.offset_min_tolerance,
                velocity_tolerance=args.velocity_tolerance,
                use_drums_only=args.drums,
                program=args.program,
            )
            m["stem"] = stem
            rows.append(m)
        except Exception as e:
            print(f"[skip] {stem}: {e}")

    print(f"\nMatched {matched}/{len(pred_files)} prediction files to MAPS references")
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
