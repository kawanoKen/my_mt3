from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pretty_midi


def _maestro_ref_map(root: Path, split: str) -> dict[str, Path]:
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    out: dict[str, Path] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("split") != split:
                continue
            midi_rel = row.get("midi_filename", "")
            if not midi_rel:
                continue
            p = root / midi_rel
            if p.exists():
                out[p.stem] = p
    return out


def _load_notes(mid_path: Path) -> list[tuple[int, float, float]]:
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    notes: list[tuple[int, float, float]] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((int(n.pitch), float(n.start), float(n.end)))
    return notes


def _overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _count_overlaps_per_gt(
    gt_notes: list[tuple[int, float, float]],
    pred_notes: list[tuple[int, float, float]],
    *,
    min_overlap_sec: float,
) -> np.ndarray:
    by_pitch_pred: dict[int, list[tuple[float, float]]] = {}
    for p, s, e in pred_notes:
        by_pitch_pred.setdefault(p, []).append((s, e))

    counts: list[int] = []
    for p, g0, g1 in gt_notes:
        c = 0
        for p0, p1 in by_pitch_pred.get(p, []):
            if _overlap_len(g0, g1, p0, p1) >= min_overlap_sec:
                c += 1
        counts.append(c)
    return np.asarray(counts, dtype=np.int32)


def _analyze_dir(
    pred_dir: Path,
    stems: list[str],
    ref_map: dict[str, Path],
    *,
    min_overlap_sec: float,
) -> dict:
    overlap_counts_all: list[np.ndarray] = []
    n_pred_total = 0
    n_gt_total = 0

    for stem in stems:
        pred_path = pred_dir / f"{stem}.pred.mid"
        gt_path = ref_map[stem]
        pred_notes = _load_notes(pred_path)
        gt_notes = _load_notes(gt_path)
        n_pred_total += len(pred_notes)
        n_gt_total += len(gt_notes)
        overlap_counts_all.append(
            _count_overlaps_per_gt(gt_notes, pred_notes, min_overlap_sec=min_overlap_sec)
        )

    if overlap_counts_all:
        counts = np.concatenate(overlap_counts_all)
    else:
        counts = np.zeros((0,), dtype=np.int32)

    n = int(counts.size)
    if n == 0:
        return {
            "gt_notes": 0,
            "pred_notes": n_pred_total,
            "mean_pred_per_gt": 0.0,
            "rates": {},
            "counts": {},
        }

    c0 = int((counts == 0).sum())  # miss
    c1 = int((counts == 1).sum())  # one-to-one
    c2p = int((counts >= 2).sum())  # split
    c3p = int((counts >= 3).sum())

    return {
        "gt_notes": n_gt_total,
        "pred_notes": n_pred_total,
        "mean_pred_per_gt": float(counts.mean()),
        "counts": {
            "miss_0": c0,
            "exact_1": c1,
            "split_2plus": c2p,
            "split_3plus": c3p,
        },
        "rates": {
            "miss_0": float(c0 / n),
            "exact_1": float(c1 / n),
            "split_2plus": float(c2p / n),
            "split_3plus": float(c3p / n),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze note split against GT: how many predicted notes overlap each GT note."
    )
    ap.add_argument("--pred_a", type=str, required=True)
    ap.add_argument("--label_a", type=str, default="A")
    ap.add_argument("--pred_b", type=str, required=True)
    ap.add_argument("--label_b", type=str, default="B")
    ap.add_argument("--maestro_root", type=str, default="dataset/maestro-v3.0.0")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--min_overlap_ms", type=float, default=1.0, help="minimum overlap to count as match")
    ap.add_argument("--out_json", type=str, default="outputs/note_split_vs_gt.json")
    args = ap.parse_args()

    pred_a = Path(args.pred_a)
    pred_b = Path(args.pred_b)
    if not pred_a.is_dir() or not pred_b.is_dir():
        raise SystemExit("Both prediction directories must exist.")

    ref_map = _maestro_ref_map(Path(args.maestro_root), split=args.split)
    stems_a = {p.name[:-9] for p in pred_a.glob("*.pred.mid")}
    stems_b = {p.name[:-9] for p in pred_b.glob("*.pred.mid")}
    stems = sorted((stems_a & stems_b) & set(ref_map.keys()))
    if not stems:
        raise SystemExit("No common stems among pred_a, pred_b, and MAESTRO references.")

    min_overlap_sec = float(args.min_overlap_ms) / 1000.0
    a = _analyze_dir(pred_a, stems, ref_map, min_overlap_sec=min_overlap_sec)
    b = _analyze_dir(pred_b, stems, ref_map, min_overlap_sec=min_overlap_sec)

    print(f"common_files={len(stems)}")
    for label, res in [(args.label_a, a), (args.label_b, b)]:
        print(f"\n=== {label} ===")
        print(f"gt_notes={res['gt_notes']} pred_notes={res['pred_notes']} mean_pred_per_gt={res['mean_pred_per_gt']:.4f}")
        print(
            "rates: "
            f"miss_0={res['rates']['miss_0']:.4%}  "
            f"exact_1={res['rates']['exact_1']:.4%}  "
            f"split_2plus={res['rates']['split_2plus']:.4%}  "
            f"split_3plus={res['rates']['split_3plus']:.4%}"
        )

    out = {
        "common_files": len(stems),
        "min_overlap_ms": float(args.min_overlap_ms),
        "pred_a": str(pred_a),
        "label_a": args.label_a,
        "stats_a": a,
        "pred_b": str(pred_b),
        "label_b": args.label_b,
        "stats_b": b,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
