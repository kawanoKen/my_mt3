from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pretty_midi


def _load_notes(mid_path: Path) -> list[tuple[int, float, float]]:
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    notes: list[tuple[int, float, float]] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((int(n.pitch), float(n.start), float(n.end)))
    return notes


def _bucket_ms(values_sec: np.ndarray) -> dict[str, int]:
    ms = values_sec * 1000.0
    return {
        "0_50": int(((ms >= 0) & (ms < 50)).sum()),
        "50_100": int(((ms >= 50) & (ms < 100)).sum()),
        "100_200": int(((ms >= 100) & (ms < 200)).sum()),
        "200_plus": int((ms >= 200).sum()),
    }


def _summary_ms(values_sec: np.ndarray) -> dict[str, float | int]:
    if values_sec.size == 0:
        return {"n": 0}
    return {
        "n": int(values_sec.size),
        "mean_ms": float(values_sec.mean() * 1000.0),
        "median_ms": float(np.quantile(values_sec, 0.5) * 1000.0),
        "p90_ms": float(np.quantile(values_sec, 0.9) * 1000.0),
        "p99_ms": float(np.quantile(values_sec, 0.99) * 1000.0),
    }


def _stats_for_dir(pred_dir: Path, stems: set[str]) -> dict:
    gaps: list[float] = []
    durs: list[float] = []

    for stem in stems:
        pred_path = pred_dir / f"{stem}.pred.mid"
        notes = _load_notes(pred_path)

        by_pitch: dict[int, list[float]] = {}
        for p, start, end in notes:
            durs.append(max(0.0, end - start))
            by_pitch.setdefault(p, []).append(start)

        for starts in by_pitch.values():
            starts.sort()
            if len(starts) >= 2:
                gaps.extend(np.diff(np.array(starts)).tolist())

    gaps_arr = np.array(gaps, dtype=np.float64)
    durs_arr = np.array(durs, dtype=np.float64)
    return {
        "same_pitch_onset_gap": {
            "summary": _summary_ms(gaps_arr),
            "buckets": _bucket_ms(gaps_arr),
        },
        "note_duration": {
            "summary": _summary_ms(durs_arr),
            "buckets": _bucket_ms(durs_arr),
        },
    }


def _bucket_percentages(stats: dict, key: str) -> dict[str, float]:
    n = int(stats[key]["summary"]["n"])
    if n <= 0:
        return {k: 0.0 for k in ["0_50", "50_100", "100_200", "200_plus"]}
    out: dict[str, float] = {}
    for k, v in stats[key]["buckets"].items():
        out[k] = 100.0 * float(v) / float(n)
    return out


def _print_report(label: str, stats: dict) -> None:
    print(f"\n=== {label} ===")
    for key in ["same_pitch_onset_gap", "note_duration"]:
        s = stats[key]["summary"]
        b = stats[key]["buckets"]
        p = _bucket_percentages(stats, key)
        print(f"[{key}] n={s['n']} mean={s['mean_ms']:.2f}ms median={s['median_ms']:.2f}ms")
        print(
            "  "
            f"0-50={b['0_50']} ({p['0_50']:.2f}%)  "
            f"50-100={b['50_100']} ({p['50_100']:.2f}%)  "
            f"100-200={b['100_200']} ({p['100_200']:.2f}%)  "
            f"200+={b['200_plus']} ({p['200_plus']:.2f}%)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare two prediction directories with (1) same-pitch onset spacing and "
            "(2) note duration distributions."
        )
    )
    ap.add_argument("--pred_a", type=str, required=True, help="first directory with *.pred.mid")
    ap.add_argument("--label_a", type=str, default="A")
    ap.add_argument("--pred_b", type=str, required=True, help="second directory with *.pred.mid")
    ap.add_argument("--label_b", type=str, default="B")
    ap.add_argument(
        "--out_json",
        type=str,
        default="outputs/note_spacing_duration_compare.json",
        help="path to save detailed result json",
    )
    args = ap.parse_args()

    pred_a = Path(args.pred_a)
    pred_b = Path(args.pred_b)
    if not pred_a.is_dir() or not pred_b.is_dir():
        raise SystemExit("Both --pred_a and --pred_b must be existing directories.")

    stems_a = {p.name[:-9] for p in pred_a.glob("*.pred.mid")}
    stems_b = {p.name[:-9] for p in pred_b.glob("*.pred.mid")}
    common = sorted(stems_a & stems_b)
    if not common:
        raise SystemExit("No common *.pred.mid stems found between the two dirs.")
    common_set = set(common)

    stats_a = _stats_for_dir(pred_a, common_set)
    stats_b = _stats_for_dir(pred_b, common_set)

    _print_report(args.label_a, stats_a)
    _print_report(args.label_b, stats_b)

    out = {
        "pred_a": str(pred_a),
        "label_a": args.label_a,
        "pred_b": str(pred_b),
        "label_b": args.label_b,
        "common_files": len(common),
        "stats_a": stats_a,
        "stats_b": stats_b,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ncommon_files={len(common)}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
