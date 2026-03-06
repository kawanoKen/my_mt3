from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from my_mt3.eval import evaluate_notes_direct
from oaf.datasets import read_maestro_pairs
from oaf.eval_metrics import frame_prf, note_onset_prf, notes_to_frame_roll
from oaf.midi_utils import load_midi_notes


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _note_events_to_arrays(notes):
    if not notes:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )
    intervals = np.asarray([[n.start, n.end] for n in notes], dtype=float)
    pitches = np.asarray([n.pitch for n in notes], dtype=int)
    velocities = np.asarray([n.velocity for n in notes], dtype=int)
    order = np.argsort(intervals[:, 0])
    return intervals[order], pitches[order], velocities[order]


def _find_pred_midi(
    pred_dir: Path,
    idx: int,
    audio_path: str,
) -> Optional[Path]:
    stem = Path(audio_path).stem
    candidates = [
        pred_dir / f"{idx:04d}.mid",         # infer_maestro.py --save_midis
        pred_dir / f"{stem}.pred.mid",       # run_mt3/infer_maestro.py
        pred_dir / f"{stem}.mid",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate MAESTRO split from pre-generated MIDI predictions")
    ap.add_argument("--maestro_root", type=str, required=True, help="MAESTRO v3 root")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--pred_midi_dir", type=str, required=True, help="directory containing predicted MIDI files")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--onset_tolerance", type=float, default=0.05)
    ap.add_argument("--offset_ratio", type=float, default=0.2)
    ap.add_argument("--offset_min_tolerance", type=float, default=0.05)
    ap.add_argument("--velocity_tolerance", type=float, default=0.1)
    ap.add_argument("--hop_length", type=int, default=512, help="for frame metric time quantization")
    ap.add_argument("--sample_rate", type=int, default=16000, help="for frame metric time quantization")
    ap.add_argument("--midi_min", type=int, default=21)
    ap.add_argument("--max_items", type=int, default=0, help="0 means evaluate all")
    args = ap.parse_args()

    pred_dir = Path(args.pred_midi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pred_dir.exists():
        raise SystemExit(f"pred_midi_dir not found: {pred_dir}")

    items = read_maestro_pairs(args.maestro_root, split=args.split)
    if args.max_items > 0:
        items = items[: args.max_items]
    if not items:
        raise SystemExit(f"No MAESTRO pairs found for split={args.split}")

    dt = args.hop_length / float(args.sample_rate)
    n_pitches = 108 - args.midi_min + 1

    onset_ps: List[float] = []
    onset_rs: List[float] = []
    onset_fs: List[float] = []
    frame_ps: List[float] = []
    frame_rs: List[float] = []
    frame_fs: List[float] = []
    onset_pitch_fs: List[float] = []
    note_fs: List[float] = []
    note_vel_fs: List[float] = []
    rows: List[Dict[str, float | str | int]] = []
    missing: List[Tuple[int, str]] = []

    for idx, (audio_path, ref_midi_path) in enumerate(
        tqdm(items, desc=f"eval:{args.split}", unit="song"), start=1
    ):
        pred_midi_path = _find_pred_midi(pred_dir, idx, audio_path)
        if pred_midi_path is None:
            missing.append((idx, audio_path))
            continue

        ref_notes = load_midi_notes(ref_midi_path, apply_sustain=True)
        est_notes = load_midi_notes(str(pred_midi_path), apply_sustain=False)

        prf_on = note_onset_prf(ref_notes, est_notes, onset_tolerance=args.onset_tolerance)
        onset_ps.append(prf_on.precision)
        onset_rs.append(prf_on.recall)
        onset_fs.append(prf_on.f1)

        ref_int, ref_pitch, ref_vel = _note_events_to_arrays(ref_notes)
        est_int, est_pitch, est_vel = _note_events_to_arrays(est_notes)
        note_metrics = evaluate_notes_direct(
            ref_int, ref_pitch, ref_vel,
            est_int, est_pitch, est_vel,
            onset_tolerance=args.onset_tolerance,
            offset_ratio=args.offset_ratio,
            offset_min_tolerance=args.offset_min_tolerance,
            velocity_tolerance=args.velocity_tolerance,
        )
        onset_pitch_fs.append(float(note_metrics["onset_pitch_f"]))
        note_fs.append(float(note_metrics["note_f"]))
        note_vel_fs.append(float(note_metrics["note_vel_f"]))

        max_t = 0.0
        if len(ref_int):
            max_t = max(max_t, float(ref_int[:, 1].max()))
        if len(est_int):
            max_t = max(max_t, float(est_int[:, 1].max()))
        n_frames = max(1, int(np.ceil(max_t / dt)))

        ref_roll = notes_to_frame_roll(ref_notes, n_pitches=n_pitches, n_frames=n_frames, dt=dt, midi_min=args.midi_min)
        est_roll = notes_to_frame_roll(est_notes, n_pitches=n_pitches, n_frames=n_frames, dt=dt, midi_min=args.midi_min)
        prf_fr = frame_prf(ref_roll, est_roll)
        frame_ps.append(prf_fr.precision)
        frame_rs.append(prf_fr.recall)
        frame_fs.append(prf_fr.f1)

        rows.append(
            {
                "idx": idx,
                "audio_path": audio_path,
                "ref_midi_path": ref_midi_path,
                "pred_midi_path": str(pred_midi_path),
                "n_ref_notes": len(ref_notes),
                "n_est_notes": len(est_notes),
                "onset_p": prf_on.precision,
                "onset_r": prf_on.recall,
                "onset_f1": prf_on.f1,
                "onset_pitch_f": note_metrics["onset_pitch_f"],
                "note_f": note_metrics["note_f"],
                "note_vel_f": note_metrics["note_vel_f"],
                "frame_p": prf_fr.precision,
                "frame_r": prf_fr.recall,
                "frame_f1": prf_fr.f1,
            }
        )

    with open(out_dir / "per_piece_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "idx", "audio_path", "ref_midi_path", "pred_midi_path",
                "n_ref_notes", "n_est_notes",
                "onset_p", "onset_r", "onset_f1",
                "onset_pitch_f", "note_f", "note_vel_f",
                "frame_p", "frame_r", "frame_f1",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "maestro_root": args.maestro_root,
        "split": args.split,
        "pred_midi_dir": str(pred_dir),
        "n_items_total": len(items),
        "n_items_evaluated": len(rows),
        "n_items_missing_pred": len(missing),
        "metrics_mean": {
            "onset_p": _mean(onset_ps),
            "onset_r": _mean(onset_rs),
            "onset_f1": _mean(onset_fs),
            "onset_pitch_f": _mean(onset_pitch_fs),
            "note_f": _mean(note_fs),
            "note_vel_f": _mean(note_vel_fs),
            "frame_p": _mean(frame_ps),
            "frame_r": _mean(frame_rs),
            "frame_f1": _mean(frame_fs),
        },
        "eval_params": {
            "onset_tolerance": args.onset_tolerance,
            "offset_ratio": args.offset_ratio,
            "offset_min_tolerance": args.offset_min_tolerance,
            "velocity_tolerance": args.velocity_tolerance,
            "hop_length": args.hop_length,
            "sample_rate": args.sample_rate,
            "midi_min": args.midi_min,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if missing:
        with open(out_dir / "missing_predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "audio_path"])
            w.writerows(missing)

    m = summary["metrics_mean"]
    print(
        f"[{args.split}] n_eval={len(rows)}/{len(items)} | "
        f"onset_f1={m['onset_f1']:.4f} | "
        f"onset_pitch_f={m['onset_pitch_f']:.4f} | "
        f"note_f={m['note_f']:.4f} | "
        f"note_vel_f={m['note_vel_f']:.4f} | "
        f"frame_f1={m['frame_f1']:.4f}"
    )
    print(f"Saved: {out_dir / 'summary.json'}")
    print(f"Saved: {out_dir / 'per_piece_metrics.csv'}")
    if missing:
        print(f"Saved: {out_dir / 'missing_predictions.csv'}")


if __name__ == "__main__":
    main()
