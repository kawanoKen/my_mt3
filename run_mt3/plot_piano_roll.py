from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def load_note_arrays(
    midi_path: str | Path,
    *,
    program: int = 0,
    include_drums: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    intervals: list[list[float]] = []
    pitches: list[int] = []
    for inst in pm.instruments:
        if not include_drums and inst.is_drum:
            continue
        if (not inst.is_drum) and (program is not None) and (inst.program != int(program)):
            continue
        for n in inst.notes:
            intervals.append([float(n.start), float(n.end)])
            pitches.append(int(n.pitch))
    if not intervals:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)
    arr_i = np.asarray(intervals, dtype=float)
    arr_p = np.asarray(pitches, dtype=int)
    order = np.argsort(arr_i[:, 0])
    return arr_i[order], arr_p[order]


def crop_note_arrays(
    intervals: np.ndarray,
    pitches: np.ndarray,
    *,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(intervals) == 0:
        return intervals, pitches
    if end_sec <= start_sec:
        keep = intervals[:, 1] > start_sec
        out_i = intervals[keep].copy()
        out_p = pitches[keep].copy()
        out_i[:, 0] = np.maximum(out_i[:, 0], start_sec) - start_sec
        out_i[:, 1] = np.maximum(out_i[:, 1], start_sec) - start_sec
        return out_i, out_p
    keep = (intervals[:, 1] > start_sec) & (intervals[:, 0] < end_sec)
    out_i = intervals[keep].copy()
    out_p = pitches[keep].copy()
    out_i[:, 0] = np.clip(out_i[:, 0], start_sec, end_sec) - start_sec
    out_i[:, 1] = np.clip(out_i[:, 1], start_sec, end_sec) - start_sec
    return out_i, out_p


def collect_ref_midi_maestro(root: str | Path, split: str = "validation") -> dict[str, Path]:
    root = Path(root)
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
            midi_path = root / midi_rel
            if midi_path.exists():
                out[midi_path.stem] = midi_path
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple MIDI piano-roll plotter")
    ap.add_argument("--midi", type=str, default=None, help="target MIDI (.mid) for single-file mode")
    ap.add_argument("--out_png", type=str, default=None, help="output PNG path for single-file mode")
    ap.add_argument("--midi_dir", type=str, default=None, help="input directory for batch mode")
    ap.add_argument("--out_dir", type=str, default=None, help="output directory for batch mode")
    ap.add_argument("--glob", type=str, default="*.pred.mid", help="glob pattern in --midi_dir (e.g., *.mid)")
    ap.add_argument("--ref_midi", type=str, default=None, help="optional reference MIDI to overlay")
    ap.add_argument("--ref_dir", type=str, default=None,
                    help="optional reference MIDI directory in batch mode (match by stem)")
    ap.add_argument("--maestro_root", type=str, default=None,
                    help="optional MAESTRO root for automatic GT matching by stem in batch mode")
    ap.add_argument("--split", type=str, default="validation",
                    choices=["train", "validation", "test"],
                    help="MAESTRO split used with --maestro_root")
    ap.add_argument("--program", type=int, default=0, help="MIDI program filter (0=piano)")
    ap.add_argument("--include_drums", action="store_true", help="include drum tracks")
    ap.add_argument("--fs", type=int, default=100, help="time resolution (frames/sec)")
    ap.add_argument("--start_sec", type=float, default=0.0, help="start time (sec)")
    ap.add_argument("--end_sec", type=float, default=0.0, help="end time (sec), 0=full")
    args = ap.parse_args()

    use_single = bool(args.midi)
    use_batch = bool(args.midi_dir)
    if use_single == use_batch:
        raise SystemExit("Specify exactly one mode: (--midi, --out_png) or (--midi_dir, --out_dir)")
    if use_single and not args.out_png:
        raise SystemExit("--out_png is required in single-file mode")
    if use_batch and not args.out_dir:
        raise SystemExit("--out_dir is required in batch mode")

    def _plot_one(pred_midi_path: Path, out_png_path: Path, ref_midi_path: Path | None):
        pred_int, pred_pitch = load_note_arrays(
            pred_midi_path,
            program=args.program,
            include_drums=args.include_drums,
        )
        ref_int = np.zeros((0, 2), dtype=float)
        ref_pitch = np.zeros((0,), dtype=int)
        if ref_midi_path is not None:
            ref_int, ref_pitch = load_note_arrays(
                ref_midi_path,
                program=args.program,
                include_drums=args.include_drums,
            )

        if args.start_sec > 0 or args.end_sec > 0:
            pred_int, pred_pitch = crop_note_arrays(
                pred_int, pred_pitch, start_sec=float(args.start_sec), end_sec=float(args.end_sec)
            )
            if ref_midi_path is not None:
                ref_int, ref_pitch = crop_note_arrays(
                    ref_int, ref_pitch, start_sec=float(args.start_sec), end_sec=float(args.end_sec)
                )

        out_png_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        note_h = 0.8

        if ref_midi_path is not None:
            for i in range(len(ref_int)):
                onset, offset = ref_int[i]
                p = ref_pitch[i]
                ax.add_patch(Rectangle(
                    (onset, p - note_h / 2), max(1e-4, offset - onset), note_h,
                    facecolor="royalblue", edgecolor="navy", alpha=0.5, linewidth=0.5,
                ))

        for i in range(len(pred_int)):
            onset, offset = pred_int[i]
            p = pred_pitch[i]
            ax.add_patch(Rectangle(
                (onset, p - note_h / 2), max(1e-4, offset - onset), note_h,
                facecolor="crimson", edgecolor="darkred", alpha=0.5, linewidth=0.5,
            ))

        if (len(ref_pitch) + len(pred_pitch)) > 0:
            all_pitches = np.concatenate([ref_pitch, pred_pitch]) if len(ref_pitch) > 0 else pred_pitch
            pmin, pmax = int(all_pitches.min()) - 2, int(all_pitches.max()) + 2
        else:
            pmin, pmax = 58, 62

        if len(pred_int) > 0:
            max_t = float(pred_int[:, 1].max())
        else:
            max_t = 0.0
        if len(ref_int) > 0:
            max_t = max(max_t, float(ref_int[:, 1].max()))
        max_t = max(max_t, 0.1)

        ax.set_xlim(0.0, max_t)
        ax.set_ylim(pmin, pmax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI Pitch")
        if ref_midi_path is None:
            ax.set_title(f"Piano Roll: {pred_midi_path.name}", fontsize=10)
            legend_elems = [
                Line2D([0], [0], color="crimson", lw=6, alpha=0.5, label=f"Pred ({len(pred_pitch)} notes)"),
            ]
        else:
            ax.set_title("Piano Roll Overlay  red=pred, blue=ref", fontsize=10)
            legend_elems = [
                Line2D([0], [0], color="royalblue", lw=6, alpha=0.5, label=f"GT ({len(ref_pitch)} notes)"),
                Line2D([0], [0], color="crimson", lw=6, alpha=0.5, label=f"Pred ({len(pred_pitch)} notes)"),
            ]
        ax.legend(handles=legend_elems, loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out_png_path}")

    if use_single:
        ref_path = Path(args.ref_midi) if args.ref_midi else None
        _plot_one(Path(args.midi), Path(args.out_png), ref_path)
        return

    midi_dir = Path(args.midi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_midis = sorted(midi_dir.glob(args.glob))
    if not pred_midis:
        raise SystemExit(f"No MIDI files found: dir={midi_dir}, glob={args.glob}")

    ref_dir = Path(args.ref_dir) if args.ref_dir else None
    maestro_ref_map: dict[str, Path] = {}
    if ref_dir is None and args.maestro_root:
        maestro_ref_map = collect_ref_midi_maestro(args.maestro_root, split=args.split)
        print(f"MAESTRO refs loaded: {len(maestro_ref_map)} ({args.split})")
    n_ok = 0
    n_ref_hit = 0
    for pred_path in pred_midis:
        stem = pred_path.stem
        if stem.endswith(".pred"):
            stem = stem[:-5]
        out_png = out_dir / f"{stem}.png"
        ref_path = None
        if ref_dir is not None:
            cand_mid = ref_dir / f"{stem}.mid"
            cand_midi = ref_dir / f"{stem}.midi"
            if cand_mid.exists():
                ref_path = cand_mid
            elif cand_midi.exists():
                ref_path = cand_midi
        elif maestro_ref_map:
            ref_path = maestro_ref_map.get(stem)
        if ref_path is not None:
            n_ref_hit += 1
        _plot_one(pred_path, out_png, ref_path)
        n_ok += 1
    print(f"done: {n_ok} files -> {out_dir}  (with_ref={n_ref_hit})")


if __name__ == "__main__":
    main()


# uv run run_mt3/plot_piano_roll.py \
#   --midi_dir "outputs/unsup10p_10000e_chunk&note_filtered_313" \
#   --glob "*.pred.mid" \
#   --maestro_root "dataset/maestro-v3.0.0" \
#   --split validation \
#   --out_dir "outputs/unsup10p_10000e_chunk&note_filtered_313/piano_rolls_overlay"