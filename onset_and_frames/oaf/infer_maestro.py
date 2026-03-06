from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from my_mt3.eval import evaluate_notes_direct
from oaf.config import DecodeConfig, FeatureConfig, LabelConfig, ModelConfig
from oaf.datasets import read_maestro_pairs, read_split_labeled_csv
from oaf.decoding import decode_notes, save_notes_as_midi
from oaf.eval_metrics import frame_prf, note_onset_prf, notes_to_frame_roll
from oaf.features import LogMelSpec, load_audio_mono, ensure_wave_cache
from oaf.midi_utils import load_midi_notes
from oaf.model import OnsetsAndFrames


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _resolve_head_flag(mode: str, inferred: bool) -> bool:
    m = mode.lower()
    if m == "auto":
        return inferred
    if m == "true":
        return True
    if m == "false":
        return False
    raise ValueError(f"invalid mode: {mode}")


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


def main() -> None:
    ap = argparse.ArgumentParser(description="MAESTRO inference + evaluation for OAF model")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--maestro_root", type=str, default=None, help="MAESTRO v3 root")
    ap.add_argument("--pairs_csv", type=str, default=None,
                    help="split-aware CSV with columns: split,audio_path,midi_path (e.g., MAPS_Full_scenario.csv)")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tau_on", type=float, default=0.5)
    ap.add_argument("--tau_fr", type=float, default=0.5)
    ap.add_argument("--no_onset_gating", action="store_true")
    ap.add_argument("--onset_tolerance", type=float, default=0.05)
    ap.add_argument("--offset_ratio", type=float, default=0.2)
    ap.add_argument("--offset_min_tolerance", type=float, default=0.05)
    ap.add_argument("--velocity_tolerance", type=float, default=0.1)
    ap.add_argument("--model_use_offset_head", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--model_use_velocity_head", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--use_velocity", action="store_true", help="use velocity output in decoding/eval (if available)")
    ap.add_argument("--cache_dir", type=str, default=None, help="wave cache dir (optional)")
    ap.add_argument("--max_items", type=int, default=0, help="0 means evaluate all")
    ap.add_argument("--save_midis", action="store_true", help="save estimated MIDI files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_out_dir = out_dir / "est_midis"
    if args.save_midis:
        midi_out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    has_offset_head = any(k.startswith("offset_") for k in state.keys())
    has_velocity_head = any(k.startswith("velocity_") for k in state.keys())

    use_offset_head = _resolve_head_flag(args.model_use_offset_head, has_offset_head)
    use_velocity_head = _resolve_head_flag(args.model_use_velocity_head, has_velocity_head)

    feat_cfg = FeatureConfig()
    lab_cfg = LabelConfig()
    model_cfg = ModelConfig(
        n_mels=feat_cfg.n_mels,
        n_pitches=lab_cfg.midi_max - lab_cfg.midi_min + 1,
        use_offset_head=use_offset_head,
        use_velocity_head=use_velocity_head,
    )
    model = OnsetsAndFrames(model_cfg).to(args.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[model heads] offset={use_offset_head} velocity={use_velocity_head}")

    if bool(args.maestro_root) == bool(args.pairs_csv):
        raise SystemExit("Specify exactly one of --maestro_root or --pairs_csv")

    if args.pairs_csv is not None:
        items = read_split_labeled_csv(args.pairs_csv, split=args.split)
        source_name = f"pairs_csv:{args.pairs_csv}"
    else:
        items = read_maestro_pairs(args.maestro_root, split=args.split)
        source_name = f"maestro_root:{args.maestro_root}"

    if args.max_items > 0:
        items = items[: args.max_items]
    if not items:
        raise SystemExit(f"No pairs found for split={args.split} from {source_name}")

    featurizer = LogMelSpec(feat_cfg)
    dec_cfg = DecodeConfig(
        tau_on=args.tau_on,
        tau_fr=args.tau_fr,
        onset_gating=not args.no_onset_gating,
    )

    onset_ps: List[float] = []
    onset_rs: List[float] = []
    onset_fs: List[float] = []
    frame_ps: List[float] = []
    frame_rs: List[float] = []
    frame_fs: List[float] = []
    onset_pitch_fs: List[float] = []
    note_fs: List[float] = []
    note_vel_fs: List[float] = []

    dt = feat_cfg.hop_length / feat_cfg.sample_rate
    n_pitches = lab_cfg.midi_max - lab_cfg.midi_min + 1
    rows: List[Dict[str, float | str | int]] = []

    for idx, (audio_path, midi_path) in enumerate(tqdm(items, desc=f"infer+eval:{args.split}", unit="song"), start=1):
        audio_for_load = audio_path
        if args.cache_dir:
            audio_for_load = ensure_wave_cache(audio_path, cache_dir=args.cache_dir, sr=feat_cfg.sample_rate)

        wav, _sr = load_audio_mono(audio_for_load, target_sr=feat_cfg.sample_rate)
        with torch.no_grad():
            log_mel = featurizer(wav).contiguous().to(args.device, non_blocking=True)  # (1,F,T)
            try:
                out = model(log_mel)
            except RuntimeError as e:
                # Some environments hit intermittent cuDNN LSTM issues
                # (e.g., CUDNN_STATUS_NOT_SUPPORTED on certain tensor layouts).
                if "CUDNN_STATUS_NOT_SUPPORTED" not in str(e):
                    raise
                with torch.backends.cudnn.flags(enabled=False):
                    out = model(log_mel.contiguous())

        vel = out.get("velocity") if args.use_velocity and use_velocity_head else None
        est_notes = decode_notes(
            onset=out["onset"].cpu(),
            frame=out["frame"].cpu(),
            velocity=vel.cpu() if vel is not None else None,
            cfg=dec_cfg,
            hop_length=feat_cfg.hop_length,
            sample_rate=feat_cfg.sample_rate,
            midi_min=lab_cfg.midi_min,
        )
        ref_notes = load_midi_notes(
            midi_path,
            apply_sustain=lab_cfg.use_sustain,
            sustain_cc=lab_cfg.sustain_cc,
            sustain_threshold=lab_cfg.sustain_threshold,
        )

        prf_on = note_onset_prf(ref_notes, est_notes, onset_tolerance=args.onset_tolerance)
        onset_ps.append(prf_on.precision)
        onset_rs.append(prf_on.recall)
        onset_fs.append(prf_on.f1)

        n_frames = int(out["frame"].shape[-1])
        ref_roll = notes_to_frame_roll(ref_notes, n_pitches=n_pitches, n_frames=n_frames, dt=dt, midi_min=lab_cfg.midi_min)
        est_roll = notes_to_frame_roll(est_notes, n_pitches=n_pitches, n_frames=n_frames, dt=dt, midi_min=lab_cfg.midi_min)
        prf_fr = frame_prf(ref_roll, est_roll)
        frame_ps.append(prf_fr.precision)
        frame_rs.append(prf_fr.recall)
        frame_fs.append(prf_fr.f1)

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

        rows.append(
            {
                "idx": idx,
                "audio_path": audio_path,
                "midi_path": midi_path,
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

        if args.save_midis:
            stem = Path(audio_path).stem
            out_midi = midi_out_dir / f"{stem}.pred.mid"
            save_notes_as_midi(est_notes, out_midi, program=0)

    summary = {
        "ckpt": args.ckpt,
        "maestro_root": args.maestro_root,
        "pairs_csv": args.pairs_csv,
        "split": args.split,
        "n_items": len(items),
        "decode": {
            "tau_on": args.tau_on,
            "tau_fr": args.tau_fr,
            "onset_gating": not args.no_onset_gating,
            "onset_tolerance": args.onset_tolerance,
            "offset_ratio": args.offset_ratio,
            "offset_min_tolerance": args.offset_min_tolerance,
            "velocity_tolerance": args.velocity_tolerance,
        },
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
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "per_piece_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "idx", "audio_path", "midi_path", "n_ref_notes", "n_est_notes",
                "onset_p", "onset_r", "onset_f1",
                "onset_pitch_f", "note_f", "note_vel_f",
                "frame_p", "frame_r", "frame_f1",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    m = summary["metrics_mean"]
    print(
        f"[{args.split}] "
        f"onset_f1={m['onset_f1']:.4f} "
        f"(P={m['onset_p']:.4f}, R={m['onset_r']:.4f}) | "
        f"onset_pitch_f={m['onset_pitch_f']:.4f} | "
        f"note_f={m['note_f']:.4f} | "
        f"note_vel_f={m['note_vel_f']:.4f} | "
        f"frame_f1={m['frame_f1']:.4f} "
        f"(P={m['frame_p']:.4f}, R={m['frame_r']:.4f})"
    )
    print(f"Saved: {out_dir / 'summary.json'}")
    print(f"Saved: {out_dir / 'per_piece_metrics.csv'}")
    if args.save_midis:
        print(f"Saved MIDI dir: {midi_out_dir}")


if __name__ == "__main__":
    main()
