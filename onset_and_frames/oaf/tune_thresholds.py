from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from oaf.config import DecodeConfig, FeatureConfig, LabelConfig, ModelConfig
from oaf.decoding import decode_notes
from oaf.eval_metrics import frame_prf, note_onset_prf, notes_to_frame_roll
from oaf.features import LogMelSpec, load_audio_mono
from oaf.midi_utils import load_midi_notes
from oaf.model import OnsetsAndFrames
from oaf.utils import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--valid_csv", type=str, required=True, help="CSV with audio_path,midi_path")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tau_grid", type=str, default="0.05:0.95:0.05", help="start:end:step")
    args = ap.parse_args()

    start, end, step = [float(x) for x in args.tau_grid.split(":")]
    taus = np.arange(start, end + 1e-9, step).tolist()

    feat_cfg = FeatureConfig()
    lab_cfg = LabelConfig()
    model_cfg = ModelConfig(
        n_mels=feat_cfg.n_mels,
        n_pitches=lab_cfg.midi_max - lab_cfg.midi_min + 1,
        use_offset_head=True,
        use_velocity_head=False,
    )
    model = OnsetsAndFrames(model_cfg).to(args.device)
    load_checkpoint(args.ckpt, model, map_location="cpu")
    model.eval()

    # load valid list
    import csv
    items: List[Tuple[str, str]] = []
    with open(args.valid_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((row["audio_path"], row["midi_path"]))
    featurizer = LogMelSpec(feat_cfg)

    # precompute predictions for each piece (to avoid recomputing in grid-search)
    cached = []
    for audio_path, midi_path in tqdm(items, desc="precompute"):
        wav, sr = load_audio_mono(audio_path, target_sr=feat_cfg.sample_rate)
        with torch.no_grad():
            log_mel = featurizer(wav).to(args.device)
            out = model(log_mel)
        onset = out["onset"].detach().cpu()[0]  # (P,T)
        frame = out["frame"].detach().cpu()[0]
        # reference notes (with sustain)
        ref_notes = load_midi_notes(midi_path, apply_sustain=lab_cfg.use_sustain,
                                    sustain_cc=lab_cfg.sustain_cc, sustain_threshold=lab_cfg.sustain_threshold)
        cached.append((onset, frame, ref_notes))

    dt = feat_cfg.hop_length / feat_cfg.sample_rate
    P = lab_cfg.midi_max - lab_cfg.midi_min + 1

    # 1) tune tau_on by maximizing note-onset F1 (paper procedure)
    best_tau_on = 0.5
    best_note_f1 = -1.0
    fixed_tau_fr = 0.5

    for tau_on in tqdm(taus, desc="search tau_on"):
        f1s = []
        for onset, frame, ref_notes in cached:
            dec_cfg = DecodeConfig(tau_on=tau_on, tau_fr=fixed_tau_fr, onset_gating=True)
            est_notes = decode_notes(onset, frame, cfg=dec_cfg, hop_length=feat_cfg.hop_length,
                                     sample_rate=feat_cfg.sample_rate, midi_min=lab_cfg.midi_min)
            prf = note_onset_prf(ref_notes, est_notes, onset_tolerance=0.05)
            f1s.append(prf.f1)
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        if mean_f1 > best_note_f1:
            best_note_f1 = mean_f1
            best_tau_on = tau_on

    # 2) tune tau_fr by maximizing frame F1 (using decoded notes -> frame roll)
    best_tau_fr = 0.5
    best_frame_f1 = -1.0

    for tau_fr in tqdm(taus, desc="search tau_fr"):
        f1s = []
        for onset, frame, ref_notes in cached:
            dec_cfg = DecodeConfig(tau_on=best_tau_on, tau_fr=tau_fr, onset_gating=True)
            est_notes = decode_notes(onset, frame, cfg=dec_cfg, hop_length=feat_cfg.hop_length,
                                     sample_rate=feat_cfg.sample_rate, midi_min=lab_cfg.midi_min)
            # reference roll from ref notes; predicted roll from est notes
            n_frames = frame.shape[-1]
            ref_roll = notes_to_frame_roll(ref_notes, n_pitches=P, n_frames=n_frames, dt=dt, midi_min=lab_cfg.midi_min)
            est_roll = notes_to_frame_roll(est_notes, n_pitches=P, n_frames=n_frames, dt=dt, midi_min=lab_cfg.midi_min)
            prf = frame_prf(ref_roll, est_roll)
            f1s.append(prf.f1)
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        if mean_f1 > best_frame_f1:
            best_frame_f1 = mean_f1
            best_tau_fr = tau_fr

    print(f"best tau_on={best_tau_on:.3f} (note F1={best_note_f1:.4f})")
    print(f"best tau_fr={best_tau_fr:.3f} (frame F1={best_frame_f1:.4f})")


if __name__ == "__main__":
    main()
