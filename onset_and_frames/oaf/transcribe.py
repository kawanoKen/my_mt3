from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch

from oaf.config import DecodeConfig, FeatureConfig, LabelConfig, ModelConfig
from oaf.decoding import decode_notes, save_notes_as_midi
from oaf.features import LogMelSpec, load_audio_mono
from oaf.model import OnsetsAndFrames
from oaf.utils import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--out_midi", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tau_on", type=float, default=0.5)
    ap.add_argument("--tau_fr", type=float, default=0.5)
    ap.add_argument("--no_onset_gating", action="store_true")
    ap.add_argument("--use_velocity", action="store_true")
    args = ap.parse_args()

    feat_cfg = FeatureConfig()
    lab_cfg = LabelConfig()
    model_cfg = ModelConfig(
        n_mels=feat_cfg.n_mels,
        n_pitches=lab_cfg.midi_max - lab_cfg.midi_min + 1,
        use_offset_head=True,
        use_velocity_head=args.use_velocity,
    )
    model = OnsetsAndFrames(model_cfg).to(args.device)
    load_checkpoint(args.ckpt, model, map_location="cpu")
    model.eval()

    wav, sr = load_audio_mono(args.audio, target_sr=feat_cfg.sample_rate)
    featurizer = LogMelSpec(feat_cfg)
    with torch.no_grad():
        log_mel = featurizer(wav).to(args.device)  # (1,F,T)
        out = model(log_mel)

    dec_cfg = DecodeConfig(
        tau_on=args.tau_on,
        tau_fr=args.tau_fr,
        onset_gating=not args.no_onset_gating,
    )
    vel = out.get("velocity") if args.use_velocity else None
    notes = decode_notes(
        onset=out["onset"].cpu(),
        frame=out["frame"].cpu(),
        velocity=vel.cpu() if vel is not None else None,
        cfg=dec_cfg,
        hop_length=feat_cfg.hop_length,
        sample_rate=feat_cfg.sample_rate,
        midi_min=lab_cfg.midi_min,
    )

    save_notes_as_midi(notes, args.out_midi, program=0)
    print(f"Wrote {args.out_midi} with {len(notes)} notes.")


if __name__ == "__main__":
    main()
