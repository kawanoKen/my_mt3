"""Offline visualizer for pseudo labels.

Loads a checkpoint (typically from a finished SSL run), feeds a few unlabeled
audio segments through the model, computes the pseudo labels with the same
threshold/distribution-matching settings as during training, and saves PNG
heatmaps under ``<out_dir>/pseudo_debug/``.

Use this for runs that were trained without ``--pseudo_debug_save_images``.

Example:
    python -m oaf.visualize_pseudo \
        --ckpt runs/maps_maestro_ssl_5000_tau_up07/checkpoints/last.pt \
        --unlabeled_csv runs/maps_maestro_ssl_splits/MAPS_Full/unlabeled_train.csv \
        --ratios_json runs/maps_maestro_ssl_5000_tau_up07/ref_ratios.json \
        --tau_lo 0.05 --tau_up 0.7 \
        --out_dir runs/maps_maestro_ssl_5000_tau_up07/pseudo_debug_offline \
        --n_samples 4
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch

from oaf.config import FeatureConfig, LabelConfig, ModelConfig, TrainConfig
from oaf.datasets import UnlabeledPianoDataset, read_unlabeled_csv
from oaf.model import OnsetsAndFrames
from oaf.pseudo_label import PseudoLabelConfig, make_pseudo_labels
from oaf.utils import read_json
from oaf.augment import AugmentConfig, augment_spectrogram
from oaf.train_ssl import _save_pseudo_label_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="path to a checkpoint (.pt) produced by train_ssl/train_supervised")
    ap.add_argument("--unlabeled_csv", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--ratios_json", type=str, default=None,
                    help="optional ref_ratios.json for distribution matching")
    ap.add_argument("--no_distribution_matching", action="store_true")
    ap.add_argument("--tau_lo", type=float, default=0.05)
    ap.add_argument("--tau_up", type=float, default=0.7)
    ap.add_argument("--segment_seconds", type=float, default=20.0)
    ap.add_argument("--n_samples", type=int, default=4,
                    help="number of unlabeled segments to visualize")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="directory to write PNGs into (will be created)")
    ap.add_argument("--save_aug", action="store_true",
                    help="also run an augmented forward pass and show aug-vs-clean diff")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_cfg = FeatureConfig()
    lab_cfg = LabelConfig()
    train_cfg = TrainConfig(
        batch_size_labeled=1,
        batch_size_unlabeled=1,
        segment_seconds=args.segment_seconds,
        device=str(device),
    )

    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    # has_offset = any(k.startswith("offset_") for k in state.keys())
    # has_velocity = any(k.startswith("velocity_") for k in state.keys())
    has_offset = any(k.startswith("offset_stack") or k.startswith("offset_") for k in state.keys())
    has_velocity = any(k.startswith("velocity_stack") or k.startswith("velocity_") for k in state.keys())

    model_cfg = ModelConfig(
        n_mels=feat_cfg.n_mels,
        n_pitches=lab_cfg.midi_max - lab_cfg.midi_min + 1,
        use_offset_head=has_offset,
        use_velocity_head=has_velocity,
    )
    model = OnsetsAndFrames(model_cfg)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"[ckpt] loaded {args.ckpt} (offset={has_offset}, velocity={has_velocity})")

    if args.no_distribution_matching:
        desired_ratios = None
    elif args.ratios_json is not None:
        desired_ratios = read_json(args.ratios_json)
        print(f"[ratios] loaded {desired_ratios}")
    else:
        desired_ratios = None

    pseudo_cfg = PseudoLabelConfig(
        tau_lo=args.tau_lo,
        tau_up=args.tau_up,
        use_distribution_matching=not args.no_distribution_matching,
        seed=args.seed,
    )
    aug_cfg = AugmentConfig()

    items = read_unlabeled_csv(args.unlabeled_csv)
    ds = UnlabeledPianoDataset(items, feat_cfg, train_cfg, cache_dir=args.cache_dir)
    print(f"[data] {len(items)} unlabeled items, sampling {args.n_samples}")

    g = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(ds), generator=g)[: args.n_samples].tolist()

    for k, idx in enumerate(indices):
        ex = ds[idx]
        x = ex["log_mel"].unsqueeze(0).to(device)

        with torch.no_grad():
            out_clean = model(x)
            pseudo, mask = make_pseudo_labels(out_clean, pseudo_cfg, desired_ratios=desired_ratios)
            if args.save_aug:
                x_aug = augment_spectrogram(x, aug_cfg)
                out_aug = model(x_aug)
            else:
                out_aug = {kk: vv.clone() for kk, vv in out_clean.items()}

        png_path = out_dir / f"sample_{k:02d}_dsidx_{idx:05d}.png"
        _save_pseudo_label_image(
            png_path,
            step=0,
            clean=out_clean,
            pseudo=pseudo,
            mask=mask,
            aug=out_aug,
            sample_idx=0,
        )
        print(f"[saved] {png_path}")

    print("Done.")


if __name__ == "__main__":
    main()
