from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from oaf.augment import AugmentConfig, augment_spectrogram
from oaf.config import FeatureConfig, LabelConfig, ModelConfig, TrainConfig
from oaf.datasets import (
    LabeledPianoDataset,
    UnlabeledPianoDataset,
    read_labeled_csv,
    read_unlabeled_csv,
    read_maestro_pairs,
    read_split_labeled_csv,
)
from oaf.labels import notes_to_labels
from oaf.losses import SupervisedLossConfig, UnsupervisedLossConfig, supervised_loss, unsupervised_loss
from oaf.midi_utils import load_midi_notes
from oaf.model import OnsetsAndFrames
from oaf.pseudo_label import PseudoLabelConfig, make_pseudo_labels
from oaf.utils import load_checkpoint, save_checkpoint, set_seed, write_json, read_json, LossHistory


def _collate(batch):
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def _is_ddp() -> bool:
    return "LOCAL_RANK" in os.environ


def _checkpoint_has_head(state: Dict[str, torch.Tensor], head: str) -> bool:
    """Support both old names such as offset_* and official names offset_stack.*."""
    return any(k.startswith(f"{head}_") or k.startswith(f"{head}_stack") for k in state.keys())


def _binary_counts_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    if float(logits.min().item()) >= 0.0 and float(logits.max().item()) <= 1.0:
        probs = logits
    else:
        probs = torch.sigmoid(logits)
    pred = probs >= threshold
    gt = target > 0.5
    tp = (pred & gt).sum().item()
    fp = (pred & (~gt)).sum().item()
    fn = ((~pred) & gt).sum().item()
    return float(tp), float(fp), float(fn)


def _safe_ratio(num: float, den: float) -> float:
    den = float(den)
    if den <= 0.0:
        return 0.0
    return float(num) / den


def _save_pseudo_label_image(
    out_path: Path,
    step: int,
    clean: Dict[str, torch.Tensor],
    pseudo: Dict[str, torch.Tensor],
    mask: Dict[str, torch.Tensor],
    aug: Dict[str, torch.Tensor],
    sample_idx: int = 0,
    heads: Optional[Sequence[str]] = None,
) -> None:
    """Save a PNG visualization of pseudo labels for one unlabeled sample.

    Layout:
        row 1: clean probability
        row 2: pseudo label with ignore mask
        row 3: augmented probability
        row 4: augmented probability - clean probability

    Tensors are expected to have shape (B, P, T); this function slices [sample_idx].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if heads is None:
        heads = ["onset", "frame"]
    heads = [h for h in heads if h in clean and h in pseudo and h in mask and h in aug]
    if not heads:
        return

    n_rows = 4
    n_cols = len(heads)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 2.6 * n_rows))
    if n_cols == 1:
        axes = axes[:, None]

    for j, head in enumerate(heads):
        clean_p = clean[head][sample_idx].detach().float().cpu().numpy()
        pseudo_p = pseudo[head][sample_idx].detach().float().cpu().numpy()
        mask_p = mask[head][sample_idx].detach().float().cpu().numpy()
        aug_p = aug[head][sample_idx].detach().float().cpu().numpy()

        pseudo_disp = np.where(mask_p > 0.5, pseudo_p, np.nan)
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color="#444444")

        im0 = axes[0, j].imshow(
            clean_p, aspect="auto", origin="lower",
            vmin=0.0, vmax=1.0, cmap="magma", interpolation="nearest",
        )
        axes[0, j].set_title(f"{head}: clean prob (max={clean_p.max():.3f})")
        plt.colorbar(im0, ax=axes[0, j], fraction=0.03)

        im1 = axes[1, j].imshow(
            pseudo_disp, aspect="auto", origin="lower",
            vmin=0.0, vmax=1.0, cmap=cmap, interpolation="nearest",
        )
        keep_ratio = float(mask_p.mean())
        pos_ratio = float((pseudo_p * mask_p).sum() / max(mask_p.sum(), 1.0))
        axes[1, j].set_title(
            f"{head}: pseudo (mask kept={keep_ratio:.2%}, pos@conf={pos_ratio:.2%})"
        )
        plt.colorbar(im1, ax=axes[1, j], fraction=0.03)

        im2 = axes[2, j].imshow(
            aug_p, aspect="auto", origin="lower",
            vmin=0.0, vmax=1.0, cmap="magma", interpolation="nearest",
        )
        axes[2, j].set_title(f"{head}: aug prob (max={aug_p.max():.3f})")
        plt.colorbar(im2, ax=axes[2, j], fraction=0.03)

        diff = aug_p - clean_p
        im3 = axes[3, j].imshow(
            diff, aspect="auto", origin="lower",
            vmin=-0.5, vmax=0.5, cmap="RdBu_r", interpolation="nearest",
        )
        axes[3, j].set_title(f"{head}: aug - clean")
        plt.colorbar(im3, ax=axes[3, j], fraction=0.03)

        for r in range(n_rows):
            axes[r, j].set_xlabel("frame")
            axes[r, j].set_ylabel("pitch")

    fig.suptitle(f"step {step}  (sample {sample_idx})", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _pseudo_head_stats(pseudo: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    mask_f = mask.to(torch.float32)
    pseudo_f = pseudo.to(torch.float32)
    total = float(mask_f.numel())
    confident = float(mask_f.sum().item())
    positives = float((pseudo_f * mask_f).sum().item())
    negatives = float((mask_f - pseudo_f * mask_f).sum().item())
    return {
        "confident_ratio": _safe_ratio(confident, total),
        "positive_ratio_total": _safe_ratio(positives, total),
        "positive_ratio_confident": _safe_ratio(positives, confident),
        "negative_ratio_confident": _safe_ratio(negatives, confident),
    }


def _mask_degenerate_pseudo_heads(
    pseudo: Dict[str, torch.Tensor],
    mask: Dict[str, torch.Tensor],
    *,
    heads: Sequence[str],
    min_positive: int = 1,
    min_negative: int = 1,
) -> Dict[str, torch.Tensor]:
    """Disable unsupervised loss for heads whose pseudo labels have no usable positives/negatives.

    Without this guard, a batch with confident zeros but no confident positives can push a sparse
    head such as onset further toward all-zero collapse.
    """
    if min_positive <= 0 and min_negative <= 0:
        return mask

    out = dict(mask)
    for h in heads:
        if h not in pseudo or h not in mask:
            continue
        m = mask[h]
        p = pseudo[h]
        pos = int(((p > 0.5) & m).sum().item())
        neg = int(((p <= 0.5) & m).sum().item())
        if pos < min_positive or neg < min_negative:
            out[h] = torch.zeros_like(m, dtype=torch.bool)
    return out


def _estimate_reference_ratios_full(
    dataset: LabeledPianoDataset,
    *,
    max_items: int = 0,
    desc: str = "estimate ratios full",
) -> Dict[str, float]:
    """Estimate ones/zeros ratios from full pieces, not random training crops.

    The original dataset.estimate_reference_ratios() goes through __getitem__(), which returns a
    random segment. For distribution matching, a deterministic full-length estimate is more stable.
    """
    n = len(dataset) if max_items <= 0 else min(len(dataset), int(max_items))
    ones = {"onset": 0.0, "frame": 0.0, "offset": 0.0}
    zeros = {"onset": 0.0, "frame": 0.0, "offset": 0.0}

    for i in tqdm(range(n), desc=desc, unit="piece"):
        audio_path, midi_path = dataset.items[i]
        log_mel = dataset._load_mel(audio_path)
        notes = load_midi_notes(
            midi_path,
            apply_sustain=dataset.lab_cfg.use_sustain,
            sustain_cc=dataset.lab_cfg.sustain_cc,
            sustain_threshold=dataset.lab_cfg.sustain_threshold,
        )
        labs = notes_to_labels(
            notes,
            log_mel.shape[-1],
            dataset.feat_cfg,
            dataset.lab_cfg,
            compute_velocity=False,
        )
        for k, lab in {
            "onset": labs.onset,
            "frame": labs.frame,
            "offset": labs.offset,
        }.items():
            ones[k] += float((lab > 0.5).sum().item())
            zeros[k] += float((lab <= 0.5).sum().item())

    return {k: float(ones[k]) / max(float(zeros[k]), 1.0) for k in ones}


def _nonempty_or_die(name: str, items: Sequence[object]) -> None:
    if len(items) == 0:
        raise SystemExit(f"No items found for {name}.")


def main():
    ap = argparse.ArgumentParser(description="Semi-supervised O&F training with MAPS-labeled + MAESTRO-unlabeled support")

    # Data modes:
    #   1) --maps_csv + (--maestro_root or --unlabeled_csv): MAPS labeled/valid, MAESTRO/audio CSV unlabeled
    #   2) --maestro_root only: split MAESTRO train into labeled/unlabeled by label_frac
    #   3) --labeled_csv + --unlabeled_csv: fully manual CSV mode
    ap.add_argument("--maps_csv", type=str, default=None,
                    help="split-aware CSV with columns split,audio_path,midi_path for MAPS labeled/validation")
    ap.add_argument("--maps_train_split", type=str, default="train")
    ap.add_argument("--maps_valid_split", type=str, default="validation")
    ap.add_argument("--labeled_csv", type=str, default=None)
    ap.add_argument("--unlabeled_csv", type=str, default=None)
    ap.add_argument("--valid_csv", type=str, default=None)
    ap.add_argument("--maestro_root", type=str, default=None,
                    help="MAESTRO v3 root. With --maps_csv, used as unlabeled MAESTRO train audio.")
    ap.add_argument("--label_frac", type=float, default=0.1,
                    help="fraction of MAESTRO train to use as labeled when using MAESTRO-only mode")
    ap.add_argument("--label_seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--init_ckpt", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--batch_size_labeled", type=int, default=8)
    ap.add_argument("--batch_size_unlabeled", type=int, default=8)
    ap.add_argument("--segment_seconds", type=float, default=20.48,
                    help="official O&F-style segment is 327680 samples at 16 kHz = 20.48 s")
    ap.add_argument("--num_workers", type=int, default=None)

    # Stable defaults after switching to the official-size O&F model.
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", help="enable AMP; disabled by default for stability")
    ap.add_argument("--no_amp", action="store_true", help="kept for compatibility; AMP is already disabled by default")

    ap.add_argument("--tau_lo", type=float, default=0.05)
    ap.add_argument("--tau_up", type=float, default=0.95)

    ap.add_argument("--lambda_u", type=float, default=0.05)
    ap.add_argument("--lambda_u_warmup_steps", type=int, default=0,
                    help="linearly ramp lambda_u for the first N steps; 0 disables warmup")

    ap.add_argument("--no_distribution_matching", action="store_true")
    ap.add_argument("--ratios_json", type=str, default=None)
    ap.add_argument("--ratio_max_items", type=int, default=0,
                    help="0 means use all labeled items for full-length ratio estimation")
    ap.add_argument("--ratio_use_random_crops", action="store_true",
                    help="use the old random-crop ratio estimator instead of full-length labels")
    ap.add_argument("--min_pseudo_positives_per_head", type=int, default=1,
                    help="if a pseudo head has fewer positives than this in a batch, ignore that head for unsup loss")
    ap.add_argument("--min_pseudo_negatives_per_head", type=int, default=1)

    ap.add_argument("--ckpt_every", type=int, default=1000, help="save last.pt every N steps")
    ap.add_argument("--save_every", type=int, default=0, help="save numbered checkpoint every N steps (0=off)")
    ap.add_argument("--keep_last_n", type=int, default=0, help="keep only last N numbered checkpoints (0=keep all)")
    ap.add_argument("--valid_every", type=int, default=2000, help="run validation every N steps")
    ap.add_argument("--log_every", type=int, default=50, help="update progress bar every N steps")
    ap.add_argument("--pseudo_debug_every", type=int, default=200,
                    help="save pseudo-label debug stats every N steps (0=off)")
    ap.add_argument("--pseudo_debug_save_arrays", action="store_true",
                    help="save sample pseudo/mask tensors as .pt files (can be large)")
    ap.add_argument("--pseudo_debug_save_images", action="store_true",
                    help="save PNG pseudo-label visualizations")
    ap.add_argument("--pseudo_debug_include_offset", action="store_true",
                    help="include offset maps in pseudo debug PNGs if the model outputs offset")

    args = ap.parse_args()

    # ---- DDP setup ----
    ddp = _is_ddp()
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", device_id=device)
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device(args.device)
        rank = 0
        world = 1

    set_seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    if rank == 0:
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        if args.pseudo_debug_every > 0:
            (out_dir / "pseudo_debug").mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "hparams.json", vars(args))
    if ddp:
        dist.barrier()

    init_obj = torch.load(args.init_ckpt, map_location="cpu")
    init_state = init_obj["model"] if isinstance(init_obj, dict) and "model" in init_obj else init_obj
    if any(k.startswith("module.") for k in init_state.keys()):
        init_state = {k[len("module."):]: v for k, v in init_state.items()}

    has_offset_head = _checkpoint_has_head(init_state, "offset")
    has_velocity_head = _checkpoint_has_head(init_state, "velocity")

    feat_cfg = FeatureConfig()
    lab_cfg = LabelConfig()
    model_cfg = ModelConfig(
        n_mels=feat_cfg.n_mels,
        n_pitches=lab_cfg.midi_max - lab_cfg.midi_min + 1,
        use_offset_head=has_offset_head,
        use_velocity_head=has_velocity_head,
    )
    train_cfg = TrainConfig(
        batch_size_labeled=args.batch_size_labeled,
        batch_size_unlabeled=args.batch_size_unlabeled,
        segment_seconds=args.segment_seconds,
        device=str(device),
    )
    train_cfg.lr = float(args.lr)
    train_cfg.grad_clip_norm = float(args.grad_clip_norm)
    train_cfg.amp = bool(args.amp) and not bool(args.no_amp)
    if args.num_workers is not None:
        train_cfg.num_workers = int(args.num_workers)

    # --- data loading ---
    if args.maps_csv is not None:
        # Main paper-style setup: MAPS labeled/validation + MAESTRO train audio as unlabeled.
        labeled_items = read_split_labeled_csv(args.maps_csv, split=args.maps_train_split)
        valid_items = read_split_labeled_csv(args.maps_csv, split=args.maps_valid_split)

        if args.unlabeled_csv is not None:
            unlabeled_items = read_unlabeled_csv(args.unlabeled_csv)
            unlabeled_source = f"unlabeled_csv:{args.unlabeled_csv}"
        elif args.maestro_root is not None:
            unlabeled_pairs = read_maestro_pairs(args.maestro_root, split="train")
            unlabeled_items = [wav for wav, _mid in unlabeled_pairs]
            unlabeled_source = f"maestro_train_audio:{args.maestro_root}"
        else:
            raise SystemExit(
                "--maps_csv requires either --unlabeled_csv or --maestro_root to provide unlabeled audio."
            )

        source_name = f"maps_csv:{args.maps_csv}"
        if rank == 0:
            print(
                f"[MAPS SSL] labeled({args.maps_train_split})={len(labeled_items)} | "
                f"valid({args.maps_valid_split})={len(valid_items)} | "
                f"unlabeled={len(unlabeled_items)} ({unlabeled_source})"
            )

    elif args.maestro_root is not None:
        # Alternative MAESTRO-only SSL mode: split MAESTRO train into labeled/unlabeled.
        import random as _rng
        all_train = read_maestro_pairs(args.maestro_root, split="train")
        r = _rng.Random(args.label_seed)
        r.shuffle(all_train)
        n_lab = max(1, int(len(all_train) * args.label_frac))
        labeled_items = all_train[:n_lab]
        unlabeled_items = [wav for wav, _mid in all_train[n_lab:]]
        valid_items = read_maestro_pairs(args.maestro_root, split="validation")
        source_name = f"maestro_split:{args.maestro_root}"
        if rank == 0:
            print(
                f"[MAESTRO SSL] labeled={len(labeled_items)} | "
                f"unlabeled={len(unlabeled_items)} | val={len(valid_items)}"
            )

    elif args.labeled_csv is not None and args.unlabeled_csv is not None:
        labeled_items = read_labeled_csv(args.labeled_csv)
        unlabeled_items = read_unlabeled_csv(args.unlabeled_csv)
        valid_items = read_labeled_csv(args.valid_csv) if args.valid_csv else []
        source_name = f"csv:labeled={args.labeled_csv},unlabeled={args.unlabeled_csv}"
        if rank == 0:
            print(f"[CSV SSL] labeled={len(labeled_items)} | unlabeled={len(unlabeled_items)} | val={len(valid_items)}")

    else:
        raise SystemExit(
            "Specify one of: --maps_csv with --unlabeled_csv/--maestro_root, "
            "--maestro_root, or --labeled_csv and --unlabeled_csv."
        )

    _nonempty_or_die("labeled training set", labeled_items)
    _nonempty_or_die("unlabeled training set", unlabeled_items)

    if rank == 0 and ddp:
        print(
            f"[DDP] world={world}; batch sizes are per-rank. "
            f"global labeled={world * args.batch_size_labeled}, "
            f"global unlabeled={world * args.batch_size_unlabeled}"
        )

    labeled_ds = LabeledPianoDataset(
        labeled_items, feat_cfg, lab_cfg, train_cfg,
        cache_dir=args.cache_dir,
        compute_velocity=False,
    )
    unlabeled_ds = UnlabeledPianoDataset(
        unlabeled_items, feat_cfg, train_cfg,
        cache_dir=args.cache_dir,
    )

    # drop_last=False in DistributedSampler avoids empty samplers for very small labeled sets.
    # DataLoader still uses drop_last=True to keep fixed batch shapes.
    lab_sampler = DistributedSampler(labeled_ds, shuffle=True, drop_last=False) if ddp else None
    unlab_sampler = DistributedSampler(unlabeled_ds, shuffle=True, drop_last=False) if ddp else None

    labeled_dl = DataLoader(
        labeled_ds,
        batch_size=train_cfg.batch_size_labeled,
        shuffle=(lab_sampler is None),
        sampler=lab_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(train_cfg.num_workers > 0),
        collate_fn=_collate,
    )
    unlabeled_dl = DataLoader(
        unlabeled_ds,
        batch_size=train_cfg.batch_size_unlabeled,
        shuffle=(unlab_sampler is None),
        sampler=unlab_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(train_cfg.num_workers > 0),
        collate_fn=_collate,
    )
    if len(labeled_dl) == 0:
        raise SystemExit(
            f"Labeled DataLoader is empty. Reduce --batch_size_labeled "
            f"(current per-rank batch={train_cfg.batch_size_labeled}) or use fewer DDP ranks."
        )
    if len(unlabeled_dl) == 0:
        raise SystemExit(
            f"Unlabeled DataLoader is empty. Reduce --batch_size_unlabeled "
            f"(current per-rank batch={train_cfg.batch_size_unlabeled})."
        )

    valid_dl = None
    if valid_items:
        valid_ds = LabeledPianoDataset(
            valid_items, feat_cfg, lab_cfg, train_cfg,
            cache_dir=args.cache_dir,
            compute_velocity=False,
        )
        valid_sampler = DistributedSampler(valid_ds, shuffle=False, drop_last=False) if ddp else None
        valid_dl = DataLoader(
            valid_ds,
            batch_size=train_cfg.batch_size_labeled,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(train_cfg.num_workers > 0),
            collate_fn=_collate,
        )

    # --- reference ratios for distribution matching ---
    if args.no_distribution_matching:
        desired_ratios = None
    elif args.ratios_json is not None:
        desired_ratios = read_json(args.ratios_json)
        if rank == 0:
            write_json(out_dir / "ref_ratios.json", desired_ratios)
    else:
        ratios_path = out_dir / "ref_ratios.json"
        if rank == 0:
            if args.ratio_use_random_crops:
                desired_ratios = labeled_ds.estimate_reference_ratios(
                    max_items=(None if args.ratio_max_items <= 0 else args.ratio_max_items)
                )
            else:
                desired_ratios = _estimate_reference_ratios_full(
                    labeled_ds,
                    max_items=args.ratio_max_items,
                    desc="estimate ref ratios",
                )
            write_json(ratios_path, desired_ratios)
        if ddp:
            dist.barrier()
        desired_ratios = read_json(ratios_path)

    if rank == 0:
        print(f"[data source] {source_name}")
        if desired_ratios is not None:
            print(f"[distribution matching ratios] {desired_ratios}")

    model = OnsetsAndFrames(model_cfg).to(device)
    if rank == 0:
        print(f"[model heads] offset={has_offset_head} velocity={has_velocity_head}")
        print(
            f"[train cfg] lr={train_cfg.lr:g} clip={train_cfg.grad_clip_norm:g} "
            f"amp={train_cfg.amp} segment_seconds={train_cfg.segment_seconds} workers={train_cfg.num_workers}"
        )

    # Load initial supervised weights.
    load_checkpoint(args.init_ckpt, model, optimizer=None, scheduler=None, map_location="cpu")

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, map_location="cpu")
        if rank == 0:
            print(f"[resume] path={args.resume} | start_step={start_step} | target_iters={args.iters}")

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if ddp else model

    opt = torch.optim.Adam(raw_model.parameters(), lr=train_cfg.lr)
    sch = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=train_cfg.lr_decay_every_steps,
        gamma=train_cfg.lr_decay_gamma,
    )

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sch.load_state_dict(ckpt["scheduler"])

    sup_cfg = SupervisedLossConfig(
        use_weighted_frame_loss=False,
        lambda_on=1.0,
        lambda_fr=1.0,
        lambda_off=1.0,
        lambda_vel=0.0,
    )
    unsup_cfg = UnsupervisedLossConfig(
        lambda_u=args.lambda_u,
        lambda_on=1.0,
        lambda_fr=1.0,
        lambda_off=0.0,  # paper default: do not use unsupervised offset loss
    )

    pseudo_cfg = PseudoLabelConfig(
        tau_lo=args.tau_lo,
        tau_up=args.tau_up,
        use_distribution_matching=not args.no_distribution_matching,
        seed=None,  # do not reset the undersampling generator at every call
    )
    aug_cfg = AugmentConfig()

    use_amp = train_cfg.amp and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = LossHistory()
    train_csv_path = out_dir / "logs" / "train_metrics.csv"
    val_csv_path = out_dir / "logs" / "val_metrics.csv"
    pseudo_csv_path = out_dir / "logs" / "pseudo_debug.csv"

    if rank == 0:
        init_log_files = not (
            args.resume is not None
            and train_csv_path.exists()
            and val_csv_path.exists()
            and pseudo_csv_path.exists()
        )
        if init_log_files:
            with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "lr",
                        "lambda_u_now",
                        "train_loss",
                        "sup_loss",
                        "unsup_loss",
                        "pseudo_onset_confident_ratio",
                        "pseudo_onset_positive_ratio_confident",
                        "pseudo_frame_confident_ratio",
                        "pseudo_frame_positive_ratio_confident",
                    ],
                )
                w.writeheader()
            with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "val_loss",
                        "onset_p",
                        "onset_r",
                        "onset_f1",
                        "frame_p",
                        "frame_r",
                        "frame_f1",
                    ],
                )
                w.writeheader()
            with open(pseudo_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "head",
                        "confident_ratio",
                        "positive_ratio_total",
                        "positive_ratio_confident",
                        "negative_ratio_confident",
                    ],
                )
                w.writeheader()
        else:
            print("[resume] append logs to existing CSV files")

    model.train()
    it_l = iter(labeled_dl)
    it_u = iter(unlabeled_dl)
    epoch_l = 0
    epoch_u = 0
    pbar = tqdm(range(start_step, args.iters), initial=start_step, total=args.iters, disable=(rank != 0))

    best_val = float("inf")
    unsup_heads = ["onset", "frame"]
    debug_heads = ["onset", "frame", "offset"] if args.pseudo_debug_include_offset else ["onset", "frame"]

    for step in pbar:
        try:
            lb = next(it_l)
        except StopIteration:
            epoch_l += 1
            if lab_sampler is not None:
                lab_sampler.set_epoch(epoch_l)
            it_l = iter(labeled_dl)
            lb = next(it_l)

        try:
            ub = next(it_u)
        except StopIteration:
            epoch_u += 1
            if unlab_sampler is not None:
                unlab_sampler.set_epoch(epoch_u)
            it_u = iter(unlabeled_dl)
            ub = next(it_u)

        x_l = lb["log_mel"].to(device, non_blocking=True)
        y_l = {
            "onset": lb["onset"].to(device, non_blocking=True),
            "frame": lb["frame"].to(device, non_blocking=True),
            "offset": lb["offset"].to(device, non_blocking=True),
        }

        x_u = ub["log_mel"].to(device, non_blocking=True)
        x_u_aug = augment_spectrogram(x_u, aug_cfg)

        # Generate pseudo labels from the clean unlabeled input. No gradients flow through this branch.
        with torch.no_grad():
            model.eval()
            out_u_clean = model(x_u)
            model.train()
            pseudo, mask = make_pseudo_labels(out_u_clean, pseudo_cfg, desired_ratios=desired_ratios)
            mask = _mask_degenerate_pseudo_heads(
                pseudo,
                mask,
                heads=unsup_heads,
                min_positive=args.min_pseudo_positives_per_head,
                min_negative=args.min_pseudo_negatives_per_head,
            )
            pseudo_stats = {k: _pseudo_head_stats(pseudo[k], mask[k]) for k in pseudo.keys()}

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            # Single train-mode forward pass avoids BatchNorm version conflicts from multiple forwards.
            x_cat = torch.cat([x_l, x_u_aug], dim=0)
            out_cat = model(x_cat)
            b_l = x_l.size(0)
            out_l = {k: v[:b_l] for k, v in out_cat.items()}
            out_u_aug = {k: v[b_l:] for k, v in out_cat.items()}

            ls = supervised_loss(out_l, y_l, sup_cfg)
            lu = unsupervised_loss(out_u_aug, pseudo, mask, cfg=unsup_cfg)

            if args.lambda_u_warmup_steps > 0:
                ramp = min(1.0, float(step + 1) / float(args.lambda_u_warmup_steps))
            else:
                ramp = 1.0
            lambda_u_now = float(unsup_cfg.lambda_u) * ramp
            loss = (1.0 - lambda_u_now) * ls + lambda_u_now * lu

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        scaler.step(opt)
        scaler.update()
        sch.step()

        if (step + 1) % args.log_every == 0:
            l_val, ls_val, lu_val = float(loss.item()), float(ls.item()), float(lu.item())
            cur_lr = float(opt.param_groups[0]["lr"])
            pbar.set_description(
                f"loss={l_val:.4f} ls={ls_val:.4f} lu={lu_val:.4f} "
                f"lambda_u={lambda_u_now:.4g}"
            )
            if rank == 0:
                history.append(step + 1, train_loss=l_val, sup_loss=ls_val, unsup_loss=lu_val)
                row = {
                    "step": step + 1,
                    "lr": cur_lr,
                    "lambda_u_now": lambda_u_now,
                    "train_loss": l_val,
                    "sup_loss": ls_val,
                    "unsup_loss": lu_val,
                    "pseudo_onset_confident_ratio": pseudo_stats.get("onset", {}).get("confident_ratio", 0.0),
                    "pseudo_onset_positive_ratio_confident": pseudo_stats.get("onset", {}).get("positive_ratio_confident", 0.0),
                    "pseudo_frame_confident_ratio": pseudo_stats.get("frame", {}).get("confident_ratio", 0.0),
                    "pseudo_frame_positive_ratio_confident": pseudo_stats.get("frame", {}).get("positive_ratio_confident", 0.0),
                }
                with open(train_csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    w.writerow(row)

        if args.pseudo_debug_every > 0 and (step + 1) % args.pseudo_debug_every == 0:
            if rank == 0:
                with open(pseudo_csv_path, "a", newline="", encoding="utf-8") as f:
                    fieldnames = [
                        "step",
                        "head",
                        "confident_ratio",
                        "positive_ratio_total",
                        "positive_ratio_confident",
                        "negative_ratio_confident",
                    ]
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    for head, st in pseudo_stats.items():
                        w.writerow({"step": step + 1, "head": head, **st})

                if args.pseudo_debug_save_arrays:
                    sample = {"step": step + 1}
                    for h in debug_heads:
                        if h in out_u_clean:
                            sample[f"clean_{h}_0"] = out_u_clean[h][0].detach().cpu()
                        if h in pseudo:
                            sample[f"pseudo_{h}_0"] = pseudo[h][0].detach().cpu()
                        if h in mask:
                            sample[f"mask_{h}_0"] = mask[h][0].detach().cpu()
                        if h in out_u_aug:
                            sample[f"aug_{h}_0"] = out_u_aug[h][0].detach().cpu()
                    torch.save(sample, out_dir / "pseudo_debug" / f"step_{step + 1:07d}.pt")

                if args.pseudo_debug_save_images:
                    try:
                        _save_pseudo_label_image(
                            out_dir / "pseudo_debug" / f"step_{step + 1:07d}.png",
                            step=step + 1,
                            clean=out_u_clean,
                            pseudo=pseudo,
                            mask=mask,
                            aug=out_u_aug,
                            sample_idx=0,
                            heads=debug_heads,
                        )
                    except Exception as e:
                        print(f"[pseudo_debug] failed to save image at step {step + 1}: {e}")

        if rank == 0 and (step + 1) % args.ckpt_every == 0:
            save_checkpoint(out_dir / "checkpoints" / "last.pt", raw_model, opt, sch, step=step + 1)

        if rank == 0 and args.save_every > 0 and (step + 1) % args.save_every == 0:
            numbered = out_dir / "checkpoints" / f"step_{step + 1}.pt"
            save_checkpoint(numbered, raw_model, opt, sch, step=step + 1)
            if args.keep_last_n > 0:
                existing = sorted(out_dir.glob("checkpoints/step_*.pt"))
                for old in existing[: -args.keep_last_n]:
                    old.unlink(missing_ok=True)

        if valid_dl is not None and (step + 1) % args.valid_every == 0:
            model.eval()
            with torch.no_grad():
                loss_sum = 0.0
                n_batches = 0.0
                onset_tp = onset_fp = onset_fn = 0.0
                frame_tp = frame_fp = frame_fn = 0.0
                for vb in valid_dl:
                    vmel = vb["log_mel"].to(device, non_blocking=True)
                    vlabels = {
                        "onset": vb["onset"].to(device, non_blocking=True),
                        "frame": vb["frame"].to(device, non_blocking=True),
                        "offset": vb["offset"].to(device, non_blocking=True),
                    }
                    vout = model(vmel)
                    vloss = supervised_loss(vout, vlabels, sup_cfg)
                    loss_sum += float(vloss.item())
                    n_batches += 1.0
                    tp, fp, fn = _binary_counts_from_logits(vout["onset"], vlabels["onset"])
                    onset_tp += tp
                    onset_fp += fp
                    onset_fn += fn
                    tp, fp, fn = _binary_counts_from_logits(vout["frame"], vlabels["frame"])
                    frame_tp += tp
                    frame_fp += fp
                    frame_fn += fn

                val = loss_sum / max(n_batches, 1.0)
                onset_p = onset_tp / max(onset_tp + onset_fp, 1.0)
                onset_r = onset_tp / max(onset_tp + onset_fn, 1.0)
                onset_f1 = 2.0 * onset_p * onset_r / max(onset_p + onset_r, 1e-12)
                frame_p = frame_tp / max(frame_tp + frame_fp, 1.0)
                frame_r = frame_tp / max(frame_tp + frame_fn, 1.0)
                frame_f1 = 2.0 * frame_p * frame_r / max(frame_p + frame_r, 1e-12)

            if ddp:
                val_t = torch.tensor(
                    [loss_sum, n_batches, onset_tp, onset_fp, onset_fn, frame_tp, frame_fp, frame_fn],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(val_t, op=dist.ReduceOp.SUM)
                val = float(val_t[0] / val_t[1]) if val_t[1] > 0 else float("inf")
                onset_tp, onset_fp, onset_fn = float(val_t[2]), float(val_t[3]), float(val_t[4])
                frame_tp, frame_fp, frame_fn = float(val_t[5]), float(val_t[6]), float(val_t[7])
                onset_p = onset_tp / max(onset_tp + onset_fp, 1.0)
                onset_r = onset_tp / max(onset_tp + onset_fn, 1.0)
                onset_f1 = 2.0 * onset_p * onset_r / max(onset_p + onset_r, 1e-12)
                frame_p = frame_tp / max(frame_tp + frame_fp, 1.0)
                frame_r = frame_tp / max(frame_tp + frame_fn, 1.0)
                frame_f1 = 2.0 * frame_p * frame_r / max(frame_p + frame_r, 1e-12)

            if rank == 0:
                history.append(
                    step + 1,
                    val_loss=float(val),
                    val_onset_f1=float(onset_f1),
                    val_frame_f1=float(frame_f1),
                )
                print(
                    f"[val step {step + 1}] loss={val:.4f} "
                    f"onset_f1={onset_f1:.4f} frame_f1={frame_f1:.4f}"
                )
                with open(val_csv_path, "a", newline="", encoding="utf-8") as f:
                    row = {
                        "step": step + 1,
                        "val_loss": val,
                        "onset_p": onset_p,
                        "onset_r": onset_r,
                        "onset_f1": onset_f1,
                        "frame_p": frame_p,
                        "frame_r": frame_r,
                        "frame_f1": frame_f1,
                    }
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    w.writerow(row)
                if val < best_val:
                    best_val = val
                    save_checkpoint(
                        out_dir / "checkpoints" / "best.pt",
                        raw_model,
                        opt,
                        sch,
                        step=step + 1,
                        extra={
                            "val_loss": float(val),
                            "val_onset_f1": float(onset_f1),
                            "val_frame_f1": float(frame_f1),
                        },
                    )
            model.train()

    if rank == 0:
        save_checkpoint(out_dir / "checkpoints" / "last.pt", raw_model, opt, sch, step=args.iters)
        history.save(out_dir / "loss_history.json")
        history.plot(out_dir / "loss_curve.png", title="SSL Training Loss")
        print("Done.")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
