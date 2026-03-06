from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from oaf.augment import AugmentConfig, augment_spectrogram
from oaf.config import FeatureConfig, LabelConfig, ModelConfig, TrainConfig
from oaf.datasets import LabeledPianoDataset, UnlabeledPianoDataset, read_labeled_csv, read_unlabeled_csv, read_maestro_pairs
from oaf.losses import SupervisedLossConfig, UnsupervisedLossConfig, supervised_loss, unsupervised_loss
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


def _binary_counts_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    gt = target > 0.5
    tp = (pred & gt).sum().item()
    fp = (pred & (~gt)).sum().item()
    fn = ((~pred) & gt).sum().item()
    return float(tp), float(fp), float(fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_csv", type=str, default=None)
    ap.add_argument("--unlabeled_csv", type=str, default=None)
    ap.add_argument("--valid_csv", type=str, default=None)
    ap.add_argument("--maestro_root", type=str, default=None,
                     help="MAESTRO v3 root directory (alternative to CSV args)")
    ap.add_argument("--label_frac", type=float, default=0.1,
                     help="fraction of MAESTRO train to use as labeled (with --maestro_root)")
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
    ap.add_argument("--segment_seconds", type=float, default=20.0)

    ap.add_argument("--tau_lo", type=float, default=0.05)
    ap.add_argument("--tau_up", type=float, default=0.95)

    ap.add_argument("--lambda_u", type=float, default=0.05)

    ap.add_argument("--no_distribution_matching", action="store_true")
    ap.add_argument("--ratios_json", type=str, default=None)
    ap.add_argument("--ratio_max_items", type=int, default=200)

    ap.add_argument("--ckpt_every", type=int, default=1000,
                     help="save last.pt every N steps")
    ap.add_argument("--save_every", type=int, default=0,
                     help="save numbered checkpoint (step_N.pt) every N steps (0=off)")
    ap.add_argument("--keep_last_n", type=int, default=0,
                     help="keep only last N numbered checkpoints (0=keep all)")
    ap.add_argument("--valid_every", type=int, default=2000,
                     help="run validation every N steps")
    ap.add_argument("--log_every", type=int, default=50,
                     help="update progress bar every N steps")

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
        write_json(out_dir / "hparams.json", vars(args))
    if ddp:
        dist.barrier()

    init_obj = torch.load(args.init_ckpt, map_location="cpu")
    init_state = init_obj["model"] if isinstance(init_obj, dict) and "model" in init_obj else init_obj
    has_offset_head = any(k.startswith("offset_") for k in init_state.keys())
    has_velocity_head = any(k.startswith("velocity_") for k in init_state.keys())

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

    # --- data loading ---
    if args.maestro_root is not None:
        import random as _rng
        all_train = read_maestro_pairs(args.maestro_root, split="train")
        r = _rng.Random(args.label_seed)
        r.shuffle(all_train)
        n_lab = max(1, int(len(all_train) * args.label_frac))
        labeled_items = all_train[:n_lab]
        unlabeled_items = [wav for wav, _mid in all_train[n_lab:]]
        valid_items = read_maestro_pairs(args.maestro_root, split="validation")
        if rank == 0:
            print(f"[MAESTRO SSL] labeled={len(labeled_items)} | unlabeled={len(unlabeled_items)} | val={len(valid_items)}")
    elif args.labeled_csv is not None and args.unlabeled_csv is not None:
        labeled_items = read_labeled_csv(args.labeled_csv)
        unlabeled_items = read_unlabeled_csv(args.unlabeled_csv)
        valid_items = read_labeled_csv(args.valid_csv) if args.valid_csv else []
    else:
        raise SystemExit("Either --maestro_root or both --labeled_csv and --unlabeled_csv must be specified.")

    labeled_ds = LabeledPianoDataset(labeled_items, feat_cfg, lab_cfg, train_cfg, cache_dir=args.cache_dir, compute_velocity=False)
    unlabeled_ds = UnlabeledPianoDataset(unlabeled_items, feat_cfg, train_cfg, cache_dir=args.cache_dir)

    lab_sampler = DistributedSampler(labeled_ds, shuffle=True, drop_last=True) if ddp else None
    unlab_sampler = DistributedSampler(unlabeled_ds, shuffle=True, drop_last=True) if ddp else None

    labeled_dl = DataLoader(
        labeled_ds,
        batch_size=train_cfg.batch_size_labeled,
        shuffle=(lab_sampler is None),
        sampler=lab_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
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
        collate_fn=_collate,
    )

    valid_dl = None
    if valid_items:
        valid_ds = LabeledPianoDataset(valid_items, feat_cfg, lab_cfg, train_cfg, cache_dir=args.cache_dir, compute_velocity=False)
        valid_sampler = DistributedSampler(valid_ds, shuffle=False, drop_last=False) if ddp else None
        valid_dl = DataLoader(
            valid_ds,
            batch_size=train_cfg.batch_size_labeled,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=_collate,
        )

    # reference ratios for distribution matching
    if args.no_distribution_matching:
        desired_ratios = None
    else:
        if args.ratios_json is not None:
            desired_ratios = read_json(args.ratios_json)
        else:
            desired_ratios = labeled_ds.estimate_reference_ratios(max_items=args.ratio_max_items)
            if rank == 0:
                write_json(out_dir / "ref_ratios.json", desired_ratios)

    model = OnsetsAndFrames(model_cfg).to(device)
    if rank == 0:
        print(f"[model heads] offset={has_offset_head} velocity={has_velocity_head}")

    # load initial weights (pretrained supervised model)
    load_checkpoint(args.init_ckpt, model, optimizer=None, scheduler=None, map_location="cpu")

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, map_location="cpu")

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if ddp else model

    opt = torch.optim.Adam(raw_model.parameters(), lr=train_cfg.lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=train_cfg.lr_decay_every_steps, gamma=train_cfg.lr_decay_gamma)

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sch.load_state_dict(ckpt["scheduler"])

    sup_cfg = SupervisedLossConfig(
        use_weighted_frame_loss=False,
        lambda_on=1.0, lambda_fr=1.0, lambda_off=1.0, lambda_vel=0.0,
    )
    unsup_cfg = UnsupervisedLossConfig(
        lambda_u=args.lambda_u,
        lambda_on=1.0,
        lambda_fr=1.0,
        lambda_off=0.0,
    )

    pseudo_cfg = PseudoLabelConfig(
        tau_lo=args.tau_lo,
        tau_up=args.tau_up,
        use_distribution_matching=not args.no_distribution_matching,
        seed=args.seed,
    )
    aug_cfg = AugmentConfig()

    use_amp = train_cfg.amp and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = LossHistory()

    model.train()
    it_l = iter(labeled_dl)
    it_u = iter(unlabeled_dl)
    epoch_l = 0
    epoch_u = 0
    pbar = tqdm(range(start_step, args.iters), initial=start_step, total=args.iters, disable=(rank != 0))

    best_val = float("inf")

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

        x_l = lb["log_mel"].to(device)
        y_l = {
            "onset": lb["onset"].to(device),
            "frame": lb["frame"].to(device),
            "offset": lb["offset"].to(device),
        }

        x_u = ub["log_mel"].to(device)
        x_u_aug = augment_spectrogram(x_u, aug_cfg)

        # Pseudo labels are generated with eval/no_grad so BN running stats are not
        # updated on this clean branch.
        with torch.no_grad():
            model.eval()
            out_u_clean = model(x_u)
            model.train()
            pseudo, mask = make_pseudo_labels(out_u_clean, pseudo_cfg, desired_ratios=desired_ratios)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            # Use a single train-mode forward pass to avoid BN version conflicts
            # from multiple forwards before one backward.
            x_cat = torch.cat([x_l, x_u_aug], dim=0)
            out_cat = model(x_cat)
            b_l = x_l.size(0)
            out_l = {k: v[:b_l] for k, v in out_cat.items()}
            out_u_aug = {k: v[b_l:] for k, v in out_cat.items()}
            ls = supervised_loss(out_l, y_l, sup_cfg)
            lu = unsupervised_loss(out_u_aug, pseudo, mask, cfg=unsup_cfg)

            loss = (1.0 - unsup_cfg.lambda_u) * ls + unsup_cfg.lambda_u * lu

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        scaler.step(opt)
        scaler.update()
        sch.step()

        if (step + 1) % args.log_every == 0:
            l_val, ls_val, lu_val = loss.item(), ls.item(), lu.item()
            pbar.set_description(f"loss={l_val:.4f} ls={ls_val:.4f} lu={lu_val:.4f}")
            if rank == 0:
                history.append(step + 1, train_loss=l_val, sup_loss=ls_val, unsup_loss=lu_val)

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
            compute_val_metrics = ((step + 1) == args.iters)
            with torch.no_grad():
                loss_sum = 0.0
                n_batches = 0.0
                onset_tp = onset_fp = onset_fn = 0.0
                frame_tp = frame_fp = frame_fn = 0.0
                for vb in valid_dl:
                    vmel = vb["log_mel"].to(device)
                    vlabels = {
                        "onset": vb["onset"].to(device),
                        "frame": vb["frame"].to(device),
                        "offset": vb["offset"].to(device),
                    }
                    vout = model(vmel)
                    vloss = supervised_loss(vout, vlabels, sup_cfg)
                    loss_sum += float(vloss.item())
                    n_batches += 1.0
                    if compute_val_metrics:
                        tp, fp, fn = _binary_counts_from_logits(vout["onset"], vlabels["onset"])
                        onset_tp += tp
                        onset_fp += fp
                        onset_fn += fn
                        tp, fp, fn = _binary_counts_from_logits(vout["frame"], vlabels["frame"])
                        frame_tp += tp
                        frame_fp += fp
                        frame_fn += fn

                val = loss_sum / max(n_batches, 1.0)
                onset_f1 = 0.0
                frame_f1 = 0.0
                if compute_val_metrics:
                    onset_p = onset_tp / max(onset_tp + onset_fp, 1.0)
                    onset_r = onset_tp / max(onset_tp + onset_fn, 1.0)
                    onset_f1 = (2.0 * onset_p * onset_r / max(onset_p + onset_r, 1e-12))
                    frame_p = frame_tp / max(frame_tp + frame_fp, 1.0)
                    frame_r = frame_tp / max(frame_tp + frame_fn, 1.0)
                    frame_f1 = (2.0 * frame_p * frame_r / max(frame_p + frame_r, 1e-12))

            if ddp:
                val_t = torch.tensor(
                    [loss_sum, n_batches, onset_tp, onset_fp, onset_fn, frame_tp, frame_fp, frame_fn],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(val_t, op=dist.ReduceOp.SUM)
                val = float(val_t[0] / val_t[1]) if val_t[1] > 0 else float("inf")
                if compute_val_metrics:
                    onset_tp, onset_fp, onset_fn = float(val_t[2]), float(val_t[3]), float(val_t[4])
                    frame_tp, frame_fp, frame_fn = float(val_t[5]), float(val_t[6]), float(val_t[7])
                    onset_p = onset_tp / max(onset_tp + onset_fp, 1.0)
                    onset_r = onset_tp / max(onset_tp + onset_fn, 1.0)
                    onset_f1 = (2.0 * onset_p * onset_r / max(onset_p + onset_r, 1e-12))
                    frame_p = frame_tp / max(frame_tp + frame_fp, 1.0)
                    frame_r = frame_tp / max(frame_tp + frame_fn, 1.0)
                    frame_f1 = (2.0 * frame_p * frame_r / max(frame_p + frame_r, 1e-12))

            if rank == 0:
                if compute_val_metrics:
                    history.append(
                        step + 1,
                        val_loss=val,
                        val_onset_f1=float(onset_f1),
                        val_frame_f1=float(frame_f1),
                    )
                    print(
                        f"[val step {step+1}] loss={val:.4f} "
                        f"onset_f1={onset_f1:.4f} frame_f1={frame_f1:.4f}"
                    )
                else:
                    history.append(step + 1, val_loss=val)
                if val < best_val:
                    best_val = val
                    extra = {"val_loss": val}
                    if compute_val_metrics:
                        extra["val_onset_f1"] = float(onset_f1)
                        extra["val_frame_f1"] = float(frame_f1)
                    save_checkpoint(
                        out_dir / "checkpoints" / "best.pt",
                        raw_model,
                        opt,
                        sch,
                        step=step + 1,
                        extra=extra,
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
