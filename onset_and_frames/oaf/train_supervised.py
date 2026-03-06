from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import concurrent.futures as futures
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from oaf.config import FeatureConfig, LabelConfig, ModelConfig, TrainConfig
from oaf.datasets import (
    LabeledPianoDataset,
    read_labeled_csv,
    read_maestro_pairs,
    read_split_labeled_csv,
)
from oaf.features import ensure_wave_cache
from oaf.losses import SupervisedLossConfig, supervised_loss
from oaf.model import OnsetsAndFrames
from oaf.utils import load_checkpoint, save_checkpoint, set_seed, write_json, LossHistory


def _collate(batch):
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def _is_ddp() -> bool:
    return "LOCAL_RANK" in os.environ


def _prefetch_wave_cache(
    wav_paths: List[str],
    *,
    cache_dir: str,
    sr: int,
    workers: int,
) -> Dict[str, str]:
    uniq = []
    seen = set()
    for w in wav_paths:
        if w in seen or str(w).endswith(".npy"):
            continue
        seen.add(w)
        uniq.append(w)

    if not uniq:
        return {}

    def _cache_one(w: str) -> str:
        return ensure_wave_cache(w, cache_dir=cache_dir, sr=sr)

    if workers > 0:
        with futures.ThreadPoolExecutor(max_workers=workers) as ex:
            cached = list(tqdm(ex.map(_cache_one, uniq), total=len(uniq), desc="prefetch cache", unit="wav"))
    else:
        cached = [_cache_one(w) for w in tqdm(uniq, desc="prefetch cache", unit="wav")]
    return dict(zip(uniq, cached))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default=None,
                     help="labeled CSV with columns audio_path,midi_path")
    ap.add_argument("--valid_csv", type=str, default=None)
    ap.add_argument("--maps_csv", type=str, default=None,
                     help="split-aware CSV (e.g. MAPS_*_scenario.csv)")
    ap.add_argument("--maps_train_split", type=str, default="train")
    ap.add_argument("--maps_valid_split", type=str, default="validation")
    ap.add_argument("--maestro_root", type=str, default=None,
                     help="MAESTRO v3 root directory (alternative to --train_csv)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default="cache/wave_sr16000/maestro-v3.0.0")
    ap.add_argument("--prefetch_cache_workers", type=int, default=8,
                     help="workers for pre-creating wave cache before training (0=serial)")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--segment_seconds", type=float, default=20.0)
    ap.add_argument("--num_workers", type=int, default=None,
                     help="override dataloader workers (default: TrainConfig.num_workers)")

    ap.add_argument("--use_weighted_frame_loss", action="store_true")
    ap.add_argument("--use_offset_head", action="store_true")
    ap.add_argument("--use_velocity_head", action="store_true")

    ap.add_argument("--resume", type=str, default=None)

    ap.add_argument("--ckpt_every", type=int, default=200,
                     help="save last.pt every N steps")
    ap.add_argument("--save_every", type=int, default=200,
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

    feat_cfg = FeatureConfig()
    lab_cfg = LabelConfig()
    model_cfg = ModelConfig(
        n_mels=feat_cfg.n_mels,
        n_pitches=lab_cfg.midi_max - lab_cfg.midi_min + 1,
        use_offset_head=args.use_offset_head,
        use_velocity_head=args.use_velocity_head,
    )
    train_cfg = TrainConfig(
        batch_size_labeled=args.batch_size,
        segment_seconds=args.segment_seconds,
        device=str(device),
    )
    if args.num_workers is not None:
        train_cfg.num_workers = int(args.num_workers)

    # --- data loading ---
    if args.maps_csv is not None:
        train_items = read_split_labeled_csv(args.maps_csv, split=args.maps_train_split)
        valid_items = read_split_labeled_csv(args.maps_csv, split=args.maps_valid_split)
        if rank == 0:
            print(
                f"[MAPS CSV] train({args.maps_train_split})={len(train_items)} | "
                f"valid({args.maps_valid_split})={len(valid_items)}"
            )
    elif args.maestro_root is not None:
        train_items = read_maestro_pairs(args.maestro_root, split="train")
        valid_items = read_maestro_pairs(args.maestro_root, split="validation")
        if rank == 0:
            print(f"[MAESTRO] train={len(train_items)} | validation={len(valid_items)}")
    elif args.train_csv is not None:
        train_items = read_labeled_csv(args.train_csv)
        valid_items = read_labeled_csv(args.valid_csv) if args.valid_csv else []
    else:
        raise SystemExit("Specify one of --maps_csv / --maestro_root / --train_csv.")

    # Pre-create wave cache first, then replace wav paths with .npy paths.
    if args.cache_dir is not None:
        all_wavs = [w for w, _m in train_items] + [w for w, _m in valid_items]
        if rank == 0:
            print(f"[cache] prefetch {len(all_wavs)} wavs -> {args.cache_dir}")
            w2c = _prefetch_wave_cache(
                all_wavs,
                cache_dir=args.cache_dir,
                sr=feat_cfg.sample_rate,
                workers=max(0, int(args.prefetch_cache_workers)),
            )
        else:
            w2c = {}
        if ddp:
            dist.barrier()
        # Other ranks lazily resolve the same cache path; rank0 already warmed it.
        if rank == 0:
            train_items = [(w2c.get(w, w), m) for (w, m) in train_items]
            valid_items = [(w2c.get(w, w), m) for (w, m) in valid_items]
        else:
            train_items = [(ensure_wave_cache(w, cache_dir=args.cache_dir, sr=feat_cfg.sample_rate) if not str(w).endswith(".npy") else w, m) for (w, m) in train_items]
            valid_items = [(ensure_wave_cache(w, cache_dir=args.cache_dir, sr=feat_cfg.sample_rate) if not str(w).endswith(".npy") else w, m) for (w, m) in valid_items]

    train_ds = LabeledPianoDataset(
        train_items, feat_cfg, lab_cfg, train_cfg, cache_dir=args.cache_dir,
        compute_velocity=args.use_velocity_head,
    )
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if ddp else None
    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size_labeled,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(train_cfg.num_workers > 0),
        collate_fn=_collate,
    )

    valid_dl = None
    if valid_items:
        valid_ds = LabeledPianoDataset(
            valid_items, feat_cfg, lab_cfg, train_cfg, cache_dir=args.cache_dir,
            compute_velocity=args.use_velocity_head,
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

    model = OnsetsAndFrames(model_cfg).to(device)

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, map_location="cpu")
        if rank == 0:
            remaining = max(0, int(args.iters) - int(start_step))
            print(
                f"[resume] path={args.resume} | start_step={start_step} "
                f"| target_iters={args.iters} | remaining_steps={remaining}"
            )

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

    loss_cfg = SupervisedLossConfig(
        use_weighted_frame_loss=args.use_weighted_frame_loss,
        lambda_on=1.0,
        lambda_fr=1.0,
        lambda_off=1.0 if args.use_offset_head else 0.0,
        lambda_vel=1.0 if args.use_velocity_head else 0.0,
    )

    use_amp = train_cfg.amp and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = LossHistory()
    history_json_path = out_dir / "loss_history.json"
    history_png_path = out_dir / "loss_curve.png"

    model.train()
    dl_iter = iter(train_dl)
    epoch_counter = 0
    pbar = tqdm(range(start_step, args.iters), initial=start_step, total=args.iters, disable=(rank != 0))

    best_val = float("inf")
    last_valid_step = 0

    for step in pbar:
        try:
            batch = next(dl_iter)
        except StopIteration:
            epoch_counter += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_counter)
            dl_iter = iter(train_dl)
            batch = next(dl_iter)

        log_mel = batch["log_mel"].to(device)
        labels = {
            "onset": batch["onset"].to(device),
            "frame": batch["frame"].to(device),
            "offset": batch["offset"].to(device),
        }
        vel_lab = batch.get("velocity")
        if vel_lab is not None:
            vel_lab = vel_lab.to(device)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(log_mel)
            loss = supervised_loss(out, labels, loss_cfg, velocity_labels=vel_lab)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        scaler.step(opt)
        scaler.update()
        sch.step()

        if (step + 1) % args.log_every == 0:
            l_val = loss.item()
            pbar.set_description(f"loss={l_val:.4f}")
            if rank == 0:
                history.append(step + 1, train_loss=l_val)
                history.save(history_json_path)
                history.plot(history_png_path, title="Supervised Training Loss")

        if rank == 0 and (step + 1) % args.ckpt_every == 0:
            save_checkpoint(out_dir / "checkpoints" / "last.pt", raw_model, opt, sch, step=step + 1)

        if rank == 0 and args.save_every > 0 and (step + 1) % args.save_every == 0:
            numbered = out_dir / "checkpoints" / f"step_{step + 1}.pt"
            save_checkpoint(numbered, raw_model, opt, sch, step=step + 1)

        if valid_dl is not None and (step + 1) % args.valid_every == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for vb in valid_dl:
                    vmel = vb["log_mel"].to(device)
                    vlabels = {
                        "onset": vb["onset"].to(device),
                        "frame": vb["frame"].to(device),
                        "offset": vb["offset"].to(device),
                    }
                    vvel = vb.get("velocity")
                    if vvel is not None:
                        vvel = vvel.to(device)
                    vout = model(vmel)
                    vloss = supervised_loss(vout, vlabels, loss_cfg, velocity_labels=vvel)
                    losses.append(float(vloss.item()))
                val = sum(losses) / max(len(losses), 1)

            if ddp:
                val_t = torch.tensor([val, len(losses)], device=device)
                dist.all_reduce(val_t, op=dist.ReduceOp.SUM)
                val = float(val_t[0] / val_t[1]) if val_t[1] > 0 else float("inf")

            if rank == 0:
                history.append(step + 1, val_loss=val)
                last_valid_step = step + 1
                if val < best_val:
                    best_val = val
                    save_checkpoint(out_dir / "checkpoints" / "best.pt", raw_model, opt, sch, step=step + 1, extra={"val_loss": val})
                history.save(history_json_path)
                history.plot(history_png_path, title="Supervised Training Loss")
            model.train()

    # Always record final validation once, even if it does not hit valid_every.
    if valid_dl is not None and last_valid_step != args.iters:
        model.eval()
        with torch.no_grad():
            losses = []
            for vb in valid_dl:
                vmel = vb["log_mel"].to(device)
                vlabels = {
                    "onset": vb["onset"].to(device),
                    "frame": vb["frame"].to(device),
                    "offset": vb["offset"].to(device),
                }
                vvel = vb.get("velocity")
                if vvel is not None:
                    vvel = vvel.to(device)
                vout = model(vmel)
                vloss = supervised_loss(vout, vlabels, loss_cfg, velocity_labels=vvel)
                losses.append(float(vloss.item()))
            val = sum(losses) / max(len(losses), 1)

        if ddp:
            val_t = torch.tensor([val, len(losses)], device=device)
            dist.all_reduce(val_t, op=dist.ReduceOp.SUM)
            val = float(val_t[0] / val_t[1]) if val_t[1] > 0 else float("inf")

        if rank == 0:
            history.append(args.iters, val_loss=val)
            if val < best_val:
                best_val = val
                save_checkpoint(out_dir / "checkpoints" / "best.pt", raw_model, opt, sch, step=args.iters, extra={"val_loss": val})
            history.save(history_json_path)
            history.plot(history_png_path, title="Supervised Training Loss")
        model.train()

    if rank == 0:
        save_checkpoint(out_dir / "checkpoints" / "last.pt", raw_model, opt, sch, step=args.iters)
        history.save(history_json_path)
        history.plot(history_png_path, title="Supervised Training Loss")
        print("Done.")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=4 