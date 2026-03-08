# my_mt3/train_iter.py
# Iteration-based (step-based) supervised training loop.
# Epoch-based version: train.py

import math
import os
import csv as _csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .model import MT3Mini
from .tokenizer import Vocab, INPUT_FRAMES
from .dataset import AMTDataset
from .audio import DEFAULT_SR
from .train import _maybe_cache_pairs_map, make_collate, eval_loop_ddp
from .plot_utils import plot_losses_supervised


def train_loop_distributed_iter(
    pairs,
    iters=50_000,
    bs=8,
    lr=2e-4,
    *,
    lr_warmup_steps: int = 0,
    lr_min_ratio: float = 0.1,
    save_every=0,
    ckpt_every=1000,
    valid_every=2000,
    log_every=50,
    keep_last_n: int = 0,
    grad_clip: float = 0.0,
    save_dir="checkpoints",
    use_cache: bool = True,
    cache_dir: str = "cache/wave_sr16000",
    sr: int = DEFAULT_SR,
    num_workers: int = 2,
    vocab: Vocab,
    input_frames: int = INPUT_FRAMES,
    pretrained_ckpt: str | None = None,
):
    # ---- init DDP ----
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)

    rank = dist.get_rank()
    world = dist.get_world_size()

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        if use_cache and cache_dir:
            cache_dir_abs = os.path.abspath(cache_dir)
            os.makedirs(cache_dir_abs, exist_ok=True)
            cache_dir = cache_dir_abs
    dist.barrier()

    # ---- broadcast cache_dir ----
    if use_cache and cache_dir:
        cache_dir_bytes = os.path.abspath(cache_dir).encode("utf-8") if rank == 0 else b""
        length_tensor = torch.tensor([len(cache_dir_bytes)], dtype=torch.int32, device=device)
        dist.broadcast(length_tensor, src=0)
        buf = torch.empty((int(length_tensor.item()),), dtype=torch.uint8, device=device)
        if rank == 0:
            buf.copy_(torch.tensor(list(cache_dir_bytes), dtype=torch.uint8, device=device))
        dist.broadcast(buf, src=0)
        cache_dir = bytes(buf.cpu().numpy().tolist()).decode("utf-8")
    pairs = _maybe_cache_pairs_map(pairs, sr=sr, cache_dir=(cache_dir if use_cache else None))

    # ---- dataset ----
    train_ds = AMTDataset(pairs["train"], mode="train", sr=sr, input_frames=input_frames, vocab=vocab)
    val_ds = AMTDataset(pairs["validation"], mode="validation", sr=sr, input_frames=input_frames, vocab=vocab)

    # ---- sampler / loader ----
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=make_collate(vocab),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=bs,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=make_collate(vocab),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # ---- model ----
    model = MT3Mini(vocab_size=len(vocab.itos)).to(device)
    if pretrained_ckpt is not None:
        ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=True)
        state = ckpt if not isinstance(ckpt, dict) else ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if rank == 0:
            print(f"[pretrained] loaded {pretrained_ckpt}")
            if missing:
                print(f"  missing keys : {missing}")
            if unexpected:
                print(f"  unexpected keys: {unexpected}")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    raw_model = model.module

    opt = optim.AdamW(raw_model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=vocab.pad)

    # ---- LR scheduler (warmup + cosine decay) ----
    warmup_steps = max(0, lr_warmup_steps)
    min_ratio = min(max(float(lr_min_ratio), 0.0), 1.0)

    def _lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        if iters <= warmup_steps:
            return 1.0
        progress = float(step_idx - warmup_steps) / float(iters - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    if rank == 0:
        print(f"[DDP-iter] world={world} | train songs={len(train_ds)} | val songs={len(val_ds)}")
        with open(os.path.join(save_dir, "losses.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["step", "train_loss", "val_loss"])

    # ---- training loop (step-based) ----
    start_step = 0
    model.train()
    dl_iter = iter(train_dl)
    epoch_counter = 0
    running_loss = 0.0
    running_count = 0
    best_val_loss = float("inf")

    pbar = tqdm(range(start_step, iters), initial=start_step, total=iters, disable=(rank != 0))

    for step in pbar:
        try:
            mels, y_in, y_tg = next(dl_iter)
        except StopIteration:
            epoch_counter += 1
            train_sampler.set_epoch(epoch_counter)
            dl_iter = iter(train_dl)
            mels, y_in, y_tg = next(dl_iter)

        mels = mels.to(device, non_blocking=True)
        y_in = y_in.to(device, non_blocking=True)
        y_tg = y_tg.to(device, non_blocking=True)

        logits = model(mels, y_in)
        loss = crit(logits.reshape(-1, logits.size(-1)), y_tg.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        scheduler.step()

        running_loss += float(loss.item())
        running_count += 1

        if (step + 1) % log_every == 0:
            avg = running_loss / max(running_count, 1)
            pbar.set_description(f"loss={avg:.4f}")
            running_loss = 0.0
            running_count = 0

        # ---- checkpoint: last.pt ----
        if rank == 0 and (step + 1) % ckpt_every == 0:
            torch.save(raw_model.state_dict(), os.path.join(save_dir, "last.pt"))

        # ---- numbered checkpoint ----
        if rank == 0 and save_every > 0 and (step + 1) % save_every == 0:
            numbered = os.path.join(save_dir, f"step_{step + 1}.pt")
            torch.save(raw_model.state_dict(), numbered)
            if keep_last_n > 0:
                import glob as _glob
                existing = sorted(_glob.glob(os.path.join(save_dir, "step_*.pt")))
                for old in existing[:-keep_last_n]:
                    os.remove(old)

        # ---- validation ----
        if (step + 1) % valid_every == 0 or (step + 1) == iters:
            dist.barrier()
            val_loss = eval_loop_ddp(model, val_dl, crit, device)
            if rank == 0:
                print(f"[step {step + 1}] val_loss={val_loss:.4f}")
                with open(os.path.join(save_dir, "losses.csv"), "a", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow([step + 1, "", f"{val_loss:.6f}"])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(raw_model.state_dict(), os.path.join(save_dir, "best.pt"))
                    print(f"  -> best model saved (val_loss={best_val_loss:.4f})")
                plot_losses_supervised(save_dir)
            model.train()

    # ---- final save ----
    if rank == 0:
        torch.save(raw_model.state_dict(), os.path.join(save_dir, "last.pt"))
        print("Done.")

    dist.destroy_process_group()
    return model
