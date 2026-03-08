# my_mt3/train_DA_adversial_iter.py
# Iteration-based (step-based) DA adversarial training loop.
# Epoch-based version: train_DA_adversial.py

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

from my_mt3.model import MT3Mini, EMATeacher
from my_mt3.decode_kv import FastDecoderKV, pseudo_label_with_kvcache
from my_mt3.tokenizer import Vocab, INPUT_FRAMES
from my_mt3.dataset import AMTDataset
from my_mt3.dataset_unlabeled import AMTRealDataset
from my_mt3.discriminator import Discriminator, grl
from my_mt3.audio import DEFAULT_SR
from my_mt3.train import _maybe_cache_pairs_map, make_collate
from my_mt3.train_DA_confusion import eval_loop_ddp, make_collate_real, pseudo_chunk_filter

from typing import Dict
from my_mt3.plot_utils import plot_losses_da


def dann_lambda(p: float, gamma: float = 10.0) -> float:
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def train_loop_distributed_DA_adversial_iter(
    pairs,
    *,
    vocab: Vocab,
    # ---- Domain Adaptation (DC) ----
    use_dc: bool = True,
    pairs_real: dict | None = None,
    lambda_adv: float = 1.0,
    lambda_gamma: float = 10.0,
    lambda_warmup_steps: int = 0,
    lr_t: float = 2e-4,
    lr_c: float = 1e-4,
    chunk_frames: int | None = None,
    disc_hidden: int = 256,
    # ---- SSL (pseudo) ----
    use_pseudo: bool = True,
    pseudo_start_step: int = 3000,
    ema_decay: float = 0.999,
    unsup_weight: float = 1.0,
    pseudo_max_len: int = 1024,
    pseudo_threshold: float = -0.5,
    pseudo_topn: int = 0,
    # ---- pretrained ----
    pretrained_ckpt: str | None = None,
    # ---- gradient clipping ----
    grad_clip: float = 0.0,
    # ---- iter-based ----
    iters: int = 50_000,
    bs: int = 8,
    input_frames: int = INPUT_FRAMES,
    lr_warmup_steps: int = 0,
    lr_min_ratio: float = 0.1,
    save_every: int = 0,
    ckpt_every: int = 1000,
    valid_every: int = 2000,
    log_every: int = 50,
    keep_last_n: int = 0,
    save_dir: str = "checkpoints",
    use_cache: bool = True,
    cache_dir: str = "cache/wave_sr16000",
    sr: int = DEFAULT_SR,
    num_workers: int = 2,
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

    # ---- auto chunk_frames ----
    if chunk_frames is None:
        hop = 256
        frames_per_sec = sr / float(hop)
        chunk_frames = max(1, int(round(0.1 * frames_per_sec)))

    # ---- datasets (synth) ----
    train_ds = AMTDataset(pairs["train"], mode="train", sr=sr, input_frames=input_frames, vocab=vocab)
    val_ds = AMTDataset(pairs["validation"], mode="validation", sr=sr, input_frames=input_frames, vocab=vocab)

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_dl = DataLoader(
        train_ds, batch_size=bs, sampler=train_sampler, shuffle=False,
        collate_fn=make_collate(vocab), num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0),
    )
    val_dl = DataLoader(
        val_ds, batch_size=bs, sampler=val_sampler, shuffle=False,
        collate_fn=make_collate(vocab), num_workers=num_workers,
        pin_memory=True, drop_last=False, persistent_workers=(num_workers > 0),
    )

    # ---- real loader ----
    real_dl = None
    real_sampler = None
    if use_dc or use_pseudo:
        if pairs_real is None or "train" not in pairs_real:
            raise ValueError("pairs_real={'train':[wav,...]} required for DC/pseudo")
        real_wavs = pairs_real["train"]
        real_ds = AMTRealDataset(real_wavs, sr=sr, hop=256, input_frames=input_frames, n_fft=2048, n_mels=256)
        real_sampler = DistributedSampler(real_ds, shuffle=True, drop_last=True)
        real_dl = DataLoader(
            real_ds, batch_size=bs, sampler=real_sampler, shuffle=False,
            collate_fn=make_collate_real(), num_workers=num_workers,
            pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0),
        )

    # ---- models ----
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
    model_ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    crit_ce = nn.CrossEntropyLoss(ignore_index=vocab.pad)
    bce = nn.BCEWithLogitsLoss()

    disc_ddp = None
    opt_c = None
    if use_dc:
        disc = Discriminator(d=384, hidden=disc_hidden).to(device)
        disc_ddp = DDP(disc, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        opt_c = optim.AdamW(disc_ddp.parameters(), lr=lr_c)
    opt_t = optim.AdamW(model_ddp.parameters(), lr=lr_t)

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

    sch_t = optim.lr_scheduler.LambdaLR(opt_t, lr_lambda=_lr_lambda)

    ema = EMATeacher(model_ddp.module, decay=ema_decay)

    if rank == 0:
        msg = f"[DDP-DA-adv-iter] world={world} | synth_train={len(train_ds)} | val={len(val_ds)}"
        if real_dl is not None:
            msg += f" | real_wavs={len(real_ds)}"
        if use_dc:
            msg += f" | DC(lambda_adv={lambda_adv}, chunk_frames={chunk_frames})"
        print(msg)
        with open(os.path.join(save_dir, "da_losses.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["step", "train_total", "train_sup", "train_adv", "train_unsup", "train_disc", "val_loss"])

    # ---- training loop (step-based) ----
    model_ddp.train()
    if use_dc and disc_ddp is not None:
        disc_ddp.train()

    dl_iter = iter(train_dl)
    epoch_s = 0
    real_dl_iter = iter(real_dl) if real_dl is not None else None
    epoch_r = 0

    running_total = 0.0
    running_sup = 0.0
    running_adv = 0.0
    running_unsup = 0.0
    running_disc = 0.0
    running_count = 0

    best_val_loss = float("inf")
    pbar = tqdm(range(iters), total=iters, disable=(rank != 0))

    for step in pbar:
        # ---- synth batch ----
        try:
            mels_s, y_in_s, y_tg_s = next(dl_iter)
        except StopIteration:
            epoch_s += 1
            train_sampler.set_epoch(epoch_s)
            dl_iter = iter(train_dl)
            mels_s, y_in_s, y_tg_s = next(dl_iter)

        # ---- real batch ----
        mels_r = None
        if real_dl_iter is not None:
            try:
                mels_r, _real_idxs, _real_starts = next(real_dl_iter)
            except StopIteration:
                epoch_r += 1
                if real_sampler is not None:
                    real_sampler.set_epoch(epoch_r)
                real_dl_iter = iter(real_dl)
                mels_r, _real_idxs, _real_starts = next(real_dl_iter)

        mels_s = mels_s.to(device, non_blocking=True)
        y_in_s = y_in_s.to(device, non_blocking=True)
        y_tg_s = y_tg_s.to(device, non_blocking=True)
        if mels_r is not None:
            mels_r = mels_r.to(device, non_blocking=True)

        # ===== encoder =====
        mem_s = model_ddp.module.enc(mels_s)
        mem_r = model_ddp.module.enc(mels_r) if mels_r is not None else None

        # ========= (A) Discriminator step =========
        loss_disc = torch.zeros((), device=device)
        if use_dc and disc_ddp is not None and mem_r is not None:
            logit_s = disc_ddp(mem_s.detach(), chunk_frames=chunk_frames)
            logit_r = disc_ddp(mem_r.detach(), chunk_frames=chunk_frames)
            loss_disc = bce(logit_s, torch.zeros_like(logit_s)) + bce(logit_r, torch.ones_like(logit_r))
            opt_c.zero_grad(set_to_none=True)
            loss_disc.backward()
            opt_c.step()

        # ========= (B) Student step =========
        if use_dc and disc_ddp is not None:
            for p in disc_ddp.parameters():
                p.requires_grad_(False)

        # (1) supervised
        logits_s = model_ddp.module.dec(y_in_s, mem_s)
        loss_sup = crit_ce(logits_s.reshape(-1, logits_s.size(-1)), y_tg_s.reshape(-1))

        # DANN lambda schedule
        prog = min(1.0, max(0.0, (step + 1) / float(iters)))
        if lambda_warmup_steps > 0 and (step + 1) <= lambda_warmup_steps:
            lambd = 0.0
        else:
            lambd = lambda_adv * dann_lambda(prog, gamma=lambda_gamma)

        # (2) adversarial (DANN+GRL)
        loss_adv = torch.zeros((), device=device)
        if use_dc and disc_ddp is not None and mem_r is not None:
            mem_s_grl = grl(mem_s, lambd)
            mem_r_grl = grl(mem_r, lambd)
            logit_s2 = disc_ddp(mem_s_grl, chunk_frames=chunk_frames)
            logit_r2 = disc_ddp(mem_r_grl, chunk_frames=chunk_frames)
            loss_adv = bce(logit_s2, torch.zeros_like(logit_s2)) + bce(logit_r2, torch.ones_like(logit_r2))

        # (3) pseudo-label
        loss_unsup = torch.zeros((), device=device)
        if use_pseudo and real_dl is not None and (step + 1) >= pseudo_start_step:
            fast_dec = FastDecoderKV(ema.teacher.dec, max_len=pseudo_max_len).to(device).eval()
            out, _pmax, _margin, log_prob = pseudo_label_with_kvcache(
                teacher=ema.teacher,
                fast_dec=fast_dec,
                mel=mels_r,
                program_id=0,
                vocab=vocab,
                max_new_tokens=pseudo_max_len,
                return_with_prefix=False,
            )

            B = out.size(0)
            prg_id = int(vocab.instrument_type["PRG_0"])
            prg = torch.full((B, 1), prg_id, dtype=torch.long, device=out.device)
            y_in_p = torch.cat([prg, out[:, :-1]], dim=1)
            y_tg_p = out

            pad_id = int(vocab.pad)
            eos_id = int(vocab.eos)
            chunk_mask = pseudo_chunk_filter(
                out, log_prob,
                pad_id=pad_id, eos_id=eos_id, device=device,
                pseudo_threshold=pseudo_threshold,
                pseudo_topn=pseudo_topn,
            )

            if chunk_mask.any():
                y_tg_masked = y_tg_p.clone()
                y_tg_masked[~chunk_mask] = pad_id
                logits_r = model_ddp.module.dec(y_in_p.to(device), mem_r)
                loss_unsup = crit_ce(logits_r.reshape(-1, logits_r.size(-1)), y_tg_masked.to(device).reshape(-1))

        loss_total = loss_sup + lambd * loss_adv + unsup_weight * loss_unsup

        opt_t.zero_grad(set_to_none=True)
        loss_total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), grad_clip)
        opt_t.step()
        sch_t.step()

        if use_dc and disc_ddp is not None:
            for p in disc_ddp.parameters():
                p.requires_grad_(True)

        if use_pseudo and (step + 1) >= pseudo_start_step:
            ema.update(model_ddp.module)

        running_total += float(loss_total.item())
        running_sup += float(loss_sup.item())
        running_adv += float(loss_adv.item())
        running_unsup += float(loss_unsup.item())
        running_disc += float(loss_disc.item())
        running_count += 1

        if (step + 1) % log_every == 0 and rank == 0:
            d = max(running_count, 1)
            pbar.set_description(
                f"loss={running_total/d:.4f} sup={running_sup/d:.4f} "
                f"adv={running_adv/d:.4f} unsup={running_unsup/d:.4f} disc={running_disc/d:.4f}"
            )
            running_total = running_sup = running_adv = running_unsup = running_disc = 0.0
            running_count = 0

        # ---- checkpoint: last.pt ----
        if rank == 0 and (step + 1) % ckpt_every == 0:
            torch.save(model_ddp.module.state_dict(), os.path.join(save_dir, "last.pt"))

        # ---- numbered checkpoint ----
        if rank == 0 and save_every > 0 and (step + 1) % save_every == 0:
            numbered = os.path.join(save_dir, f"step_{step + 1}.pt")
            torch.save(model_ddp.module.state_dict(), numbered)
            if use_dc and disc_ddp is not None:
                torch.save(disc_ddp.module.state_dict(), os.path.join(save_dir, f"disc_step_{step + 1}.pt"))
            if keep_last_n > 0:
                import glob as _glob
                existing = sorted(_glob.glob(os.path.join(save_dir, "step_*.pt")))
                for old in existing[:-keep_last_n]:
                    os.remove(old)

        # ---- validation ----
        if (step + 1) % valid_every == 0 or (step + 1) == iters:
            dist.barrier()
            val_loss = eval_loop_ddp(model_ddp, val_dl, crit_ce, device)
            if isinstance(val_loss, dict):
                val_loss = val_loss["val_loss"]
            if rank == 0:
                print(f"[step {step + 1}] val_loss={val_loss:.4f}")
                with open(os.path.join(save_dir, "da_losses.csv"), "a", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow([step + 1, "", "", "", "", "", f"{val_loss:.6f}"])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model_ddp.module.state_dict(), os.path.join(save_dir, "best.pt"))
                    print(f"  -> best model saved (val_loss={best_val_loss:.4f})")
                plot_losses_da(save_dir)
            model_ddp.train()
            if use_dc and disc_ddp is not None:
                disc_ddp.train()

    # ---- final save ----
    if rank == 0:
        torch.save(model_ddp.module.state_dict(), os.path.join(save_dir, "last.pt"))
        if use_dc and disc_ddp is not None:
            torch.save(disc_ddp.module.state_dict(), os.path.join(save_dir, "disc_last.pt"))
        print("Done.")

    dist.destroy_process_group()
    return model_ddp
