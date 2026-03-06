"""
MLM pre-training for the Discriminator Transformer encoder.

Masks 15 % of teacher-MIDI tokens (BERT-style) and trains the model to
reconstruct them.  The resulting encoder weights can be loaded by
train_conf_clf.py via --pretrained_ckpt for downstream fine-tuning.

Single GPU:
    python Discriminator/train_conf_mlm.py \
        --train_midi_root dataset/maestro-v3.0.0 \
        --max_steps 5000 --batch_size 128

DDP (multi-GPU):
    torchrun --nproc_per_node=4 Discriminator/train_conf_mlm.py \
        --train_midi_root dataset/maestro-v3.0.0 \
        --max_steps 5000 --batch_size 128
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from conf_clf_model import ConfClfCfg, TransformerConfidenceClf
from midi_tokenizer import MidiTokCfg, AugCfg
from conf_data import list_midi_files, MidiTokenMLMDataset


def _is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def save_ckpt(path: Path, model, opt, step: int, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(
        {"step": step, "model": state, "opt": opt.state_dict(), "cfg": cfg},
        str(path),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_midi_root", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=512)

    # model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    # tokenizer
    ap.add_argument("--time_step_sec", type=float, default=0.01)
    ap.add_argument("--max_shift_steps", type=int, default=100)

    # augmentation
    ap.add_argument("--pitch_shift_min", type=int, default=-5)
    ap.add_argument("--pitch_shift_max", type=int, default=5)
    ap.add_argument("--pitch_shift_prob", type=float, default=0.8)
    ap.add_argument("--time_scale_min", type=float, default=0.9)
    ap.add_argument("--time_scale_max", type=float, default=1.1)
    ap.add_argument("--time_scale_prob", type=float, default=0.8)

    # MLM
    ap.add_argument("--mask_prob", type=float, default=0.15)

    # training
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--windows_per_file", type=int, default=8)

    ap.add_argument("--out_dir", type=str, default="ckpt_conf_mlm")
    args = ap.parse_args()

    # ---- DDP or single-GPU setup ----
    use_ddp = _is_distributed()
    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", device_id=device)
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world = 1

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    tok_cfg = MidiTokCfg(time_step_sec=args.time_step_sec, max_shift_steps=args.max_shift_steps)
    vocab_size = tok_cfg.vocab_size()

    midi_files = list_midi_files(args.train_midi_root)
    if rank == 0:
        print(f"[MLM] world={world} | MIDI files: {len(midi_files)} | vocab: {vocab_size} | mask_prob: {args.mask_prob}")

    aug_cfg = AugCfg(
        pitch_shift_min=args.pitch_shift_min,
        pitch_shift_max=args.pitch_shift_max,
        pitch_shift_prob=args.pitch_shift_prob,
        time_scale_min=args.time_scale_min,
        time_scale_max=args.time_scale_max,
        time_scale_prob=args.time_scale_prob,
    )

    ds = MidiTokenMLMDataset(
        midi_files,
        tok_cfg=tok_cfg,
        aug_cfg=aug_cfg,
        max_len=args.max_len,
        windows_per_file=args.windows_per_file,
        mask_prob=args.mask_prob,
    )

    sampler = DistributedSampler(ds, shuffle=True) if use_ddp else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    mcfg = ConfClfCfg(
        vocab_size=vocab_size,
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_id=tok_cfg.pad_id,
        cls_id=tok_cfg.cls_id,
    )
    model = TransformerConfidenceClf(mcfg).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[device])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    csv_path = out_dir / "losses.csv"
    if rank == 0:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "acc"])

    model.train()
    it = iter(dl)
    t0 = time.time()
    step = 0
    epoch = 0

    pbar = tqdm(total=args.max_steps, desc="MLM pre-train", unit="step", disable=(rank != 0))
    while step < args.max_steps:
        try:
            masked_tokens, attn, labels = next(it)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(dl)
            masked_tokens, attn, labels = next(it)

        masked_tokens = masked_tokens.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model.module.forward_mlm(masked_tokens, attn_mask=attn) if use_ddp else model.forward_mlm(masked_tokens, attn_mask=attn)
        loss = ce(logits.view(-1, vocab_size), labels.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        params = model.module.parameters() if use_ddp else model.parameters()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        step += 1
        pbar.update(1)

        if rank == 0 and step % args.log_every == 0:
            with torch.no_grad():
                mask_pos = labels != -100
                if mask_pos.any():
                    preds = logits.argmax(dim=-1)
                    acc = (preds[mask_pos] == labels[mask_pos]).float().mean().item()
                else:
                    acc = 0.0
            loss_val = loss.item()
            dt = time.time() - t0
            print(f"step {step:6d} | loss={loss_val:.4f} | acc={acc:.3f} | {dt / args.log_every:.3f}s/it")
            pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc:.3f}")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{loss_val:.6f}", f"{acc:.4f}"])
            t0 = time.time()

        if rank == 0 and step % args.save_every == 0:
            save_ckpt(
                out_dir / f"mlm_step{step}.pt", model, opt, step,
                cfg=vars(args) | {"vocab_size": vocab_size},
            )
            print(f"saved: {out_dir / f'mlm_step{step}.pt'}")

    pbar.close()
    if rank == 0:
        save_ckpt(
            out_dir / "mlm_final.pt", model, opt, step,
            cfg=vars(args) | {"vocab_size": vocab_size},
        )
        print(f"done: {out_dir / 'mlm_final.pt'}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
