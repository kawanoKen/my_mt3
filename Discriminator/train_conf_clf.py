from __future__ import annotations

import argparse
from pathlib import Path
import time
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from conf_clf_model import ConfClfCfg, TransformerConfidenceClf
from midi_tokenizer import MidiTokCfg, AugCfg
from conf_data import list_midi_files, MidiTokenWindowBinaryDataset


def save_ckpt(path: Path, model, opt, step: int, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg}, str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_midi_root", type=str, required=True, help="folder containing piano MIDI files")
    ap.add_argument("--max_len", type=int, default=512)

    # model size
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

    # corruption (negative)
    ap.add_argument("--p_delete", type=float, default=0.10)
    ap.add_argument("--p_insert", type=float, default=0.05)
    ap.add_argument("--p_replace", type=float, default=0.10)
    ap.add_argument("--span_shuffle_prob", type=float, default=0.15)

    # training
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--windows_per_file", type=int, default=8)

    ap.add_argument("--out_dir", type=str, default="ckpt_conf_midi")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    tok_cfg = MidiTokCfg(time_step_sec=args.time_step_sec, max_shift_steps=args.max_shift_steps)
    vocab_size = tok_cfg.vocab_size()

    # data
    midi_files = list_midi_files(args.train_midi_root)
    aug_cfg = AugCfg(
        pitch_shift_min=args.pitch_shift_min,
        pitch_shift_max=args.pitch_shift_max,
        pitch_shift_prob=args.pitch_shift_prob,
        time_scale_min=args.time_scale_min,
        time_scale_max=args.time_scale_max,
        time_scale_prob=args.time_scale_prob,
    )
    ds = MidiTokenWindowBinaryDataset(
        midi_files,
        tok_cfg=tok_cfg,
        aug_cfg=aug_cfg,
        max_len=args.max_len,
        windows_per_file=args.windows_per_file,
        corruption_kwargs=dict(
            p_delete=args.p_delete,
            p_insert=args.p_insert,
            p_replace=args.p_replace,
            span_shuffle_prob=args.span_shuffle_prob,
        ),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True)

    # model
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()

    # train loop
    model.train()
    it = iter(dl)
    t0 = time.time()
    step = 0

    pbar = tqdm(total=args.max_steps, desc="training", unit="step")
    while step < args.max_steps:
        try:
            tokens, attn, y = next(it)
        except StopIteration:
            it = iter(dl)
            tokens, attn, y = next(it)

        tokens = tokens.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(tokens, attn_mask=attn)
        loss = bce(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step += 1
        pbar.update(1)

        if step % args.log_every == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()
                acc = (pred == y).float().mean().item()
            dt = time.time() - t0
            print(f"step {step:6d} | loss={loss.item():.4f} | acc={acc:.3f} | {dt/args.log_every:.3f}s/it")
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")
            t0 = time.time()

        if step % args.save_every == 0:
            save_ckpt(out_dir / f"conf_step{step}.pt", model, opt, step, cfg=vars(args) | {"vocab_size": vocab_size})
            print("saved:", out_dir / f"conf_step{step}.pt")

    pbar.close()
    save_ckpt(out_dir / "conf_final.pt", model, opt, step, cfg=vars(args) | {"vocab_size": vocab_size})
    print("done:", out_dir / "conf_final.pt")


if __name__ == "__main__":
    main()
