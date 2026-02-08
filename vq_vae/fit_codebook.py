from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from vq_vae import DrumVQCfg, DrumDenoiseVQVAE
from groove_midi import GrooveRollCfg, GrooveMIDIRollDataset


@torch.no_grad()
def fit_codebook(model: DrumDenoiseVQVAE, dl: DataLoader, device: str, max_steps: int):
    model.eval()

    # freeze enc/dec
    for p in model.enc.parameters():
        p.requires_grad = False
    for p in model.dec.parameters():
        p.requires_grad = False

    step = 0
    for y_clean in dl:
        y_clean = y_clean.to(device, non_blocking=True)

        # clean から潜在抽出（noisyにしたいならここで corrupt する）
        z_flat = model.encode_to_z_flat(y_clean)   # (B*Tq, d)

        # EMA update only
        model.vq.update_codebook_only(z_flat.detach())

        step += 1
        if step % 50 == 0:
            print(f"[fit_codebook] step={step}")
        if max_steps > 0 and step >= max_steps:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_in", type=str, required=True)
    ap.add_argument("--ckpt_out", type=str, required=True)
    ap.add_argument("--root", type=str, default="dataset/groove")
    ap.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--hop_sec", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device

    cfg = DrumVQCfg(
        n_drums=args.K,
        latent_dim=256,
        n_codes=512,
        n_layers=3,
        ema_decay=0.99,
        beta_commit=0.0,
        gamma_anchor=0.0,
    )

    model = DrumDenoiseVQVAE(cfg).to(device)
    ckpt = torch.load(args.ckpt_in, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    dcfg = GrooveRollCfg(root=args.root, split=args.split, T=args.T, K=args.K, hop_sec=args.hop_sec, chunks_per_file=4)
    ds = GrooveMIDIRollDataset(dcfg)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True)

    fit_codebook(model, dl, device=device, max_steps=args.max_steps)

    out = Path(args.ckpt_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, str(out))
    print("[OK] saved:", out)


if __name__ == "__main__":
    main()
