from __future__ import annotations

from pathlib import Path
import time
import os
from datetime import datetime
import argparse

import torch
from torch.utils.data import DataLoader
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLOT_OK = True
except Exception:
    _PLOT_OK = False

from vq_vae import DrumVQCfg, DrumDenoiseVQVAE
from groove_pred_dataset import PredVsGTCfg, PredVsGTRollDataset


def save_ckpt(path: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, step: int, cfg: DrumVQCfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
            "opt": opt.state_dict(),
            "cfg": cfg.__dict__,
        },
        str(path),
    )


def main():
    ap = argparse.ArgumentParser(description="Train denoiser from predicted MIDI (noisy) vs groove GT (clean)")
    ap.add_argument("--pred_root", type=str, default="outputs/groove_test_pred")
    ap.add_argument("--gt_root", type=str, default="dataset/groove")
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--hop_sec", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=20000)
    args = ap.parse_args()

    # device / rank
    use_cuda = torch.cuda.is_available()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_cuda and world_size > 1:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if use_cuda else "cpu"
    is_main = (local_rank == 0)
    if is_main:
        print("device:", device)

    # run dir
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("ckpt_vae") / f"frompreds_run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # dataset / loader
    ds_cfg = PredVsGTCfg(
        pred_root=args.pred_root,
        gt_root=args.gt_root,
        T=int(args.T),
        hop_sec=float(args.hop_sec),
        K=int(args.K),
        chunks_per_file=4,
        loop_short=True,
    )
    ds = PredVsGTRollDataset(ds_cfg)
    dl = DataLoader(ds, batch_size=int(args.bs), shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # model (phase-1: rec loss only)
    model_cfg = DrumVQCfg(
        n_drums=int(args.K),
        latent_dim=256,
        n_codes=512,
        n_layers=3,
        ema_decay=0.99,
        beta_commit=0.0,
        gamma_anchor=0.0,
    )
    model = DrumDenoiseVQVAE(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), betas=(0.9, 0.95), weight_decay=1e-4)

    # train
    log_every = int(args.log_every)
    save_every = int(args.save_every)
    max_steps = int(args.max_steps)

    model.train()
    t0 = time.time()
    step = 0
    hist_steps, hist_loss, hist_rec = [], [], []

    it = iter(dl)
    while step < max_steps:
        try:
            x_noisy, y_clean = next(it)  # (B,T,K), (B,T,K)
        except StopIteration:
            it = iter(dl)
            x_noisy, y_clean = next(it)

        x_noisy = x_noisy.to(device, non_blocking=True)
        y_clean = y_clean.to(device, non_blocking=True)

        out = model.forward_phase1(x_noisy)
        loss, logs = model.loss_phase1(out, y_clean)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1

        if is_main and step % log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d} | loss={float(loss):.4f} | rec={float(logs['rec']):.4f} | {dt/log_every:.3f}s/iter")
            if _PLOT_OK:
                hist_steps.append(step)
                hist_loss.append(float(loss))
                hist_rec.append(float(logs["rec"]))
                try:
                    import matplotlib.pyplot as plt  # re-import for some envs
                    plt.figure(figsize=(6, 3))
                    plt.plot(hist_steps, hist_loss, label="loss")
                    plt.plot(hist_steps, hist_rec, label="rec")
                    plt.xlabel("step")
                    plt.ylabel("value")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
                    plt.savefig(run_dir / "figs" / "loss_curve.png", dpi=120)
                    plt.close()
                except Exception:
                    pass
            t0 = time.time()

        if is_main and step % save_every == 0:
            save_ckpt(run_dir / f"frompreds_step{step}.pt", model, opt, step, model_cfg)
            print(f"saved ckpt -> {run_dir / f'frompreds_step{step}.pt'}")

    if is_main:
        save_ckpt(run_dir / "frompreds_final.pt", model, opt, step, model_cfg)
        print(f"done -> {run_dir / 'frompreds_final.pt'}")


if __name__ == "__main__":
    main()

