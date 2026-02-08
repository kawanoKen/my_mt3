from __future__ import annotations

from pathlib import Path
import time
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vq_vae import DrumVQCfg, DrumDenoiseVQVAE, train_step_phase1
from groove_midi import GrooveRollCfg, GrooveMIDIRollDataset




def save_ckpt(path: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, step: int, cfg: DrumVQCfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "cfg": cfg.__dict__,
        },
        str(path),
    )


def main():
    # デバイス割当て（torchrun時は各rankを専用GPUへ）
    use_cuda = torch.cuda.is_available()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_cuda and world_size > 1:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if use_cuda else "cpu"
    # 表示は1つだけ（torchrun時はrank0のみ）
    is_main = (local_rank == 0)
    if is_main:
        print("device:", device)

    # 日時ディレクトリ
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("ckpt_vae") / f"phase1_run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- config ----
    T = 1024
    K = 9
    cfg = DrumVQCfg(
        n_drums=K,
        latent_dim=256,
        n_codes=512,
        n_layers=3,
        ema_decay=0.99,   # Phase-1では未使用（codebook更新しないので）
        beta_commit=0.0,
        gamma_anchor=0.0,
    )

    # ---- data ----
    dcfg = GrooveRollCfg(root="dataset/groove", split="train", T=T, K=K, hop_sec=0.01, beat_types=(
        r"^pop",          # pop で始まる
        r"^rock",
        r"^funk",
    ), chunks_per_file=4)
    ds = GrooveMIDIRollDataset(dcfg)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # ---- model/opt ----
    model = DrumDenoiseVQVAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=1e-4)

    # ---- training loop ----
    log_every = 50
    save_every = 2000
    max_steps = 20000

    model.train()
    t0 = time.time()
    step = 0
    # online plot buffers (rank0 only)
    hist_steps = []
    hist_loss = []
    hist_rec = []

    it = iter(dl)
    while step < max_steps:
        try:
            y_clean = next(it)
        except StopIteration:
            it = iter(dl)
            y_clean = next(it)

        y_clean = y_clean.to(device, non_blocking=True)  # (B,T,K)

        loss, logs = train_step_phase1(model, opt, y_clean)
        step += 1

        if is_main and step % log_every == 0:
            dt = time.time() - t0
            print(
                f"step {step:6d} | loss={loss:.4f} | rec={logs['rec']:.4f} | {dt/log_every:.3f}s/iter"
            )
            # update history and save plot
            hist_steps.append(step)
            hist_loss.append(float(loss))
            hist_rec.append(float(logs["rec"]))
            try:
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
                # plotting is best-effort; ignore errors in headless envs
                pass
            t0 = time.time()

        if is_main and step % save_every == 0:
            save_ckpt(run_dir / f"phase1_step{step}.pt", model, opt, step, cfg)
            print(f"saved ckpt -> {run_dir / f'phase1_step{step}.pt'}")

    if is_main:
        save_ckpt(run_dir / "phase1_final.pt", model, opt, step, cfg)
        print(f"done -> {run_dir / 'phase1_final.pt'}")


if __name__ == "__main__":
    main()

# python -m torch.distributed.run --nproc_per_node=4 --master_port=1234 vq_vae/train_phase1.py