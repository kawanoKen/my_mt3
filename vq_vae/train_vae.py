from __future__ import annotations

from pathlib import Path
import time
import os
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from vq_vae_failed import DrumVQCfg, DrumDenoiseVQVAE, train_step
from groove_midi import GrooveRollCfg, GrooveMIDIRollDataset

class DummyRollDataset(Dataset):
    """
    動作確認用：適当なドラムパターンを生成するだけ。
    本番では GrooveMIDI 等から roll を作る Dataset に置換。
    """
    def __init__(self, n_samples=20000, T=1024, K=9):
        self.n_samples = n_samples
        self.T = T
        self.K = K

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 簡単な4/4パターンを合成（0/1）
        y = torch.zeros(self.T, self.K, dtype=torch.float32)
        # kick: 0, 1/2, snare: 1/4, 3/4, hihat: 1/8
        kick, snare, hihat = 0, 1, 2
        for t in range(0, self.T, 128):   # ざっくり拍
            y[t, kick] = 1.0
        for t in range(64, self.T, 128):
            y[t, snare] = 1.0
        for t in range(0, self.T, 32):
            y[t, hihat] = 1.0
        # ちょいランダム
        if torch.rand(()) < 0.3:
            y[96::128, kick] = 1.0
        return y  # (T,K)


def save_ckpt(path: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, step: int, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
        },
        str(path),
    )


def gamma_anchor_schedule(step: int,
                          warmup_steps: int = 3000,
                          ramp_steps: int = 7000,
                          gamma_target: float = 0.2) -> float:
    if step < warmup_steps:
        return 0.0
    t = (step - warmup_steps) / max(1, ramp_steps)
    t = max(0.0, min(1.0, t))
    return gamma_target * t


def main():
    # ---- run id / csv log ----
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("ckpt_vae") / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "train_log.csv"
    csv_header = [
        "run_ts", "step", "loss", "rec", "anch", "commit",
        "dist_ylogit_anchor",
        "gamma_anchor", "lr", "batch_size", "world_size", "device", "is_ddp", "ckpt_path"
    ]
    def append_csv_row(row: dict):
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_header)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in csv_header})

    # ---- runtime mode detection ----
    use_cuda = torch.cuda.is_available()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_ddp = use_cuda and world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if is_ddp else 0

    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        is_main = (rank == 0)
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
        is_main = True

    if is_main:
        print("device:", device)

    # ---- config ----
    T = 1024
    K = 9
    cfg = DrumVQCfg(
        n_drums=K,
        latent_dim=256,
        n_codes=512,
        n_layers=3,
        ema_decay=0.99,
        beta_commit=0.05,     # ← 最初は小さめ推奨
        gamma_anchor=0.2,     # ← 最初は小さめ推奨
    )

    # ---- data ----
    roll_cfg = GrooveRollCfg(
        root="dataset/groove",
        T=T,
        hop_sec=0.01,
        K=K,
        split="train",              # "validation" / "test" も可
        seed=0,
        include_eval_session=True,
        chunks_per_file=4,
        loop_short=True,
    )

    ds = GrooveMIDIRollDataset(roll_cfg)
    if is_ddp:
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        batch_size = 32
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    else:
        batch_size = 32
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # ---- model/opt ----
    base_model = DrumDenoiseVQVAE(cfg).to(device)
    if is_ddp:
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif use_cuda and torch.cuda.device_count() > 1:
        # Fallback: single-process multi-GPU
        model = nn.DataParallel(base_model).to(device)
    else:
        model = base_model
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=1e-4)

    # ---- training loop ----
    log_every = 50
    save_every = 200
    max_steps = 2000

    model.train()
    t0 = time.time()
    step = 0

    it = iter(dl)
    while step < max_steps:
        if is_ddp:
            # samplerにエポック相当の値を渡すとshuffleシードが変わる
            # 厳密なepochはないので、概ねlog_every単位で更新
            if step % log_every == 0:
                dl.sampler.set_epoch(step // max(1, log_every))
        try:
            y_clean = next(it)
        except StopIteration:
            it = iter(dl)
            y_clean = next(it)

        y_clean = y_clean.to(device, non_blocking=True)  # (B,T,K)
        # スケジューリング
        _m = model.module if hasattr(model, "module") else model
        _m.cfg.gamma_anchor = gamma_anchor_schedule(
            step,
            warmup_steps=1000,
            ramp_steps=max_steps-1000,
            gamma_target=0.2
        )

        loss, logs = train_step(model, opt, y_clean)
        step += 1

        if step % log_every == 0:
            dt = time.time() - t0
            if is_main:
                print(
                    f"step {step:6d} | loss={loss:.4f} "
                    f"| rec={logs['rec']:.4f} anch={logs['anch']:.4f} commit={logs['commit']:.4f} "
                    f"| ya={float(logs.get('dist_ylogit_anchor', float('nan'))):.4f} "
                    f"| {dt/log_every:.3f}s/iter"
                )
                # csv log
                append_csv_row({
                    "run_ts": run_ts,
                    "step": step,
                    "loss": loss,
                    "rec": float(logs["rec"]),
                    "anch": float(logs["anch"]),
                    "commit": float(logs["commit"]),
                    "dist_ylogit_anchor": float(logs.get("dist_ylogit_anchor", float("nan"))),
                    "gamma_anchor": float(model.module.cfg.gamma_anchor if hasattr(model, "module") else model.cfg.gamma_anchor),
                    "lr": float(next(iter(opt.param_groups))["lr"]),
                    "batch_size": batch_size,
                    "world_size": world_size,
                    "device": str(device),
                    "is_ddp": int(is_ddp),
                    "ckpt_path": "",
                })
            t0 = time.time()

        if step % save_every == 0:
            if is_main:
                to_save = model.module if hasattr(model, "module") else model
                ckpt_path = run_dir / f"model_step{step}.pt"
                save_ckpt(ckpt_path, to_save, opt, step, cfg)
                print("saved ckpt")
                # csv log with ckpt path
                append_csv_row({
                    "run_ts": run_ts,
                    "step": step,
                    "loss": loss,
                    "rec": float(logs["rec"]),
                    "anch": float(logs["anch"]),
                    "commit": float(logs["commit"]),
                    "dist_ylogit_anchor": float(logs.get("dist_ylogit_anchor", float("nan"))),
                    "gamma_anchor": float(model.module.cfg.gamma_anchor if hasattr(model, "module") else model.cfg.gamma_anchor),
                    "lr": float(next(iter(opt.param_groups))["lr"]),
                    "batch_size": batch_size,
                    "world_size": world_size,
                    "device": str(device),
                    "is_ddp": int(is_ddp),
                    "ckpt_path": str(ckpt_path),
                })
            if is_ddp:
                dist.barrier()

    if is_main:
        to_save = model.module if hasattr(model, "module") else model
        final_ckpt = run_dir / "model_final.pt"
        save_ckpt(final_ckpt, to_save, opt, step, cfg)
        print("done")
        # final csv line
        append_csv_row({
            "run_ts": run_ts,
            "step": step,
            "loss": loss,
            "rec": float(logs["rec"]),
            "anch": float(logs["anch"]),
            "commit": float(logs["commit"]),
            "dist_ylogit_anchor": float(logs.get("dist_ylogit_anchor", float("nan"))),
            "gamma_anchor": float(model.module.cfg.gamma_anchor if hasattr(model, "module") else model.cfg.gamma_anchor),
            "lr": float(next(iter(opt.param_groups))["lr"]),
            "batch_size": batch_size,
            "world_size": world_size,
            "device": str(device),
            "is_ddp": int(is_ddp),
            "ckpt_path": str(final_ckpt),
        })

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
