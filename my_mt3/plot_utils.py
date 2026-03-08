# my_mt3/plot_utils.py
# Incremental loss plotting utilities for iter-based training loops.

from __future__ import annotations

import csv
import os
from pathlib import Path


def _safe_float(v: str) -> float | None:
    if not v:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def plot_losses_da(save_dir: str, fig_name: str = "da_losses.png") -> None:
    """Read da_losses.csv and save a plot.

    Columns: step, train_total, train_sup, train_adv, train_unsup,
             train_disc, val_loss, [val_token_acc, pseudo_chunks, pseudo_notes]
    """
    csv_path = os.path.join(save_dir, "da_losses.csv")
    if not os.path.exists(csv_path):
        return

    steps_train, total, sup, adv, unsup = [], [], [], [], []
    steps_val, val_l = [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row["step"])
            t = _safe_float(row.get("train_total", ""))
            v = _safe_float(row.get("val_loss", ""))
            if t is not None:
                steps_train.append(s)
                total.append(t)
                sup.append(_safe_float(row.get("train_sup", "")) or 0.0)
                adv.append(_safe_float(row.get("train_adv", "")) or 0.0)
                unsup.append(_safe_float(row.get("train_unsup", "")) or 0.0)
            if v is not None:
                steps_val.append(s)
                val_l.append(v)

    if not steps_train and not steps_val:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        if steps_train:
            ax.plot(steps_train, total, label="total", alpha=0.8)
            ax.plot(steps_train, sup, label="supervised", alpha=0.7)
            if any(v > 0 for v in unsup):
                ax.plot(steps_train, unsup, label="unsup (pseudo)", alpha=0.7)
            if any(v > 0 for v in adv):
                ax.plot(steps_train, adv, label="adversarial", alpha=0.7)
        if steps_val:
            ax.plot(steps_val, val_l, label="val_loss", linestyle="--", linewidth=2)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, fig_name), dpi=150)
        plt.close(fig)
    except Exception:
        pass


def plot_losses_supervised(save_dir: str, fig_name: str = "losses.png") -> None:
    """Read losses.csv (step, train_loss, val_loss) and save a plot."""
    csv_path = os.path.join(save_dir, "losses.csv")
    if not os.path.exists(csv_path):
        return

    steps_train, train_l = [], []
    steps_val, val_l = [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row["step"])
            t = _safe_float(row.get("train_loss", ""))
            v = _safe_float(row.get("val_loss", ""))
            if t is not None:
                steps_train.append(s)
                train_l.append(t)
            if v is not None:
                steps_val.append(s)
                val_l.append(v)

    if not steps_train and not steps_val:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        if steps_train:
            ax.plot(steps_train, train_l, label="train_loss", alpha=0.8)
        if steps_val:
            ax.plot(steps_val, val_l, label="val_loss", linestyle="--", linewidth=2)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, fig_name), dpi=150)
        plt.close(fig)
    except Exception:
        pass
