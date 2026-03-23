from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot train_sup/train_unsup/val_loss from da_losses.csv")
    ap.add_argument("--csv", type=str, required=True, help="path to da_losses.csv")
    ap.add_argument("--out", type=str, default=None, help="output png path")
    ap.add_argument(
        "--val_nonzero_only",
        action="store_true",
        help="plot val_loss only at epochs where val_loss > 0 (recommended)",
    )
    ap.add_argument("--ymin", type=float, default=None, help="y-axis min")
    ap.add_argument("--ymax", type=float, default=None, help="y-axis max")
    ap.add_argument("--ytick", type=float, default=None, help="y-axis tick interval")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_path = Path(args.out) if args.out else csv_path.with_name("ssl_losses_sup_unsup_val.png")

    df = pd.read_csv(csv_path)
    required = {"epoch", "train_sup", "train_unsup", "val_loss"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    x = df["epoch"].values
    y_sup = df["train_sup"].values
    y_unsup = df["train_unsup"].values
    y_val = df["val_loss"].values

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, y_sup, label="train_sup", linewidth=1.4)
    ax.plot(x, y_unsup, label="train_unsup", linewidth=1.2)

    if args.val_nonzero_only:
        mask = y_val > 0
        ax.plot(x[mask], y_val[mask], label="val_loss(nonzero only)", linewidth=1.6, marker="o", markersize=2.5)
    else:
        ax.plot(x, y_val, label="val_loss", linewidth=1.6)

    # annotate minimum validation loss (on non-zero points when available)
    val_mask = y_val > 0
    if val_mask.any():
        x_val = x[val_mask]
        y_val_nz = y_val[val_mask]
        i_min = int(y_val_nz.argmin())
        ep_min = float(x_val[i_min])
        v_min = float(y_val_nz[i_min])
        ax.scatter([ep_min], [v_min], color="crimson", s=35, zorder=5)
        ax.annotate(
            f"min val_loss={v_min:.4f} @ep{int(ep_min)}",
            xy=(ep_min, v_min),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="crimson",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="crimson", alpha=0.8),
        )

    if args.ymin is not None or args.ymax is not None:
        ymin = args.ymin if args.ymin is not None else ax.get_ylim()[0]
        ymax = args.ymax if args.ymax is not None else ax.get_ylim()[1]
        ax.set_ylim(float(ymin), float(ymax))
    if args.ytick is not None and args.ytick > 0:
        yl = ax.get_ylim()
        ticks = np.arange(yl[0], yl[1] + 1e-9, float(args.ytick))
        ax.set_yticks(ticks)

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("SSL losses: train_sup / train_unsup / val_loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
