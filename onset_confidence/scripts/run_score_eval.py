from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_cfg(path: str) -> dict:
    txt = pathlib.Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception:
        return json.loads(txt)


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U based
    r = np.argsort(np.argsort(s)) + 1
    r_pos = np.sum(r[y == 1])
    n_pos = len(pos)
    n_neg = len(neg)
    u = r_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    idx = np.argsort(-s)
    y = y[idx]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    p = tp / np.maximum(tp + fp, 1)
    r = tp / max(np.sum(y == 1), 1)
    # step-wise integral
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def best_threshold(y: np.ndarray, s: np.ndarray):
    ths = np.unique(s)
    best = (0.5, -1.0, 0.0)
    for th in ths:
        pred = (s >= th).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        if f1 > best[1]:
            acc = float(np.mean(pred == y))
            best = (float(th), float(f1), acc)
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="onset_confidence/conf/default.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)["evaluate"]

    df = pd.read_csv(cfg["input_csv"])
    out_dir = pathlib.Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = [c for c in cfg["score_columns"] if c in df.columns]
    rows = []
    y = df["is_correct"].astype(int).values
    for c in scores:
        sub = df[["is_correct", c]].dropna()
        if len(sub) < 5:
            continue
        yy = sub["is_correct"].astype(int).values
        ss = sub[c].astype(float).values
        th, f1, acc = best_threshold(yy, ss)
        rows.append(
            dict(
                score=c,
                n=len(sub),
                roc_auc=roc_auc(yy, ss),
                pr_auc=pr_auc(yy, ss),
                best_threshold=th,
                best_f1=f1,
                best_acc=acc,
                mean_pos=float(np.mean(ss[yy == 1])) if np.any(yy == 1) else np.nan,
                mean_neg=float(np.mean(ss[yy == 0])) if np.any(yy == 0) else np.nan,
            )
        )
        pos = ss[yy == 1]
        neg = ss[yy == 0]
        plt.figure(figsize=(6, 4))
        plt.hist(pos, bins=50, alpha=0.5, density=True, label=f"pos({len(pos)})")
        plt.hist(neg, bins=50, alpha=0.5, density=True, label=f"neg({len(neg)})")
        plt.title(f"hist: {c}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{c}.png", dpi=140)
        plt.close()
    tab = pd.DataFrame(rows).sort_values("roc_auc", ascending=False) if rows else pd.DataFrame()
    tab.to_csv(out_dir / "score_eval_summary.csv", index=False)
    print(tab.to_string(index=False) if not tab.empty else "No valid score rows")


if __name__ == "__main__":
    main()
