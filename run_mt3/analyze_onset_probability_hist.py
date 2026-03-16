from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from my_mt3.dataset import AMTDataset
from my_mt3.model import MT3Mini
from my_mt3.tokenizer import Vocab, build_vocab
from my_mt3.train import make_collate


def collect_maestro_pairs(
    root: str | Path,
    *,
    split: str = "validation",
    max_songs: int = 0,
    program_id: int = 0,
) -> list[tuple[str, str, int]]:
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    rows = df[df["split"] == split]
    pairs: list[tuple[str, str, int]] = []
    for _, row in rows.iterrows():
        wav = root / str(row["audio_filename"])
        mid = root / str(row["midi_filename"])
        if not wav.exists() or not mid.exists():
            continue
        pairs.append((str(wav), str(mid), int(program_id)))
        if max_songs > 0 and len(pairs) >= max_songs:
            break
    return pairs


def _load_model(ckpt_path: str | Path, *, vocab_size: int, device: torch.device) -> MT3Mini:
    model = MT3Mini(vocab_size=vocab_size).to(device)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(
            f"[warn] unexpected keys ({len(unexpected)}): "
            f"{unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}"
        )
    model.eval()
    return model


@torch.no_grad()
def collect_onset_probabilities(
    model: MT3Mini,
    dl: DataLoader,
    *,
    vocab: Vocab,
    device: torch.device,
    max_batches: int = 0,
    condition: str = "after_time",
) -> np.ndarray:
    note_on_ids = torch.tensor(list(vocab.note_on.values()), dtype=torch.long, device=device)
    time_ids = set(vocab.time.values())
    pad_id = int(vocab.pad)

    values: list[np.ndarray] = []
    n_batches = 0

    for mels, y_in, y_tg in dl:
        mels = mels.to(device, non_blocking=True)
        y_in = y_in.to(device, non_blocking=True)
        y_tg = y_tg.to(device, non_blocking=True)

        logits = model(mels, y_in)  # [B, S, V]
        probs = torch.softmax(logits, dim=-1)  # [B, S, V]
        p_on = probs.index_select(dim=2, index=note_on_ids).sum(dim=2)  # [B, S]

        valid = y_tg != pad_id
        if condition == "after_time":
            valid = valid & torch.isin(y_in, torch.tensor(list(time_ids), device=device))
        elif condition == "all":
            pass
        else:
            raise ValueError(f"Unknown condition: {condition}")

        if valid.any():
            values.append(p_on[valid].detach().cpu().numpy())

        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break

    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values, axis=0).astype(np.float32)


def _hist_lines(values: np.ndarray, *, bins: Iterable[float]) -> list[str]:
    bins_arr = np.array(list(bins), dtype=np.float32)
    counts, edges = np.histogram(values, bins=bins_arr)
    max_c = max(int(counts.max()), 1)

    lines: list[str] = []
    for i, c in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        bar_n = int(round((c / max_c) * 24))
        bar = "█" * max(bar_n, 1 if c > 0 else 0)
        lines.append(f"{lo:0.1f}-{hi:0.1f} {bar} ({int(c)})")
    return lines


def _save_hist_plot(
    by_label: dict[str, np.ndarray],
    *,
    out_png: Path,
    bins: np.ndarray,
    density: bool,
) -> None:
    plt.figure(figsize=(9, 5))
    for label, vals in by_label.items():
        if vals.size == 0:
            continue
        plt.hist(vals, bins=bins, alpha=0.45, density=density, label=label)
    plt.xlabel("P(note_on | audio, context)")
    plt.ylabel("Density" if density else "Count")
    plt.title("Onset Probability Histogram")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Collect onset probability P(note_on|audio,context) on MAESTRO validation and "
            "compare checkpoints."
        )
    )
    ap.add_argument("--ckpt", action="append", required=True, help="checkpoint path (repeatable)")
    ap.add_argument(
        "--label",
        action="append",
        default=None,
        help="label per checkpoint (repeatable, same count as --ckpt). If omitted, filename stem is used.",
    )
    ap.add_argument("--root", type=str, default="dataset/maestro-v3.0.0")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--input_frames", type=int, default=256)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_songs", type=int, default=0, help="0 means all songs")
    ap.add_argument("--max_batches", type=int, default=0, help="0 means all batches")
    ap.add_argument(
        "--condition",
        type=str,
        default="after_time",
        choices=["after_time", "all"],
        help="after_time: only decoder positions just after TIM_* tokens (frame-like).",
    )
    ap.add_argument("--bin_width", type=float, default=0.1)
    ap.add_argument("--density", action="store_true")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/onset_prob_hist",
        help="directory to save histogram artifacts",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts: list[str] = list(args.ckpt)
    labels: list[str]
    if args.label is None:
        labels = [Path(p).stem for p in ckpts]
    else:
        labels = list(args.label)
        if len(labels) != len(ckpts):
            raise SystemExit("len(--label) must match len(--ckpt)")

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(input_frames=int(args.input_frames), instrument_type="piano", include_note_off=True)
    pairs = collect_maestro_pairs(
        args.root,
        split=args.split,
        max_songs=int(args.max_songs),
        program_id=0,
    )
    if not pairs:
        raise SystemExit("No MAESTRO pairs found.")

    ds = AMTDataset(
        pairs,
        mode="validation",
        sr=16000,
        input_frames=int(args.input_frames),
        vocab=vocab,
    )
    dl = DataLoader(
        ds,
        batch_size=int(args.bs),
        shuffle=False,
        collate_fn=make_collate(vocab),
        num_workers=int(args.num_workers),
        pin_memory=False,
    )

    bw = float(args.bin_width)
    bins = np.arange(0.0, 1.0 + 1e-8, bw, dtype=np.float32)
    if bins.size == 0 or bins[0] != 0.0:
        bins = np.insert(bins, 0, 0.0)
    if bins[-1] < 1.0:
        bins = np.append(bins, 1.0)
    else:
        bins[-1] = 1.0

    by_label: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    for ckpt, label in zip(ckpts, labels):
        print(f"\n=== {label} ===")
        model = _load_model(ckpt, vocab_size=len(vocab.itos), device=device)
        vals = collect_onset_probabilities(
            model,
            dl,
            vocab=vocab,
            device=device,
            max_batches=int(args.max_batches),
            condition=args.condition,
        )
        by_label[label] = vals

        print(f"samples: {vals.size}")
        if vals.size > 0:
            print(
                "stats: "
                f"mean={vals.mean():.6f}  std={vals.std():.6f}  "
                f"p50={np.quantile(vals, 0.5):.6f}  p90={np.quantile(vals, 0.9):.6f}  "
                f"p99={np.quantile(vals, 0.99):.6f}"
            )
        print("histogram:")
        lines = _hist_lines(vals, bins=bins)
        for ln in lines:
            print(ln)

        np.save(out_dir / f"{label}_onset_probs.npy", vals)
        summary[label] = {
            "ckpt": str(ckpt),
            "samples": int(vals.size),
            "mean": float(vals.mean()) if vals.size > 0 else None,
            "std": float(vals.std()) if vals.size > 0 else None,
            "p50": float(np.quantile(vals, 0.5)) if vals.size > 0 else None,
            "p90": float(np.quantile(vals, 0.9)) if vals.size > 0 else None,
            "p99": float(np.quantile(vals, 0.99)) if vals.size > 0 else None,
            "condition": args.condition,
        }

    plot_path = out_dir / "onset_probability_hist.png"
    _save_hist_plot(by_label, out_png=plot_path, bins=bins, density=bool(args.density))
    summary_path = out_dir / "onset_probability_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved plot: {plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
