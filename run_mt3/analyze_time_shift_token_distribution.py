from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _time_hist_from_tokens(token_ids_2d: torch.Tensor, vocab: Vocab) -> np.ndarray:
    # token_ids_2d: [B, S]
    num_time = len(vocab.time)
    hist = np.zeros((num_time,), dtype=np.int64)
    for tid, t in ((tid, t) for t, tid in vocab.time.items()):
        c = int((token_ids_2d == int(tid)).sum().item())
        hist[int(t)] = c
    return hist


@torch.no_grad()
def collect_time_hist(
    model: MT3Mini | None,
    dl: DataLoader,
    *,
    vocab: Vocab,
    device: torch.device,
    mode: str,
    max_batches: int = 0,
) -> np.ndarray:
    # mode: "target" or "pred"
    total = np.zeros((len(vocab.time),), dtype=np.int64)
    n_batches = 0

    for mels, y_in, y_tg in dl:
        mels = mels.to(device, non_blocking=True)
        y_in = y_in.to(device, non_blocking=True)
        y_tg = y_tg.to(device, non_blocking=True)

        if mode == "target":
            toks = y_tg
        elif mode == "pred":
            if model is None:
                raise ValueError("model is required for pred mode")
            logits = model(mels, y_in)
            toks = logits.argmax(dim=-1)
        else:
            raise ValueError(f"unknown mode: {mode}")

        total += _time_hist_from_tokens(toks, vocab)
        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break

    return total


def _plot_hist(
    by_label: dict[str, np.ndarray],
    *,
    out_png: Path,
    top_k: int = 80,
) -> None:
    totals = {k: int(v.sum()) for k, v in by_label.items()}
    n_time = max(len(v) for v in by_label.values()) if by_label else 0
    x = np.arange(n_time, dtype=np.int32)

    # 可視性のため、合計頻度上位top_kのみ表示
    score = np.zeros((n_time,), dtype=np.float64)
    for v in by_label.values():
        score += v.astype(np.float64)
    if n_time > top_k:
        idx = np.argsort(score)[-top_k:]
        idx = np.sort(idx)
    else:
        idx = x

    plt.figure(figsize=(12, 6))
    for label, hist in by_label.items():
        denom = max(totals[label], 1)
        y = hist[idx].astype(np.float64) / float(denom)
        plt.plot(idx, y, marker="o", linewidth=1.2, markersize=2.5, label=label)
    plt.xlabel("TIM_k (time token index)")
    plt.ylabel("Relative frequency")
    plt.title("Time-shift token distribution")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze TIM_* token distribution on MAESTRO split")
    ap.add_argument("--ckpt", action="append", default=None, help="checkpoint path (repeatable)")
    ap.add_argument("--label", action="append", default=None, help="label per checkpoint")
    ap.add_argument(
        "--include_target",
        action="store_true",
        help="also include GT target(y_tg) TIM distribution as reference",
    )
    ap.add_argument("--root", type=str, default="dataset/maestro-v3.0.0")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--input_frames", type=int, default=256)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_songs", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="outputs/time_shift_token_dist")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts = list(args.ckpt or [])
    labels = list(args.label or [])
    if ckpts:
        if labels and len(labels) != len(ckpts):
            raise SystemExit("len(--label) must match len(--ckpt)")
        if not labels:
            labels = [Path(p).stem for p in ckpts]
    elif not args.include_target:
        raise SystemExit("Specify at least one --ckpt or --include_target")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    vocab = build_vocab(input_frames=int(args.input_frames), instrument_type="piano", include_note_off=True)
    pairs = collect_maestro_pairs(
        args.root, split=args.split, max_songs=int(args.max_songs), program_id=0
    )
    if not pairs:
        raise SystemExit("No pairs found")

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

    by_label: dict[str, np.ndarray] = {}
    if args.include_target:
        print("\n=== target ===")
        h_tg = collect_time_hist(
            None, dl, vocab=vocab, device=device, mode="target", max_batches=int(args.max_batches)
        )
        by_label["target"] = h_tg
        print(f"tokens={int(h_tg.sum())}")

    for ckpt, label in zip(ckpts, labels):
        print(f"\n=== {label} ===")
        model = _load_model(ckpt, vocab_size=len(vocab.itos), device=device)
        h_pr = collect_time_hist(
            model, dl, vocab=vocab, device=device, mode="pred", max_batches=int(args.max_batches)
        )
        by_label[label] = h_pr
        print(f"tokens={int(h_pr.sum())}")

    # 保存
    out_json = out_dir / "time_shift_token_distribution.json"
    serial = {}
    for label, hist in by_label.items():
        total = int(hist.sum())
        rel = (hist.astype(np.float64) / float(max(total, 1))).tolist()
        serial[label] = {
            "total_time_tokens": total,
            "hist_counts": hist.tolist(),
            "hist_rel": rel,
            "argmax_time_idx": int(np.argmax(hist)) if total > 0 else None,
        }
    out_json.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")

    out_png = out_dir / "time_shift_token_distribution.png"
    _plot_hist(by_label, out_png=out_png, top_k=80)
    print(f"\nSaved JSON: {out_json}")
    print(f"Saved PNG:  {out_png}")


if __name__ == "__main__":
    main()
