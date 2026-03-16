from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from my_mt3.dataset import AMTDataset
from my_mt3.model import MT3Mini
from my_mt3.tokenizer import build_vocab
from my_mt3.train import make_collate


def collect_maestro_pairs(
    root: str | Path,
    *,
    split: str = "validation",
    max_songs: int = 8,
    program_id: int = 0,
) -> List[Tuple[str, str, int]]:
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    rows = df[df["split"] == split]
    pairs: List[Tuple[str, str, int]] = []
    for _, row in rows.iterrows():
        wav = root / str(row["audio_filename"])
        mid = root / str(row["midi_filename"])
        if not wav.exists() or not mid.exists():
            continue
        pairs.append((str(wav), str(mid), int(program_id)))
        if max_songs > 0 and len(pairs) >= max_songs:
            break
    return pairs


@torch.no_grad()
def evaluate_audio_dependency(
    model: MT3Mini,
    dl: DataLoader,
    *,
    pad_id: int,
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    model.eval()
    loss_clean_sum = 0.0
    loss_rand_sum = 0.0
    token_count = 0
    n_batches = 0

    for mels, y_in, y_tg in dl:
        mels = mels.to(device, non_blocking=True)
        y_in = y_in.to(device, non_blocking=True)
        y_tg = y_tg.to(device, non_blocking=True)

        bsz = mels.size(0)
        if bsz >= 2:
            perm = torch.randperm(bsz, device=device)
            if torch.all(perm == torch.arange(bsz, device=device)):
                perm = torch.roll(perm, shifts=1)
            mels_rand = mels[perm]
        else:
            mels_rand = torch.randn_like(mels)

        logits_clean = model(mels, y_in)
        logits_rand = model(mels_rand, y_in)

        valid = (y_tg != pad_id)
        n_tok = int(valid.sum().item())
        if n_tok == 0:
            continue

        l_clean = F.cross_entropy(
            logits_clean.reshape(-1, logits_clean.size(-1)),
            y_tg.reshape(-1),
            ignore_index=pad_id,
            reduction="sum",
        )
        l_rand = F.cross_entropy(
            logits_rand.reshape(-1, logits_rand.size(-1)),
            y_tg.reshape(-1),
            ignore_index=pad_id,
            reduction="sum",
        )

        loss_clean_sum += float(l_clean.item())
        loss_rand_sum += float(l_rand.item())
        token_count += n_tok
        n_batches += 1

        if max_batches > 0 and n_batches >= max_batches:
            break

    if token_count == 0:
        raise RuntimeError("No valid tokens were evaluated.")

    clean = loss_clean_sum / token_count
    rand = loss_rand_sum / token_count
    return {
        "batches": n_batches,
        "tokens": token_count,
        "loss_clean": clean,
        "loss_rand_audio": rand,
        "delta": rand - clean,
        "ratio": rand / max(clean, 1e-12),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check whether decoder uses audio by randomizing encoder input."
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--root", type=str, default="dataset/maestro-v3.0.0")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--max_songs", type=int, default=8, help="number of songs to sample (0=all)")
    ap.add_argument("--max_batches", type=int, default=20, help="number of mini-batches to evaluate (0=all)")
    ap.add_argument("--bs", type=int, default=2, help="song batch size for dataset loader")
    ap.add_argument("--input_frames", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

    pairs = collect_maestro_pairs(args.root, split=args.split, max_songs=int(args.max_songs), program_id=0)
    if not pairs:
        raise SystemExit("No pairs found.")

    vocab = build_vocab(input_frames=int(args.input_frames), instrument_type="piano", include_note_off=True)
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
        num_workers=0,
        pin_memory=False,
    )

    model = MT3Mini(vocab_size=len(vocab.itos)).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)

    res = evaluate_audio_dependency(
        model,
        dl,
        pad_id=int(vocab.pad),
        device=device,
        max_batches=int(args.max_batches),
    )

    print("=== Audio Dependency Check ===")
    print(f"ckpt: {args.ckpt}")
    print(f"split: {args.split}  songs_used: {len(pairs)}  batches_used: {res['batches']}  tokens: {res['tokens']}")
    print(f"loss(clean audio):      {res['loss_clean']:.6f}")
    print(f"loss(randomized audio): {res['loss_rand_audio']:.6f}")
    print(f"delta(rand-clean):      {res['delta']:.6f}")
    print(f"ratio(rand/clean):      {res['ratio']:.4f}")


if __name__ == "__main__":
    main()
