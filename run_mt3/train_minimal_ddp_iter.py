# run_mt3/train_minimal_ddp_iter.py
#
# Iteration-based (step-based) supervised training on GrooveMIDI.
# Epoch-based version: train_minimal_ddp.py
#
# Usage:
#   python -m torch.distributed.run --nproc_per_node=2 run_mt3/train_minimal_ddp_iter.py \
#     --iters 50000 --bs 16

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os
import re
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import warnings

from my_mt3.train_iter import train_loop_distributed_iter
from my_mt3.tokenizer import build_vocab, INPUT_FRAMES
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


def collect_pairs_groove(
    root="dataset/groove",
    splits=("train", "validation", "test"),
    program_id=0,
    require_exists=True,
    beat_types=None,
) -> Dict[str, List[tuple]]:
    root = Path(root)
    csv_path = root / "info.csv"
    df = pd.read_csv(csv_path)
    out: Dict[str, List[tuple]] = {sp: [] for sp in splits}
    for sp in splits:
        subset = df[df["split"] == sp]
        if beat_types is not None and "style" in df.columns:
            pats = tuple(beat_types)
            def _ok_style(s):
                s = str(s or "")
                return any(re.search(p, s, flags=re.IGNORECASE) for p in pats)
            subset = subset[subset["style"].apply(_ok_style)]
        for audio_rel, midi_rel in zip(subset["audio_filename"], subset["midi_filename"]):
            audio_path = root / str(audio_rel)
            midi_path = root / str(midi_rel)
            if require_exists and (not audio_path.exists() or not midi_path.exists()):
                continue
            if program_id is None:
                out[sp].append((str(audio_path), str(midi_path)))
            else:
                out[sp].append((str(audio_path), str(midi_path), int(program_id)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", type=str, default="pop,rock,funk",
                     help="style filter (comma-separated regex; 'all' for no filter)")
    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_warmup_steps", type=int, default=0)
    ap.add_argument("--lr_min_ratio", type=float, default=0.1)

    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--valid_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--keep_last_n", type=int, default=0)

    ap.add_argument("--grad_clip", type=float, default=0.0,
                     help="max norm for gradient clipping (0=disabled)")

    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()

    raw = (args.styles or "").strip()
    if raw.lower() in ("all", "*") or raw == "":
        styles = None
    else:
        styles = tuple(f"^{s.strip()}" for s in raw.split(",") if s.strip())

    pairs = collect_pairs_groove(beat_types=styles)
    print(f"train: {len(pairs['train'])} | validation: {len(pairs['validation'])} | test: {len(pairs['test'])}")

    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("checkpoints", f"run_{ts}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints -> {save_dir}")

    vocab = build_vocab(input_frames=INPUT_FRAMES, instrument_type="drum", include_note_off=False)

    model = train_loop_distributed_iter(
        pairs,
        iters=args.iters,
        bs=args.bs,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_min_ratio=args.lr_min_ratio,
        save_every=args.save_every,
        ckpt_every=args.ckpt_every,
        valid_every=args.valid_every,
        log_every=args.log_every,
        keep_last_n=args.keep_last_n,
        save_dir=save_dir,
        use_cache=True,
        cache_dir="cache/wave_sr16000",
        sr=16000,
        vocab=vocab,
        grad_clip=args.grad_clip,
    )

    print(f"Training finished -> {save_dir}")

# python -m torch.distributed.run --nproc_per_node=2 run_mt3/train_minimal_ddp_iter.py
