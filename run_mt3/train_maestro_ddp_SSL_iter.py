#!/usr/bin/env python3
# run_mt3/train_maestro_ddp_SSL_iter.py
#
# Semi-supervised training on MAESTRO (iteration-based):
#   - A small fraction of MAESTRO train is used as labeled data
#   - The rest is treated as unlabeled (pseudo-label with confidence filtering)
#   - Validation split is kept intact for evaluation
#
# Epoch-based version: train_maestro_ddp_SSL.py
#
# Usage:
#   python -m torch.distributed.run --nproc_per_node=4 run_mt3/train_maestro_ddp_SSL_iter.py \
#     --root dataset/maestro-v3.0.0 \
#     --label_frac 0.05 \
#     --iters 50000 --bs 8

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
import warnings
from tqdm import tqdm

from my_mt3.tokenizer import build_vocab, INPUT_FRAMES
from my_mt3.audio import ensure_wave_cache
from my_mt3.train_DA_confusion_iter import train_loop_distributed_DA_confusion_iter
import concurrent.futures as futures
import csv
import json

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


def collect_pairs_maestro_ssl(
    root: str | Path,
    *,
    label_frac: float = 0.05,
    seed: int = 42,
    program_id: int = 0,
    require_exists: bool = True,
) -> Tuple[Dict[str, List[Tuple[str, ...]]], List[str], List[str]]:
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("split", "audio_filename", "midi_filename"):
        if col not in df.columns:
            raise ValueError(f"CSV column '{col}' not found in {csv_path}")

    all_train: List[Tuple[str, str, int]] = []
    val_pairs: List[Tuple[str, str, int]] = []

    for _, row in df.iterrows():
        audio_path = root / str(row["audio_filename"])
        midi_path = root / str(row["midi_filename"])
        if require_exists and (not audio_path.exists() or not midi_path.exists()):
            continue
        entry = (str(audio_path), str(midi_path), int(program_id))
        if row["split"] == "train":
            all_train.append(entry)
        elif row["split"] == "validation":
            val_pairs.append(entry)

    rng = random.Random(seed)
    rng.shuffle(all_train)

    n_labeled = max(1, int(len(all_train) * label_frac))
    labeled = all_train[:n_labeled]
    unlabeled = all_train[n_labeled:]

    unlabeled_wavs = [wav for wav, _midi, _pid in unlabeled]
    unlabeled_midis = [midi for _wav, midi, _pid in unlabeled]

    pairs_labeled = {
        "train": labeled,
        "validation": val_pairs,
    }
    return pairs_labeled, unlabeled_wavs, unlabeled_midis


def load_split_from_csv(
    split_csv: str | Path,
    root: str | Path,
    *,
    program_id: int = 0,
) -> Tuple[Dict[str, List[Tuple[str, ...]]], List[str], List[str]]:
    root = Path(root)
    split_df = pd.read_csv(split_csv)

    labeled: List[Tuple[str, str, int]] = []
    unlabeled_wavs: List[str] = []
    unlabeled_midis: List[str] = []

    for _, row in split_df.iterrows():
        if row["type"] == "labeled":
            labeled.append((str(row["audio_path"]), str(row["midi_path"]), int(program_id)))
        elif row["type"] == "unlabeled":
            unlabeled_wavs.append(str(row["audio_path"]))
            unlabeled_midis.append(str(row.get("midi_path", "")))

    maestro_csv = root / "maestro-v3.0.0.csv"
    val_pairs: List[Tuple[str, str, int]] = []
    if maestro_csv.exists():
        mdf = pd.read_csv(maestro_csv)
        for _, row in mdf[mdf["split"] == "validation"].iterrows():
            audio_path = root / str(row["audio_filename"])
            midi_path = root / str(row["midi_filename"])
            if audio_path.exists() and midi_path.exists():
                val_pairs.append((str(audio_path), str(midi_path), int(program_id)))

    pairs_labeled = {"train": labeled, "validation": val_pairs}
    return pairs_labeled, unlabeled_wavs, unlabeled_midis


def load_maps_csv(
    maps_csv: str | Path,
    *,
    program_id: int = 0,
) -> Tuple[Dict[str, List[Tuple[str, ...]]], List[str], List[str]]:
    df = pd.read_csv(maps_csv)
    train_pairs: List[Tuple[str, str, int]] = []
    val_pairs: List[Tuple[str, str, int]] = []
    for _, row in df.iterrows():
        entry = (str(row["audio_path"]), str(row["midi_path"]), int(program_id))
        if row["split"] == "train":
            train_pairs.append(entry)
        elif row["split"] == "validation":
            val_pairs.append(entry)
    return {"train": train_pairs, "validation": val_pairs}, [], []


def load_maestro_unlabeled(
    root: str | Path,
    *,
    require_exists: bool = True,
) -> Tuple[List[str], List[str]]:
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("split", "audio_filename", "midi_filename"):
        if col not in df.columns:
            raise ValueError(f"CSV column '{col}' not found in {csv_path}")

    unlabeled_wavs: List[str] = []
    unlabeled_midis: List[str] = []
    for _, row in df[df["split"] == "train"].iterrows():
        audio_path = root / str(row["audio_filename"])
        midi_path = root / str(row["midi_filename"])
        if require_exists and (not audio_path.exists() or not midi_path.exists()):
            continue
        unlabeled_wavs.append(str(audio_path))
        unlabeled_midis.append(str(midi_path))
    return unlabeled_wavs, unlabeled_midis


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Semi-supervised MAESTRO training (iteration-based)")
    ap.add_argument("--root", type=str, default="dataset/maestro-v3.0.0",
                    help="MAESTRO v3.0.0 root directory")
    ap.add_argument("--label_frac", type=float, default=0.1,
                    help="fraction of MAESTRO train to use as labeled (0.0-1.0)")
    ap.add_argument("--label_seed", type=int, default=42,
                    help="random seed for labeled/unlabeled split")
    ap.add_argument("--split_csv", type=str, default=None,
                    help="path to a saved ssl_split.csv to reproduce an exact split "
                         "(overrides --label_frac and --label_seed)")
    ap.add_argument("--maps_csv", type=str, default=None,
                    help="path to a MAPS_*_scenario.csv (overrides --root/--split_csv; "
                         "supervised-only if --maps_labeled_maestro_unlabeled is not set)")
    ap.add_argument("--maps_labeled_maestro_unlabeled", action="store_true",
                    help="use MAPS (maps_csv) as labeled data and MAESTRO train (root) as unlabeled data")

    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr_t", type=float, default=2e-4, help="Transformer lr")
    ap.add_argument("--input_frames", type=int, default=INPUT_FRAMES,
                    help="segment length in frames")
    ap.add_argument("--lr_warmup_steps", type=int, default=0,
                    help="number of warmup steps for LR scheduler")
    ap.add_argument("--lr_min_ratio", type=float, default=0.1,
                    help="minimum LR ratio for cosine decay scheduler")

    # Pseudo-label (SSL)
    ap.add_argument("--pseudo_start_step", type=int, default=4000,
                    help="start pseudo-label training from this step")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--unsup_weight", type=float, default=1.0,
                    help="weight for unsupervised pseudo-label loss")
    ap.add_argument("--pseudo_max_len", type=int, default=1024)
    ap.add_argument("--pseudo_threshold", type=float, default=-0.6,
                    help="chunk-level mean log-prob threshold for pseudo-label (used when pseudo_topn=0)")
    ap.add_argument("--pseudo_topn", type=int, default=0,
                    help="select top-N most confident chunks per batch as pseudo-labels "
                         "(0=use threshold mode instead)")
    ap.add_argument("--pseudo_note_target_only", action="store_true",
                    help="after chunk filtering, compute unsupervised loss only on selected pseudo note tokens")
    ap.add_argument("--pseudo_note_onset_only", action="store_true",
                    help="when pseudo_note_target_only is enabled, keep only NOTE_ON tokens for loss")
    ap.add_argument("--pseudo_note_threshold", type=float, default=-0.5,
                    help="note-level mean log-prob threshold for token filtering")

    # Pretrained
    ap.add_argument("--pretrained_ckpt", type=str, default=None,
                    help="path to a pretrained MT3 checkpoint (.pt) to initialise model weights")

    # Oracle filter
    ap.add_argument("--oracle_filter", action="store_true",
                    help="use ground-truth MIDI to filter pseudo-labels (oracle experiment)")
    ap.add_argument("--oracle_metric", type=str, default="note_f",
                    help="evaluation metric for oracle filter (default: note_f)")
    ap.add_argument("--oracle_threshold", type=float, default=0.5,
                    help="minimum metric value to keep a pseudo-label chunk")
    ap.add_argument("--oracle_note_target_only", action="store_true",
                    help="after oracle chunk filtering, compute unsupervised loss only on "
                         "pseudo note tokens matched to GT notes")

    # Saving / logging
    ap.add_argument("--save_every", type=int, default=0,
                    help="save numbered checkpoint every N steps (0=off)")
    ap.add_argument("--ckpt_every", type=int, default=1000,
                    help="save last.pt every N steps")
    ap.add_argument("--valid_every", type=int, default=2000,
                    help="run validation every N steps")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--keep_last_n", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default=None)

    # Training misc
    ap.add_argument("--grad_clip", type=float, default=0.0,
                    help="max norm for gradient clipping (0=disabled)")

    # Cache
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--cache_root", type=str, default="cache/wave_sr16000")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--prefetch_cache_workers", type=int, default=0)
    ap.add_argument("--no_augment", action="store_true",
                    help="disable spectrogram augmentation for SSL pseudo-label student step")
    args = ap.parse_args()

    # Output directory
    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.maps_csv is not None:
            scenario = Path(args.maps_csv).stem
            save_dir = os.path.join("checkpoints_MAPS_SSL_iter", f"run_{ts}_{scenario}")
        else:
            frac_str = f"{args.label_frac:.0%}".replace("%", "pct")
            save_dir = os.path.join("checkpoints_maestro_SSL_iter", f"run_{ts}_frac{frac_str}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    if args.maps_csv is not None:
        print(f"[MAPS] Loading scenario CSV: {args.maps_csv}")
        pairs_labeled, unlabeled_wavs, unlabeled_midis = load_maps_csv(args.maps_csv, program_id=0)
        if args.maps_labeled_maestro_unlabeled:
            print(f"[MAPS+MAESTRO] Loading MAESTRO unlabeled from: {args.root}")
            unlabeled_wavs, unlabeled_midis = load_maestro_unlabeled(args.root)
    elif args.split_csv is not None:
        print(f"[SSL] Loading split from CSV: {args.split_csv}")
        pairs_labeled, unlabeled_wavs, unlabeled_midis = load_split_from_csv(
            args.split_csv, args.root, program_id=0,
        )
    else:
        pairs_labeled, unlabeled_wavs, unlabeled_midis = collect_pairs_maestro_ssl(
            args.root,
            label_frac=args.label_frac,
            seed=args.label_seed,
            program_id=0,
        )

    n_train_labeled = len(pairs_labeled["train"])
    n_unlabeled = len(unlabeled_wavs)
    n_val = len(pairs_labeled["validation"])
    total_train = n_train_labeled + n_unlabeled

    if args.maps_csv is not None:
        if args.maps_labeled_maestro_unlabeled:
            print(f"[MAPS+MAESTRO] labeled_train={n_train_labeled} | unlabeled_maestro={n_unlabeled} | val={n_val}")
        else:
            print(f"[MAPS] train={n_train_labeled} | val={n_val}")
    else:
        total_str = max(total_train, 1)
        print(f"[SSL-iter] label_frac={args.label_frac:.1%} | "
              f"labeled={n_train_labeled} ({n_train_labeled/total_str:.1%}) | "
              f"unlabeled={n_unlabeled} ({n_unlabeled/total_str:.1%}) | "
              f"val={n_val}")

    vocab = build_vocab(input_frames=args.input_frames, instrument_type="piano", include_note_off=True)

    meta = {
        "script": "train_maestro_ddp_SSL_iter.py",
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "data": {
            "labeled_train": n_train_labeled,
            "unlabeled_train": n_unlabeled,
            "total_train": total_train,
            "label_frac_actual": round(n_train_labeled / total_train, 4) if total_train else 0,
            "validation": n_val,
        },
    }
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Meta saved -> {meta_path}")

    split_out = os.path.join(save_dir, "ssl_split.csv")
    with open(split_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "audio_path", "midi_path"])
        for wav, midi, _pid in pairs_labeled["train"]:
            w.writerow(["labeled", wav, midi])
        for wav, midi in zip(unlabeled_wavs, unlabeled_midis):
            w.writerow(["unlabeled", wav, midi])
    print(f"Split info saved -> {split_out}")

    if args.maps_csv is not None:
        cache_subdir = "MAPS_MAESTRO" if args.maps_labeled_maestro_unlabeled else "MAPS"
    else:
        cache_subdir = "maestro-v3.0.0"
    pairs_real = {"train": unlabeled_wavs}
    cache_dir_maestro = str(Path(args.cache_root) / cache_subdir)

    if not args.no_cache:
        all_wavs = (
            [w for (w, _m, _pid) in pairs_labeled["train"] if not str(w).endswith(".npy")]
            + [w for (w, _m, _pid) in pairs_labeled["validation"] if not str(w).endswith(".npy")]
            + [w for w in unlabeled_wavs if not str(w).endswith(".npy")]
        )

        def _prefetch_set(wavs, cache_dir, title):
            if not wavs:
                return []
            if args.prefetch_cache_workers > 0:
                print(f"[cache:{title}] prefetch: {len(wavs)} files -> {cache_dir}")

                def _cache_one(w):
                    try:
                        return ensure_wave_cache(w, cache_dir=cache_dir, sr=args.sr)
                    except Exception as e:
                        return f"ERR:{w}:{e}"

                with futures.ThreadPoolExecutor(max_workers=int(args.prefetch_cache_workers)) as ex:
                    cached = list(tqdm(ex.map(_cache_one, wavs), total=len(wavs),
                                       desc=f"prefetch {title}", unit="wav"))
            else:
                cached = [ensure_wave_cache(w, cache_dir=cache_dir, sr=args.sr) for w in wavs]
            return [c for c in cached if isinstance(c, str) and not c.startswith("ERR:")]

        cached_all = _prefetch_set(all_wavs, cache_dir_maestro, "maestro")
        w2c = dict(zip(all_wavs, cached_all))

        pairs_labeled["train"] = [(w2c.get(w, w), m, pid) for (w, m, pid) in pairs_labeled["train"]]
        pairs_labeled["validation"] = [(w2c.get(w, w), m, pid) for (w, m, pid) in pairs_labeled["validation"]]
        pairs_real = {"train": [w2c.get(w, w) for w in unlabeled_wavs]}

    print(f"Checkpoints -> {save_dir}")

    _ = train_loop_distributed_DA_confusion_iter(
        pairs_labeled,
        vocab=vocab,
        use_dc=False,
        pairs_real=pairs_real,
        lambda_adv=0.0,
        lr_t=args.lr_t,
        lr_c=0.0,
        chunk_frames=None,
        disc_hidden=256,
        use_pseudo=True,
        pseudo_start_step=args.pseudo_start_step,
        ema_decay=args.ema_decay,
        unsup_weight=args.unsup_weight,
        pseudo_max_len=args.pseudo_max_len,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_topn=args.pseudo_topn,
        pretrained_ckpt=args.pretrained_ckpt,
        oracle_filter=args.oracle_filter,
        oracle_metric=args.oracle_metric,
        oracle_threshold=args.oracle_threshold,
        oracle_midi_paths=unlabeled_midis if args.oracle_filter else None,
        oracle_note_target_only=args.oracle_note_target_only,
        pseudo_note_target_only=args.pseudo_note_target_only,
        pseudo_note_onset_only=args.pseudo_note_onset_only,
        pseudo_note_threshold=args.pseudo_note_threshold,
        use_augment=not args.no_augment,
        grad_clip=args.grad_clip,
        iters=args.iters,
        bs=args.bs,
        input_frames=args.input_frames,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_min_ratio=args.lr_min_ratio,
        save_every=args.save_every,
        ckpt_every=args.ckpt_every,
        valid_every=args.valid_every,
        log_every=args.log_every,
        keep_last_n=args.keep_last_n,
        save_dir=save_dir,
        use_cache=False,
        cache_dir=args.cache_root,
        sr=args.sr,
        num_workers=2,
    )

    print(f"Training finished -> {save_dir}")

