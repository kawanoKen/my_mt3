# run_mt3/train_maestro_ddp_iter.py
#
# Iteration-based (step-based) supervised training on MAESTRO.
# Epoch-based version: train_maestro_ddp.py
#
# Usage:
#   python -m torch.distributed.run --nproc_per_node=4 run_mt3/train_maestro_ddp_iter.py \
#     --root dataset/maestro-v3.0.0 --iters 50000 --bs 4 --lr 2e-4

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from concurrent import futures

import warnings
from tqdm import tqdm

from my_mt3.train_iter import train_loop_distributed_iter
from my_mt3.tokenizer import build_vocab, INPUT_FRAMES
from my_mt3.audio import ensure_wave_cache

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


def collect_pairs_maestro(root, splits=("train", "validation"), *, program_id=0, require_exists=True):
    import pandas as pd
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    out = {sp: [] for sp in splits}
    for sp in splits:
        subset = df[df["split"] == sp]
        for audio_rel, midi_rel in zip(subset["audio_filename"], subset["midi_filename"]):
            audio_path = root / str(audio_rel)
            midi_path = root / str(midi_rel)
            if require_exists and (not audio_path.exists() or not midi_path.exists()):
                continue
            out[sp].append((str(audio_path), str(midi_path), int(program_id)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="dataset/maestro-v3.0.0")
    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_warmup_steps", type=int, default=0)
    ap.add_argument("--lr_min_ratio", type=float, default=0.1)
    ap.add_argument("--input_frames", type=int, default=INPUT_FRAMES)
    ap.add_argument("--pretrained_ckpt", type=str, default=None)

    ap.add_argument("--save_every", type=int, default=0,
                     help="save numbered checkpoint every N steps (0=off)")
    ap.add_argument("--ckpt_every", type=int, default=1000,
                     help="save last.pt every N steps")
    ap.add_argument("--valid_every", type=int, default=2000,
                     help="run validation every N steps")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--keep_last_n", type=int, default=0)

    ap.add_argument("--grad_clip", type=float, default=0.0,
                     help="max norm for gradient clipping (0=disabled)")

    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--cache_root", type=str, default="cache/wave_sr16000")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--prefetch_cache_workers", type=int, default=0)
    args = ap.parse_args()

    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("checkpoints_maestro", f"run_{ts}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    pairs = collect_pairs_maestro(args.root, splits=("train", "validation"), program_id=0)
    print(f"train pairs: {len(pairs['train'])} | validation pairs: {len(pairs['validation'])}")
    print(f"Checkpoints -> {save_dir}")

    meta = {
        "script": "train_maestro_ddp_iter.py",
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "data": {"train": len(pairs["train"]), "validation": len(pairs["validation"])},
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    cache_dir_maestro = str(Path(args.cache_root) / "maestro-v3.0.0")

    if not args.no_cache:
        all_wavs = [w for (w, _m, _pid) in pairs["train"] + pairs["validation"]
                     if not str(w).endswith(".npy")]

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
        pairs["train"] = [(w2c.get(w, w), m, pid) for (w, m, pid) in pairs["train"]]
        pairs["validation"] = [(w2c.get(w, w), m, pid) for (w, m, pid) in pairs["validation"]]

    vocab = build_vocab(input_frames=args.input_frames, instrument_type="piano", include_note_off=True)

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
        use_cache=False,
        cache_dir=args.cache_root,
        sr=args.sr,
        vocab=vocab,
        input_frames=args.input_frames,
        pretrained_ckpt=args.pretrained_ckpt,
        grad_clip=args.grad_clip,
        num_workers=4,
    )

    print(f"Training finished -> {save_dir}")

# python -m torch.distributed.run --nproc_per_node=4 run_mt3/train_maestro_ddp_iter.py
