# run_mt3/train_maestro_ddp_DA_iter.py
#
# Iteration-based DA training (adversarial or confusion) on GiantMIDI-Piano synth + real.
# Epoch-based version: train_maestro_ddp_DA.py
#
# Usage:
#   python -m torch.distributed.run --nproc_per_node=4 run_mt3/train_maestro_ddp_DA_iter.py \
#     --discriminator_mode confusion --iters 50000 --bs 8

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime

import pandas as pd
import warnings
from tqdm import tqdm

from my_mt3.train_DA_adversial_iter import train_loop_distributed_DA_adversial_iter
from my_mt3.train_DA_confusion_iter import train_loop_distributed_DA_confusion_iter
from my_mt3.tokenizer import build_vocab, INPUT_FRAMES
from my_mt3.audio import ensure_wave_cache
import concurrent.futures as futures

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


def collect_pairs_synth_from_giantmidi(midi_root, wav_root, *, program_id=0, require_exists=True):
    midi_root = Path(midi_root)
    wav_root = Path(wav_root)
    if not midi_root.exists():
        raise FileNotFoundError(f"midi_root not found: {midi_root}")
    if not wav_root.exists():
        raise FileNotFoundError(f"wav_root not found: {wav_root}")

    all_midi = sorted(list(midi_root.rglob("*.mid")) + list(midi_root.rglob("*.midi")))
    pairs_all = []
    miss = 0
    for m in all_midi:
        rel = m.relative_to(midi_root)
        w = wav_root.joinpath(rel).with_suffix(".wav")
        if require_exists and not w.exists():
            miss += 1
            continue
        pairs_all.append((str(w), str(m), int(program_id)))

    if len(pairs_all) == 0:
        raise RuntimeError("No (wav,midi) pairs matched.")

    out = {"train": pairs_all, "validation": pairs_all[:0]}
    print(f"[giantmidi] total_midi={len(all_midi)} | matched={len(pairs_all)} | missing_wav={miss}")
    return out


def load_maps_csv(maps_csv, *, program_id=0):
    df = pd.read_csv(maps_csv)
    train_pairs, val_pairs = [], []
    for _, row in df.iterrows():
        entry = (str(row["audio_path"]), str(row["midi_path"]), int(program_id))
        if row["split"] == "train":
            train_pairs.append(entry)
        elif row["split"] == "validation":
            val_pairs.append(entry)
    print(f"[MAPS] train={len(train_pairs)} | val={len(val_pairs)}")
    return {"train": train_pairs, "validation": val_pairs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps_csv", type=str, default=None,
                     help="MAPS scenario CSV (labeled). Overrides --synth_midi/wav_dir")
    ap.add_argument("--synth_midi_dir", type=str, default="dataset/GiantMIDI-PIano/surname_checked_midis")
    ap.add_argument("--synth_wav_dir", type=str, default="dataset/GiantMIDI-PIano/surname_checked_midis_synth")
    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--input_frames", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_t", type=float, default=2e-4)
    ap.add_argument("--lr_c", type=float, default=1e-4)
    ap.add_argument("--lr_warmup_steps", type=int, default=0)
    ap.add_argument("--lr_min_ratio", type=float, default=0.1)
    ap.add_argument("--lambda_adv", type=float, default=0.01)
    ap.add_argument("--discriminator_mode", type=str, default=None, choices=["adversial", "confusion"])
    ap.add_argument("--chunk_frames", type=int, default=None)
    ap.add_argument("--disc_hidden", type=int, default=256)

    ap.add_argument("--use_pseudo", action="store_true")
    ap.add_argument("--pseudo_start_step", type=int, default=5000)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--unsup_weight", type=float, default=1.0)
    ap.add_argument("--pseudo_max_len", type=int, default=1024)
    ap.add_argument("--pseudo_threshold", type=float, default=-0.5)
    ap.add_argument("--pseudo_topn", type=int, default=0)

    ap.add_argument("--pretrained_ckpt", type=str, default=None)
    ap.add_argument("--real_wav_dir", type=str, default="dataset/maestro-v3.0.0")

    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--valid_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--keep_last_n", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default=None)

    ap.add_argument("--grad_clip", type=float, default=0.0,
                     help="max norm for gradient clipping (0=disabled)")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--cache_root", type=str, default="cache/wave_sr16000")
    ap.add_argument("--cache_dir_synth", type=str, default="")
    ap.add_argument("--cache_dir_real", type=str, default="")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--prefetch_cache_workers", type=int, default=0)
    args = ap.parse_args()

    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.maps_csv is not None:
            scenario = Path(args.maps_csv).stem
            save_dir = os.path.join("checkpoints_MAPS_DA", f"run_{ts}_{scenario}")
        else:
            save_dir = os.path.join("checkpoints_maestro_DA", f"run_{ts}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ---- Load labeled pairs ----
    if args.maps_csv is not None:
        pairs = load_maps_csv(args.maps_csv, program_id=0)
    else:
        pairs = collect_pairs_synth_from_giantmidi(args.synth_midi_dir, args.synth_wav_dir, program_id=0)
        print(f"[synth] train={len(pairs['train'])}")
    print(f"Checkpoints -> {save_dir}")

    vocab = build_vocab(input_frames=args.input_frames, instrument_type="piano", include_note_off=True)

    # ---- Load unlabeled real wavs ----
    real_root = Path(args.real_wav_dir)
    if not real_root.exists():
        raise SystemExit(f"real_wav_dir not found: {real_root}")
    real_wavs = sorted([str(p) for p in real_root.rglob("*.wav")])
    if len(real_wavs) == 0:
        raise SystemExit(f"No wav files found under: {real_root}")
    pairs_real = {"train": real_wavs}

    meta = {
        "script": "train_maestro_ddp_DA_iter.py",
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "data": {
            "labeled_train": len(pairs["train"]),
            "labeled_val": len(pairs["validation"]),
            "real_wavs": len(real_wavs),
        },
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---- cache directory auto-routing ----
    def _dataset_name_from(path_str):
        p = Path(path_str).resolve()
        parts = p.parts
        if "dataset" in parts:
            i = parts.index("dataset")
            if i + 1 < len(parts):
                return parts[i + 1]
        if len(parts) >= 2:
            return parts[-2]
        return p.name

    if not args.cache_dir_synth:
        if args.maps_csv is not None:
            args.cache_dir_synth = str(Path(args.cache_root) / "MAPS")
        else:
            ds_name_synth = _dataset_name_from(args.synth_wav_dir)
            args.cache_dir_synth = str(Path(args.cache_root) / ds_name_synth / "synth")
    if not args.cache_dir_real:
        ds_name_real = _dataset_name_from(args.real_wav_dir)
        args.cache_dir_real = str(Path(args.cache_root) / ds_name_real / "real")

    if not args.no_cache:
        labeled_wavs = [w for (w, _m, _pid) in pairs["train"] + pairs.get("validation", [])
                        if not str(w).endswith(".npy")]
        real_wavs_raw = [w for w in real_wavs if not str(w).endswith(".npy")]

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

        cached_labeled = _prefetch_set(labeled_wavs, args.cache_dir_synth, "labeled")
        cached_real = _prefetch_set(real_wavs_raw, args.cache_dir_real, "real")

        if cached_labeled:
            map_w2c = dict(zip(labeled_wavs, cached_labeled))
            pairs["train"] = [(map_w2c.get(w, w), m, pid) for (w, m, pid) in pairs["train"]]
            pairs["validation"] = [(map_w2c.get(w, w), m, pid) for (w, m, pid) in pairs["validation"]]
        if cached_real:
            pairs_real = {"train": list(cached_real)}

    use_dc = args.discriminator_mode is not None
    if args.discriminator_mode == "adversial":
        model = train_loop_distributed_DA_adversial_iter(
            pairs,
            vocab=vocab,
            pairs_real=pairs_real,
            use_dc=use_dc,
            lr_t=args.lr_t,
            lr_c=args.lr_c,
            lr_warmup_steps=args.lr_warmup_steps,
            lr_min_ratio=args.lr_min_ratio,
            chunk_frames=args.chunk_frames,
            disc_hidden=args.disc_hidden,
            use_pseudo=args.use_pseudo,
            pseudo_start_step=args.pseudo_start_step,
            ema_decay=args.ema_decay,
            unsup_weight=args.unsup_weight,
            pseudo_max_len=args.pseudo_max_len,
            pseudo_threshold=args.pseudo_threshold,
            pseudo_topn=args.pseudo_topn,
            pretrained_ckpt=args.pretrained_ckpt,
            grad_clip=args.grad_clip,
            iters=args.iters,
            bs=args.bs,
            input_frames=args.input_frames,
            save_every=args.save_every,
            ckpt_every=args.ckpt_every,
            valid_every=args.valid_every,
            log_every=args.log_every,
            keep_last_n=args.keep_last_n,
            save_dir=save_dir,
            use_cache=False,
            cache_dir=args.cache_root,
            sr=args.sr,
            num_workers=4,
        )
    else:
        model = train_loop_distributed_DA_confusion_iter(
            pairs,
            vocab=vocab,
            pairs_real=pairs_real,
            lambda_adv=args.lambda_adv,
            lr_t=args.lr_t,
            lr_c=args.lr_c,
            lr_warmup_steps=args.lr_warmup_steps,
            lr_min_ratio=args.lr_min_ratio,
            chunk_frames=args.chunk_frames,
            disc_hidden=args.disc_hidden,
            use_pseudo=args.use_pseudo,
            pseudo_start_step=args.pseudo_start_step,
            ema_decay=args.ema_decay,
            unsup_weight=args.unsup_weight,
            pseudo_max_len=args.pseudo_max_len,
            pseudo_threshold=args.pseudo_threshold,
            pseudo_topn=args.pseudo_topn,
            pretrained_ckpt=args.pretrained_ckpt,
            grad_clip=args.grad_clip,
            iters=args.iters,
            bs=args.bs,
            input_frames=args.input_frames,
            save_every=args.save_every,
            ckpt_every=args.ckpt_every,
            valid_every=args.valid_every,
            log_every=args.log_every,
            keep_last_n=args.keep_last_n,
            save_dir=save_dir,
            use_cache=False,
            cache_dir=args.cache_root,
            sr=args.sr,
            num_workers=4,
        )

    print(f"Training finished -> {save_dir}")

# python -m torch.distributed.run --nproc_per_node=4 run_mt3/train_maestro_ddp_DA_iter.py
