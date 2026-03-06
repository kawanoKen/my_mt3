# run/train_maestro_ddp.py
#
# MAESTRO v3.0.0 の CSV を読み、(audio_path, midi_path, program_id) のペアを
# split ごとに作って DDP 学習を回す最小スクリプト。
#
# 例:
#   python -m torch.distributed.run --nproc_per_node=2 run/train_maestro_ddp.py \
#     --root "/work/kawano/kawano/my_mt3/maestro-v3.0.0" \
#     --epochs 10 --bs 8 --lr 2e-4
#

# ==== add this at the very top ====
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# ==================================

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import warnings
from tqdm import tqdm

from my_mt3.train_DA_adversial import train_loop_distributed_DA_adversial
from my_mt3.train_DA_confusion import train_loop_distributed_DA_confusion
from my_mt3.tokenizer import build_vocab, INPUT_FRAMES
from my_mt3.audio import ensure_wave_cache
import concurrent.futures as futures
import csv
import json

# torchaudio / numpy 由来の冗長な UserWarning を抑制（任意）
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


def collect_pairs_synth_from_giantmidi(
    midi_root: str | Path,
    wav_root: str | Path,
    *,
    program_id: int = 0,
    require_exists: bool = True,
) -> Dict[str, List[Tuple[str, str, int]]]:
    """
    GiantMIDI-Piano の構造に対応:
      - 再帰ディレクトリ or フラット（root直下に全ファイル）
      - MIDI: <midi_root>/(optional_subdirs)/<name>.mid
      - WAV : <wav_root>/(optional_subdirs)/<name>.wav   （同じ相対構造）
    を対応付け、train のみ返す（validationは不要のため最小限に抑制）。
    """
    midi_root = Path(midi_root)
    wav_root = Path(wav_root)
    if not midi_root.exists():
        raise FileNotFoundError(f"midi_root not found: {midi_root}")
    if not wav_root.exists():
        raise FileNotFoundError(f"wav_root not found: {wav_root}")

    all_midi = sorted(list(midi_root.rglob("*.mid")) + list(midi_root.rglob("*.midi")))
    pairs_all: List[Tuple[str, str, int]] = []
    miss = 0
    for m in all_midi:
        rel = m.relative_to(midi_root)
        w = wav_root.joinpath(rel).with_suffix(".wav")
        if require_exists and not w.exists():
            miss += 1
            continue
        pairs_all.append((str(w), str(m), int(program_id)))

    if len(pairs_all) == 0:
        raise RuntimeError("No (wav,midi) pairs matched. Check paths and that you rendered WAVs.")

    # validation は不要だが、下流のローダ構築の都合で最小限だけ用意（空でも可だが安全に1バッチ未満に）
    out = {
        "train": pairs_all,
        "validation": pairs_all[:0],  # 空にして評価コストをゼロ化
    }
    print(f"[giantmidi] total_midi={len(all_midi)} | matched_pairs={len(pairs_all)} | missing_wav={miss} | train={len(out['train'])}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_midi_dir", type=str, default="dataset/GiantMIDI-PIano/surname_checked_midis", help="GiantMIDI-Piano MIDI root (recursive or flat)")
    ap.add_argument("--synth_wav_dir", type=str, default="dataset/GiantMIDI-PIano/surname_checked_midis_synth", help="Rendered WAV root (recursive/flat, same relative structure)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="base lr (kept for logging)")
    ap.add_argument("--lr_t", type=float, default=2e-4, help="Transformer(E,D) lr for DA")
    ap.add_argument("--lr_c", type=float, default=1e-4, help="Discriminator lr for DA")
    ap.add_argument("--lambda_adv", type=float, default=0.01)
    ap.add_argument("--discriminator_mode", type=str, default=None, choices=["adversial", "confusion"])
    ap.add_argument("--chunk_frames", type=int, default=None, help="frames per chunk for discriminator (auto ~0.1s if None)")
    ap.add_argument("--disc_hidden", type=int, default=256)
    # Pseudo-label (SSL)
    ap.add_argument("--use_pseudo", action="store_true", help="enable pseudo-label training on real data")
    ap.add_argument("--pseudo_start_epoch", type=int, default=70)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--unsup_weight", type=float, default=1.0)
    ap.add_argument("--pseudo_max_len", type=int, default=1024)
    ap.add_argument("--pseudo_threshold", type=float, default=-0.5,
                     help="chunk-level mean log-prob threshold for pseudo-label (lower = accept more, used when pseudo_topn=0)")
    ap.add_argument("--pseudo_topn", type=int, default=0,
                     help="select top-N most confident chunks per batch as pseudo-labels (0=use threshold mode)")
    ap.add_argument("--pretrained_ckpt", type=str, default=None,
                     help="path to a pretrained MT3 checkpoint (.pt) to initialise model weights")
    ap.add_argument("--real_wav_dir", type=str, default="dataset/maestro-v3.0.0", help="real (unlabeled) WAV directory for DA (recursive, e.g., MAESTRO root)")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--save-dir", type=str, default=None, help="未指定なら checkpoints_maestro/run_YYYYmmdd_HHMMSS")
    ap.add_argument("--no-cache", action="store_true", help="波形キャッシュを無効化")
    # 互換: --cache-dir は未使用（下位互換のため残す）
    ap.add_argument("--cache-dir", type=str, default="cache/wave_sr16000")
    # 推奨: ルートを共通にし、配下にデータセット名で切る
    ap.add_argument("--cache-root", type=str, default="cache/wave_sr16000", help="キャッシュ共通ルート。配下にデータセット名で自動振り分け")
    # 明示指定したい場合のみ使う（指定があればこちらを優先）
    ap.add_argument("--cache-dir-synth", type=str, default="", help="キャッシュ（合成データ用）明示指定。未指定なら --cache-root/＜データセット名＞")
    ap.add_argument("--cache-dir-real", type=str, default="", help="キャッシュ（実データ用）明示指定。未指定なら --cache-root/＜データセット名＞")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--prefetch_cache_workers", type=int, default=0, help="学習前に現状WAVのキャッシュを並列生成（0で無効）")
    args = ap.parse_args()

    # 出力ディレクトリ
    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("checkpoints_maestro_DA", f"run_{ts}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # GiantMIDI-Piano の (wav,midi) supervised synth ペアを作成し、train/val に分割
    pairs = collect_pairs_synth_from_giantmidi(
        args.synth_midi_dir,
        args.synth_wav_dir,
        program_id=0,
        require_exists=True,
    )
    print(f"[synth] train={len(pairs['train'])} | val(skipped)={len(pairs['validation'])}")
    print(f"📁 Checkpoints will be saved to: {save_dir}")

    # ===== Save meta.json =====
    meta = {
        "script": "train_maestro_ddp_DA.py",
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "data": {
            "synth_train": len(pairs["train"]),
            "synth_val": len(pairs["validation"]),
        },
    }
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Meta saved -> {meta_path}")

    # ===== Vocab をここで固定生成 =====
    # MAESTRO（ピアノ）前提: piano / Note Off あり / tie あり（既定）
    vocab = build_vocab(input_frames=INPUT_FRAMES, instrument_type="piano", include_note_off=True)

    # Real (unlabeled) WAVs for Domain Adaptation
    real_root = Path(args.real_wav_dir)
    if not real_root.exists():
        raise SystemExit(f"real_wav_dir not found: {real_root}")
    # MAESTRO root を渡された場合、年別ディレクトリ配下の .wav をすべて拾う
    real_wavs = sorted([str(p) for p in real_root.rglob("*.wav")])
    if len(real_wavs) == 0:
        raise SystemExit(f"No wav files found under: {real_root}")
    pairs_real = {"train": real_wavs}

    # real_wavs 数をメタに追記
    meta["data"]["real_wavs"] = len(real_wavs)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---- cache ディレクトリをデータセット名で自動振り分け ----
    def _dataset_name_from(path_str: str) -> str:
        p = Path(path_str).resolve()
        parts = p.parts
        # "dataset" 直下の名前を優先
        if "dataset" in parts:
            i = parts.index("dataset")
            if i + 1 < len(parts):
                return parts[i + 1]
        # なければ親ディレクトリ名を使う（フラット想定）
        if len(parts) >= 2:
            return parts[-2]
        return p.name

    if not args.cache_dir_synth:
        ds_name_synth = _dataset_name_from(args.synth_wav_dir)
        # データセット名/synth のようにサフィックスを付けて確実に分離
        args.cache_dir_synth = str(Path(args.cache_root) / ds_name_synth / "synth")
    if not args.cache_dir_real:
        ds_name_real = _dataset_name_from(args.real_wav_dir)
        # データセット名/real のようにサフィックスを付けて確実に分離
        args.cache_dir_real = str(Path(args.cache_root) / ds_name_real / "real")

    # 事前キャッシュ生成（現状WAVがすべてcache化されるように）＆ パスの差し替え
    if not args.no_cache:
        # synth
        synth_train_wavs = [w for (w, _m, _pid) in pairs["train"] if not str(w).endswith(".npy")]
        synth_val_wavs = [w for (w, _m, _pid) in pairs.get("validation", []) if not str(w).endswith(".npy")]
        # real
        real_wavs_raw = [w for w in real_wavs if not str(w).endswith(".npy")]

        def _prefetch_set(wavs, cache_dir, title):
            if not wavs:
                return []
            if args.prefetch_cache_workers > 0:
                print(f"[cache:{title}] prefetch start: {len(wavs)} files -> {cache_dir} (sr={args.sr}, workers={args.prefetch_cache_workers})")
                def _cache_one(w):
                    try:
                        return ensure_wave_cache(w, cache_dir=cache_dir, sr=args.sr)
                    except Exception as e:
                        return f"ERR:{w}:{e}"
                with futures.ThreadPoolExecutor(max_workers=int(args.prefetch_cache_workers)) as ex:
                    cached = list(tqdm(ex.map(_cache_one, wavs), total=len(wavs), desc=f"prefetch {title}", unit="wav"))
                print(f"[cache:{title}] prefetch done.")
            else:
                cached = [ensure_wave_cache(w, cache_dir=cache_dir, sr=args.sr) for w in wavs]
            # フィルタ: エラーは除外
            return [c for c in cached if isinstance(c, str) and not c.startswith("ERR:")]

        cached_synth_train = _prefetch_set(synth_train_wavs, args.cache_dir_synth, "synth-train")
        cached_synth_val = _prefetch_set(synth_val_wavs, args.cache_dir_synth, "synth-val") if synth_val_wavs else []
        cached_real = _prefetch_set(real_wavs_raw, args.cache_dir_real, "real")

        # パス差し替え（.npy で置換。存在しなければ元のまま残す）
        if cached_synth_train:
            map_w2cached = dict(zip(synth_train_wavs, cached_synth_train))
            new_train = []
            for (w, m, pid) in pairs["train"]:
                new_train.append((map_w2cached.get(w, w), m, pid))
            pairs["train"] = new_train
        if cached_synth_val and pairs.get("validation"):
            map_w2cached = dict(zip(synth_val_wavs, cached_synth_val))
            new_val = []
            for (w, m, pid) in pairs["validation"]:
                new_val.append((map_w2cached.get(w, w), m, pid))
            pairs["validation"] = new_val
        if cached_real:
            # real_wavs を差し替え
            # 未キャッシュが混ざっても許容（ensure_wave_cacheで対応したはず）
            pairs_real = {"train": [map_path for map_path in cached_real]}

    # DDP 版のDomain Adaptation学習ループ（DA）を実行
    use_dc = args.discriminator_mode is not None
    if args.discriminator_mode == "adversial":
        model = train_loop_distributed_DA_adversial(
            pairs,
            vocab=vocab,
            pairs_real=pairs_real,
            use_dc=use_dc,
            lr_t=args.lr_t,
            lr_c=args.lr_c,
            chunk_frames=args.chunk_frames,
            disc_hidden=args.disc_hidden,
            use_pseudo=args.use_pseudo,
            pseudo_start_epoch=args.pseudo_start_epoch,
            ema_decay=args.ema_decay,
            unsup_weight=args.unsup_weight,
            pseudo_max_len=args.pseudo_max_len,
            pseudo_threshold=args.pseudo_threshold,
            pseudo_topn=args.pseudo_topn,
            pretrained_ckpt=args.pretrained_ckpt,
            epochs=args.epochs,
            bs=args.bs,
            save_every=args.save_every,
            save_dir=save_dir,
            use_cache=False,
            cache_dir=args.cache_root,
            sr=args.sr,
            num_workers=4
        )
    else:
        model = train_loop_distributed_DA_confusion(
            pairs,
            vocab=vocab,
            pairs_real=pairs_real,
            lambda_adv=args.lambda_adv,
            lr_t=args.lr_t,
            lr_c=args.lr_c,
            chunk_frames=args.chunk_frames,
            disc_hidden=args.disc_hidden,
            use_pseudo=args.use_pseudo,
            pseudo_start_epoch=args.pseudo_start_epoch,
            ema_decay=args.ema_decay,
            unsup_weight=args.unsup_weight,
            pseudo_max_len=args.pseudo_max_len,
            pseudo_threshold=args.pseudo_threshold,
            pseudo_topn=args.pseudo_topn,
            pretrained_ckpt=args.pretrained_ckpt,
            epochs=args.epochs,
            bs=args.bs,
            save_every=args.save_every,
            save_dir=save_dir,
            # 既にパス差し替え済みのため内部キャッシュは無効化
            use_cache=False,
            cache_dir=args.cache_root,
            sr=args.sr,
            num_workers=4
        )
    # Plot DA losses saved by the train loop (rank0 process)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs, total, sup, adv, unsup, disc, val = [], [], [], [], [], [], []
        csv_path = Path(save_dir) / "da_losses.csv"
        if csv_path.exists():
            with open(csv_path, "r") as f:
                r = csv.DictReader(f)
                for row in r:
                    xs.append(int(row["epoch"]))
                    total.append(float(row["train_total"]))
                    sup.append(float(row["train_sup"]))
                    adv.append(float(row["train_adv"]))
                    unsup.append(float(row["train_unsup"]))
                    disc.append(float(row["train_disc"]))
                    val.append(float(row["val_loss"]))
            plt.figure(figsize=(8,5))
            plt.plot(xs, total, label="train_total")
            plt.plot(xs, sup, label="train_sup")
            plt.plot(xs, adv, label="train_adv")
            plt.plot(xs, unsup, label="train_unsup")
            plt.plot(xs, disc, label="train_disc")
            if any(val):
                plt.plot(xs, val, label="val_loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("DA training losses")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = Path(save_dir) / "da_losses.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"🖼️ Saved loss plot -> {fig_path}")
        else:
            print(f"(info) loss CSV not found: {csv_path}")
    except Exception as e:
        print(f"(warn) failed to plot losses: {e}")
    print(f"✅ Training finished. Saved to: {save_dir}")

# python -m torch.distributed.run --nproc_per_node=4 run/train_maestro_ddp_DA.py