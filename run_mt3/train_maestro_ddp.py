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
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent import futures

import pandas as pd
import warnings
from tqdm import tqdm

from my_mt3.train import train_loop_distributed
from my_mt3.tokenizer import build_vocab, INPUT_FRAMES
from my_mt3.audio import ensure_wave_cache

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


def collect_pairs_maestro(
    root: str | Path,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    *,
    program_id: Optional[int] = 0,   # None なら (audio, midi) の2要素にする
    require_exists: bool = True,
) -> Dict[str, List[Tuple[str, ...]]]:
    """
    MAESTRO v3.0.0 の CSV (maestro-v3.0.0.csv) を読み、
    split ごとに (audio_path, midi_path, program_id) のタプルを収集する。
    """
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # 期待カラムの存在確認
    for col in ("split", "audio_filename", "midi_filename"):
        if col not in df.columns:
            raise ValueError(f"CSV column '{col}' not found in {csv_path}")

    out: Dict[str, List[Tuple[str, ...]]] = {sp: [] for sp in splits}
    for sp in splits:
        subset = df[df["split"] == sp]
        for audio_rel, midi_rel in zip(subset["audio_filename"], subset["midi_filename"]):
            audio_path = root / str(audio_rel)
            midi_path  = root / str(midi_rel)
            if require_exists and (not audio_path.exists() or not midi_path.exists()):
                continue
            if program_id is None:
                out[sp].append((str(audio_path), str(midi_path)))
            else:
                out[sp].append((str(audio_path), str(midi_path), int(program_id)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="dataset/maestro-v3.0.0",
        help="MAESTRO v3.0.0 データセットのルート（CSVと年別ディレクトリがある場所）",
    )
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--save-dir", type=str, default=None, help="未指定なら checkpoints_maestro/run_YYYYmmdd_HHMMSS")
    ap.add_argument("--no-cache", action="store_true", help="波形キャッシュを無効化")
    ap.add_argument("--cache-root", type=str, default="cache/wave_sr16000",
                    help="キャッシュ共通ルート。配下にデータセット名で自動振り分け")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--prefetch_cache_workers", type=int, default=0,
                    help="学習前にWAVキャッシュを並列生成（0で無効）")
    args = ap.parse_args()

    # 出力ディレクトリ
    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("checkpoints_maestro", f"run_{ts}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ペア収集（train / validation の2分割で十分。必要なら test も）
    pairs = collect_pairs_maestro(args.root, splits=("train", "validation"), program_id=0)
    print(f"train pairs: {len(pairs['train'])} | validation pairs: {len(pairs['validation'])}")
    print(f"Checkpoints will be saved to: {save_dir}")

    # ===== Save meta.json =====
    meta = {
        "script": "train_maestro_ddp.py",
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "data": {
            "train": len(pairs["train"]),
            "validation": len(pairs["validation"]),
        },
    }
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Meta saved -> {meta_path}")

    # ===== Cache (dataset-specific directory) =====
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

    # ===== Vocab =====
    vocab = build_vocab(input_frames=INPUT_FRAMES, instrument_type="piano", include_note_off=True)

    model = train_loop_distributed(
        pairs,
        epochs=args.epochs,
        bs=args.bs,
        lr=args.lr,
        save_every=args.save_every,
        save_dir=save_dir,
        use_cache=False,
        cache_dir=args.cache_root,
        sr=args.sr,
        vocab=vocab,
        num_workers=4,
    )

    # Plot losses
    try:
        import csv
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        csv_path = Path(save_dir) / "losses.csv"
        if csv_path.exists():
            xs, train_l, val_l = [], [], []
            with open(csv_path, "r") as f:
                for row in csv.DictReader(f):
                    xs.append(int(row["epoch"]))
                    train_l.append(float(row["train_loss"]))
                    val_l.append(float(row["val_loss"]))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(xs, train_l, label="train_loss")
            if any(v > 0 for v in val_l):
                ax.plot(xs, val_l, label="val_loss", linestyle="--")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_title("Training losses")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = Path(save_dir) / "losses.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"Loss plot -> {fig_path}")
    except Exception as e:
        print(f"(warn) failed to plot: {e}")

    print(f"Training finished -> {save_dir}")

