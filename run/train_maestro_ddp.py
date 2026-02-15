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

from my_mt3.train import train_loop_distributed

# torchaudio / numpy 由来の冗長な UserWarning を抑制（任意）
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
        default="/work/kawano/kawano/my_mt3/maestro-v3.0.0",
        help="MAESTRO v3.0.0 データセットのルート（CSVと年別ディレクトリがある場所）",
    )
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--save-dir", type=str, default=None, help="未指定なら checkpoints_maestro/run_YYYYmmdd_HHMMSS")
    ap.add_argument("--no-cache", action="store_true", help="波形キャッシュを無効化")
    ap.add_argument("--cache-dir", type=str, default="cache/wave_sr16000")
    ap.add_argument("--sr", type=int, default=16000)
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
    print(f"📁 Checkpoints will be saved to: {save_dir}")

    # DDP 版の最小トレーニングループを実行
    model = train_loop_distributed(
        pairs,
        epochs=args.epochs,
        bs=args.bs,
        lr=args.lr,
        save_every=args.save_every,
        save_dir=save_dir,
        use_cache=(not args.no_cache),
        cache_dir=args.cache_dir,
        sr=args.sr,
    )
    print(f"✅ Training finished. Saved to: {save_dir}")

