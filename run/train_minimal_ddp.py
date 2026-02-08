# run/train_minimal.py

# ==== add this at the very top ====
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# ==================================

import glob
import os
import argparse
import re
import torch
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
from my_mt3.train import train_loop_distributed
import warnings
from datetime import datetime

# torchaudioに関連する特定のUserWarningを無視する
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

def collect_pairs_groove(
    root: str | Path = "dataset/groove",
    splits: tuple[str, ...] = ("train", "validation","test"),
    program_id: Optional[int] = 0,   # None にすると pairs から pid を外す
    require_exists: bool = True,     # 存在しないパスを弾く
    beat_types: Optional[tuple[str, ...]] = None,  # 例: (r"^pop", r"^rock", r"^funk")
) -> Dict[str, List[tuple]]:
    """
    GrooveMIDI の info.csv を読んで splitごとに pairs を作る。

    returns:
      program_id is not None:
        {"train": [(audio_path, midi_path, pid), ...], "test": [...], "valid": [...]}
      program_id is None:
        {"train": [(audio_path, midi_path), ...], "test": [...], "valid": [...]}
    """
    root = Path(root)
    csv_path = root / "info.csv"
    df = pd.read_csv(csv_path)

    out: Dict[str, List[tuple]] = {sp: [] for sp in splits}

    # 必要カラムチェック（違う名前ならここを直す）
    assert "split" in df.columns
    assert "audio_filename" in df.columns
    assert "midi_filename" in df.columns

    for sp in splits:
        subset = df[df["split"] == sp]
        # style列があり、beat_typesが与えられていればフィルタ
        if beat_types is not None and "style" in df.columns:
            pats = tuple(beat_types)
            def _ok_style(s: str) -> bool:
                s = str(s or "")
                return any(re.search(p, s, flags=re.IGNORECASE) for p in pats)
            subset = subset[subset["style"].apply(_ok_style)]

        # 1行ずつ pairs 化
        for audio_rel, midi_rel in zip(subset["audio_filename"], subset["midi_filename"]):
            audio_path = root / str(audio_rel)
            midi_path  = root / str(midi_rel)

            if require_exists and (not audio_path.exists() or not midi_path.exists()):
                # 欠損がある場合はスキップ（必要ならログ出力）
                continue

            if program_id is None:
                out[sp].append((str(audio_path), str(midi_path)))
            else:
                out[sp].append((str(audio_path), str(midi_path), int(program_id)))

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--styles",
        type=str,
        default="pop,rock,funk",
        help="style フィルタ（カンマ区切り/正規表現可; all で全件）",
    )
    args = ap.parse_args()
    raw = (args.styles or "").strip()
    if raw.lower() in ("all", "*") or raw == "":
        styles = None  # フィルタ無効化（全件）
    else:
        styles = tuple(f"^{s.strip()}" for s in raw.split(",") if s.strip())

    pairs = collect_pairs_groove(beat_types=styles)
    print(f"train pairs: {len(pairs['train'])} validation pairs: {len(pairs['validation'])} test pairs: {len(pairs['test'])}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("checkpoints", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Checkpoints will be saved to: {save_dir}")

    model = train_loop_distributed(
        pairs,
        epochs=2000,          # まず10周（必要に応じて20〜30）
        save_every=100,
        save_dir=save_dir,
        bs=16,              # VRAMに応じて 8〜32
        lr=2e-4,
        # train_loop 側でキャッシュ作成する想定（train_minimalはシンプル）
        use_cache=True,
        cache_dir="cache/wave_sr16000",
        sr=16000,
    )
    print("saved -> saved checkpoints to {save_dir}")


# python -m torch.distributed.run --nproc_per_node=2 run/train_minimal_ddp.py