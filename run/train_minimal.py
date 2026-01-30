# run/train_minimal.py

# ==== add this at the very top ====
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# ==================================

import glob
import os
import torch
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
from my_mt3.train import train_loop

def collect_pairs_groove(
    root: str | Path = "dataset/groove",
    splits: tuple[str, ...] = ("train", "validation","test"),
    program_id: Optional[int] = 0,   # None にすると pairs から pid を外す
    require_exists: bool = True,     # 存在しないパスを弾く
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
    pairs = collect_pairs_groove()
    print(f"train pairs: {len(pairs['train'])} validation pairs: {len(pairs['validation'])} test pairs: {len(pairs['test'])}")

    model = train_loop(
        pairs,
        epochs=30,          # まず10周（必要に応じて20〜30）
        bs=4,              # VRAMに応じて 8〜32
        lr=2e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # train_loop 側でキャッシュ作成する想定（train_minimalはシンプル）
        use_cache=True,
        cache_dir="cache/wave_sr16000",
        sr=16000,
    )

    torch.save(model.state_dict(), "ckpt_piano.pt")
    print("saved -> ckpt_piano.pt")
