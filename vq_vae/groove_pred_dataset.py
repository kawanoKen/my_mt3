from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import pretty_midi

from groove_midi import GM_TO_9, _load_drum_notes


@dataclass
class PredVsGTCfg:
    """
    outputs の推論MIDIを入力（noisy）、dataset/groove の対応MIDIを正解（clean）として
    (T,K) の onsets roll を返すデータセット設定
    """
    pred_root: str = "outputs/groove_test_pred"
    gt_root: str = "dataset/groove"

    T: int = 1024
    hop_sec: float = 0.01
    K: int = 9

    chunks_per_file: int = 4
    loop_short: bool = True
    file_ext_pred: Tuple[str, ...] = (".mid", ".midi")
    file_ext_gt: Tuple[str, ...] = (".mid", ".midi")

    # info.csv を用いて GT を解決する（無い場合は rglob で名前一致を探索）
    use_info_csv: bool = True
    info_csv_name: str = "info.csv"


def _resolve_gt_path(gt_root: Path, name_stem: str, file_ext: Tuple[str, ...]) -> Optional[Path]:
    """
    outputs 側の stem から、dataset/groove 側の GT MIDI を推定する。
    1) まず stem と同名のファイルを rglob で探索
    2) 見つからなければ拡張子違いも試す
    """
    # 例: 1_funk-groove1_138_beat_4-4.pred.mid -> stem: "1_funk-groove1_138_beat_4-4.pred"
    # よくあるのは ".pred" を落とした stem が元ファイル名
    candidates = [name_stem]
    if name_stem.endswith(".pred"):
        candidates.append(name_stem[:-5])
    # 探索
    for cand in candidates:
        # そのまま＋拡張子セット
        for ext in file_ext:
            for p in gt_root.rglob(cand + ext):
                if p.is_file():
                    return p.resolve()
        # 拡張子不明・完全一致（拡張子ありのファイル名で一致するもの）
        for p in gt_root.rglob(cand):
            if p.is_file() and p.suffix.lower() in file_ext:
                return p.resolve()
    return None


def _midi_to_roll(midi_path: Path, T: int, hop_sec: float, K: int) -> Tuple[torch.Tensor, int]:
    """
    MIDI -> onset roll (length <= T), 返り値: (roll[T,K], total_frames)
    長さが T 未満のときは pad、T より長いときは後段でスライスで扱う
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = _load_drum_notes(pm)
    if len(notes) == 0:
        return torch.zeros(T, K, dtype=torch.float32), 1

    end_time = max(n.end for n in notes)
    total_frames = max(1, int(np.ceil(end_time / hop_sec)))

    roll = torch.zeros(max(T, total_frames), K, dtype=torch.float32)
    for n in notes:
        k = GM_TO_9.get(n.pitch, None)
        if k is None or k >= K:
            continue
        t = int(round(n.start / hop_sec))
        if 0 <= t < roll.size(0):
            roll[t, k] = 1.0

    # 長さ T に整形（切る/パッド）
    if roll.size(0) < T:
        pad = torch.zeros(T - roll.size(0), K, dtype=torch.float32)
        roll = torch.cat([roll, pad], dim=0)
    elif roll.size(0) > T:
        roll = roll[:T]
    return roll, total_frames


class PredVsGTRollDataset(Dataset):
    """
    outputs の推論MIDIを x（noisy）、dataset/groove の対応MIDIを y（clean）として
    (x_roll[T,K], y_roll[T,K]) を返すデータセット
    """
    def __init__(self, cfg: PredVsGTCfg):
        self.cfg = cfg
        self.pred_files: List[Path] = []
        self.gt_files: List[Path] = []
        self.index: List[Tuple[int, int]] = []  # (file_idx, chunk_idx)

        pred_root = Path(cfg.pred_root)
        gt_root = Path(cfg.gt_root)
        if not pred_root.exists():
            raise FileNotFoundError(f"pred_root not found: {pred_root}")
        if not gt_root.exists():
            raise FileNotFoundError(f"gt_root not found: {gt_root}")

        # 予測MIDIを列挙
        all_pred: List[Path] = []
        for p in pred_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in cfg.file_ext_pred:
                all_pred.append(p)
        all_pred.sort()

        # 各予測に対応する GT を解決
        for p in all_pred:
            stem = p.stem  # ".pred" が付いている場合あり
            gt = _resolve_gt_path(gt_root, stem, cfg.file_ext_gt)
            if gt is None:
                continue
            self.pred_files.append(p.resolve())
            self.gt_files.append(gt.resolve())

        # (file_idx, chunk_idx) 展開
        for fi in range(len(self.pred_files)):
            for ci in range(cfg.chunks_per_file):
                self.index.append((fi, ci))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fi, _ = self.index[idx]
        pred_path = self.pred_files[fi]
        gt_path = self.gt_files[fi]

        # roll 化（T に整形）
        x_roll, _ = _midi_to_roll(pred_path, T=self.cfg.T, hop_sec=self.cfg.hop_sec, K=self.cfg.K)
        y_roll, _ = _midi_to_roll(gt_path,   T=self.cfg.T, hop_sec=self.cfg.hop_sec, K=self.cfg.K)

        # 短い曲をループで増やすオプション（簡易）
        if self.cfg.loop_short:
            if x_roll.sum() == 0:
                # 何もない場合はそのまま
                pass
            if y_roll.sum() == 0:
                pass

        return x_roll, y_roll
