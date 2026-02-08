from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable
import csv

import numpy as np
import torch
from torch.utils.data import Dataset
import pretty_midi
import re



# ----------------------------
# GM drum note -> 9 classes
# ----------------------------
GM_TO_9: Dict[int, int] = {
    35: 0, 36: 0,
    38: 1, 40: 1, 37: 1, 39: 1,
    42: 2, 44: 2, 46: 3,
    41: 4, 43: 4,
    45: 5, 47: 5,
    48: 6, 50: 6,
    51: 7, 53: 7, 59: 7,
    49: 8, 52: 8, 55: 8, 57: 8,
}


@dataclass
class GrooveRollCfg:
    root: str = "dataset/groove"
    split: str = "train"  # train/validation/test (info.csvにあればそれを使う)
    seed: int = 0

    T: int = 1024
    hop_sec: float = 0.01
    K: int = 9

    include_eval_session: bool = True
    chunks_per_file: int = 4
    loop_short: bool = True
    file_ext: Tuple[str, ...] = (".mid", ".midi")

    # ★追加：beat_type フィルタ（None/空なら全て）
    beat_types: Optional[Tuple[str, ...]] = None

    # ★追加：info.csv を使うか
    use_info_csv: bool = True
    info_csv_name: str = "info.csv"


# ----------------------------
# helpers: info.csv reader & filtering
# ----------------------------
def _read_info_csv(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _normalize(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _pick_first_existing_key(row: dict, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            return str(row[k]).strip()
    return None

def _match_any_regex(value: str, patterns: Optional[Tuple[str, ...]]) -> bool:
    """
    value が patterns（正規表現）のいずれかにマッチすれば True。
    patterns が None なら常に True（フィルタしない）。
    """
    if patterns is None:
        return True
    v = (value or "").strip()
    for p in patterns:
        if re.search(p, v, flags=re.IGNORECASE):
            return True
    return False


def _resolve_midi_path(root: Path, midi_rel: str, file_ext: Tuple[str, ...]) -> Optional[Path]:
    """
    info.csv の midi_rel から実ファイルを見つける。
    1) root/midi_rel が存在すればそれ
    2) 存在しなければファイル名として root以下をrglobで探索
    """
    p = (root / midi_rel).resolve()
    if p.exists() and p.suffix.lower() in file_ext:
        return p

    # たとえば midi_rel が "xxx.mid" だけ / または拡張子無し の場合もある
    name = Path(midi_rel).name
    cand = list(root.rglob(name))
    for c in cand:
        if c.is_file() and c.suffix.lower() in file_ext:
            return c.resolve()

    # 拡張子無しの可能性
    if Path(name).suffix == "":
        for ext in file_ext:
            cand2 = list(root.rglob(name + ext))
            for c in cand2:
                if c.is_file():
                    return c.resolve()

    return None


def _collect_midi_paths_from_info(root: Path, cfg: GrooveRollCfg) -> List[Path]:
    """
    info.csv から split / beat_type(regex) で対象MIDIを集める。
    """
    info_path = root / cfg.info_csv_name
    if not info_path.exists():
        raise FileNotFoundError(f"info.csv not found: {info_path}")

    rows = _read_info_csv(info_path)
    want_split = _normalize(cfg.split)

    out: List[Path] = []
    for r in rows:
        # split filter（info.csvにsplit列があれば使う）
        split_val = _normalize(r.get("split"))
        if split_val and want_split and split_val != want_split:
            continue

        # beat_type filter（regex）
        beat_raw = (r.get("style") or "").strip()
        if not _match_any_regex(beat_raw, cfg.beat_types):
            continue

        # midi path key candidates
        midi_rel = _pick_first_existing_key(
            r,
            keys=("midi_filename", "midi_path", "midi", "midi_file", "midi_file_path"),
        )
        if midi_rel is None:
            continue

        # eval_session 除外（パス文字列中に含まれる場合）
        if (not cfg.include_eval_session) and ("eval_session" in midi_rel):
            continue

        p = _resolve_midi_path(root, midi_rel, cfg.file_ext)
        if p is not None:
            out.append(p)

    # 重複除去
    uniq: List[Path] = []
    seen = set()
    for p in out:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            uniq.append(p)

    return uniq



def _list_midi_files_fallback(root: Path, cfg: GrooveRollCfg) -> List[Path]:
    """
    info.csv が無い/使わないときの従来走査（splitは簡易分割）。
    """
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in cfg.file_ext:
            if not cfg.include_eval_session and "eval_session" in p.parts:
                continue
            files.append(p)
    files.sort()

    # 簡易split（80/10/10）
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n = len(files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    if cfg.split == "train":
        sel = idx[:n_train]
    elif cfg.split in ("val", "validation"):
        sel = idx[n_train:n_train + n_val]
    elif cfg.split == "test":
        sel = idx[n_train + n_val:]
    else:
        raise ValueError(f"Unknown split: {cfg.split}")

    return [files[i] for i in sel]


def _load_drum_notes(pm: pretty_midi.PrettyMIDI):
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            notes.extend(inst.notes)
    if len(notes) == 0:
        for inst in pm.instruments:
            notes.extend(inst.notes)
    return notes


def _midi_to_roll_chunks(
    midi_path: Path,
    *,
    T: int,
    hop_sec: float,
    K: int,
    chunks_per_file: int,
    loop_short: bool,
) -> List[torch.Tensor]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = _load_drum_notes(pm)
    if len(notes) == 0:
        return [torch.zeros(T, K, dtype=torch.float32)]

    end_time = max(n.end for n in notes)
    total_frames = max(1, int(np.ceil(end_time / hop_sec)))

    max_start = max(0, total_frames - T)
    if max_start == 0:
        starts = [0] * chunks_per_file
    else:
        starts = np.linspace(0, max_start, num=chunks_per_file, dtype=int).tolist()

    out: List[torch.Tensor] = []
    for s in starts:
        roll = torch.zeros(T, K, dtype=torch.float32)
        e = s + T

        for n in notes:
            k = GM_TO_9.get(n.pitch, None)
            if k is None or k >= K:
                continue
            t0 = int(round(n.start / hop_sec))

            if loop_short and total_frames < T:
                roll[t0 % T, k] = 1.0
            else:
                if s <= t0 < e:
                    roll[t0 - s, k] = 1.0

        out.append(roll)
    return out


class GrooveMIDIRollDataset(Dataset):
    """
    info.csv の beat_type で絞り込んだ MIDI から clean onset-roll (T,K) を返す。
    """
    def __init__(self, cfg: GrooveRollCfg):
        self.cfg = cfg
        root = Path(cfg.root)

        # 1) info.csv があればそれを使ってフィルタ
        if cfg.use_info_csv and (root / cfg.info_csv_name).exists():
            self.files = _collect_midi_paths_from_info(root, cfg)
            if len(self.files) == 0:
                raise RuntimeError(
                    f"No MIDI matched filters. "
                    f"Check beat_types={cfg.beat_types}, split={cfg.split}, and info.csv columns."
                )
        else:
            # 2) fallback：rglob + 簡易split（beat_typeは効かない）
            self.files = _list_midi_files_fallback(root, cfg)

        # (file_idx, chunk_idx) でインデックス展開
        self.index: List[Tuple[int, int]] = []
        for fi in range(len(self.files)):
            for ci in range(cfg.chunks_per_file):
                self.index.append((fi, ci))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fi, ci = self.index[idx]
        midi_path = self.files[fi]

        chunks = _midi_to_roll_chunks(
            midi_path,
            T=self.cfg.T,
            hop_sec=self.cfg.hop_sec,
            K=self.cfg.K,
            chunks_per_file=self.cfg.chunks_per_file,
            loop_short=self.cfg.loop_short,
        )
        if ci >= len(chunks):
            return torch.zeros(self.cfg.T, self.cfg.K, dtype=torch.float32)
        return chunks[ci]
