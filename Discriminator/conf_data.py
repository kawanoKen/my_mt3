from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import random

import torch
from torch.utils.data import Dataset

from midi_tokenizer import MidiTokCfg, AugCfg, load_piano_notes, apply_augmentation, midi_notes_to_tokens


def pad_and_add_cls(
    seq: torch.Tensor,
    *,
    max_len: int,
    pad_id: int,
    cls_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seq: (L,) without CLS
    returns tokens: (max_len,) and attn_mask: (max_len,) bool
    """
    L = min(seq.numel(), max_len - 1)
    out = torch.full((max_len,), pad_id, dtype=torch.long)
    out[0] = cls_id
    out[1:1+L] = seq[:L].long()
    attn = torch.zeros((max_len,), dtype=torch.bool)
    attn[:1+L] = True
    return out, attn


def sample_window(seq: torch.Tensor, window_len_wo_cls: int) -> torch.Tensor:
    N = seq.numel()
    if N <= window_len_wo_cls:
        return seq.clone()
    s = random.randint(0, N - window_len_wo_cls)
    return seq[s:s+window_len_wo_cls].clone()


def corrupt_tokens_simple(
    seq: torch.Tensor,
    *,
    vocab_size: int,
    pad_id: int = 0,
    cls_id: int = 1,
    p_delete: float = 0.10,
    p_insert: float = 0.05,
    p_replace: float = 0.10,
    span_shuffle_prob: float = 0.15,
    max_span: int = 12,
) -> torch.Tensor:
    """
    Generic corruption (structure-agnostic). Good enough as a first version.
    """
    x = seq.clone().tolist()

    # delete
    if p_delete > 0 and len(x) > 0:
        x = [t for t in x if random.random() > p_delete] or x

    # replace
    if p_replace > 0 and len(x) > 0:
        for i in range(len(x)):
            if random.random() < p_replace:
                r = random.randrange(0, vocab_size)
                while r in (pad_id, cls_id):
                    r = random.randrange(0, vocab_size)
                x[i] = r

    # insert
    if p_insert > 0:
        out = []
        for t in x:
            out.append(t)
            if random.random() < p_insert:
                r = random.randrange(0, vocab_size)
                while r in (pad_id, cls_id):
                    r = random.randrange(0, vocab_size)
                out.append(r)
        x = out

    # span shuffle
    if span_shuffle_prob > 0 and len(x) >= 4 and random.random() < span_shuffle_prob:
        span = random.randint(4, min(max_span, len(x)))
        s = random.randint(0, len(x) - span)
        chunk = x[s:s+span]
        random.shuffle(chunk)
        x[s:s+span] = chunk

    return torch.tensor(x, dtype=torch.long)


def list_midi_files(root: str | Path) -> List[Path]:
    """
    指定ディレクトリ配下（再帰）からMIDIファイルを列挙して返す。
    - 拡張子は .mid / .midi を大文字・小文字問わずサポート
    - MAESTRO v3.0.0 のような階層構造にも対応
    """
    root = Path(root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    exts = {".mid", ".midi"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)

    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No MIDI files found under: {root}")
    return files


class MidiTokenWindowBinaryDataset(Dataset):
    """
    Reads piano MIDI paths, tokenizes on-the-fly with augmentation,
    then yields (tokens_with_cls, attn_mask, label).
    Positive: clean-ish token window
    Negative: corrupted token window (from the same window)
    """
    def __init__(
        self,
        midi_files: List[Path],
        *,
        tok_cfg: MidiTokCfg,
        aug_cfg: AugCfg,
        max_len: int,
        windows_per_file: int = 8,
        corruption_kwargs: dict | None = None,
    ):
        self.files = midi_files
        self.tok_cfg = tok_cfg
        self.aug_cfg = aug_cfg
        self.max_len = max_len
        self.windows_per_file = windows_per_file
        self.corruption_kwargs = corruption_kwargs or {}

        self.index: List[int] = []
        for i in range(len(self.files)):
            for _ in range(windows_per_file):
                self.index.append(i)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fi = self.index[idx]
        midi_path = str(self.files[fi])

        notes = load_piano_notes(midi_path)
        notes = apply_augmentation(notes, aug=self.aug_cfg)
        seq = midi_notes_to_tokens(notes, self.tok_cfg)  # (N,)

        # sample window
        window_len_wo_cls = self.max_len - 1
        clean = sample_window(seq, window_len_wo_cls)

        # pos
        pos_tokens, pos_attn = pad_and_add_cls(
            clean, max_len=self.max_len, pad_id=self.tok_cfg.pad_id, cls_id=self.tok_cfg.cls_id
        )
        pos_y = torch.tensor(1.0, dtype=torch.float32)

        # neg
        bad = corrupt_tokens_simple(
            clean,
            vocab_size=self.tok_cfg.vocab_size(),
            pad_id=self.tok_cfg.pad_id,
            cls_id=self.tok_cfg.cls_id,
            **self.corruption_kwargs,
        )
        neg_tokens, neg_attn = pad_and_add_cls(
            bad, max_len=self.max_len, pad_id=self.tok_cfg.pad_id, cls_id=self.tok_cfg.cls_id
        )
        neg_y = torch.tensor(0.0, dtype=torch.float32)

        # balanced sampling
        if random.random() < 0.5:
            return pos_tokens, pos_attn, pos_y
        else:
            return neg_tokens, neg_attn, neg_y
