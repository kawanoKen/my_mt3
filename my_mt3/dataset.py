# my_mt3/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import torch
import torchaudio
import pretty_midi
from torch.utils.data import Dataset
from my_mt3.audio import load_audio_mono, LogMelExtractor, LogMelCfg
from my_mt3.tokenizer import encode_events
import random

DEFAULT_SR = 16000
# -------------------------
# Dataset (chunk enumeration)
# -------------------------

def chunk_indices(total_sec: float, chunk_sec: float = 2.048, include_last: bool = True):
    """
    0, chunk_sec, 2*chunk_sec... の固定境界で区切るチャンク列を返す。
    out: [(s_sec, e_sec), ...]
    """
    t, out, eps = 0.0, [], 5e-3
    while t + chunk_sec <= total_sec + eps:
        out.append((t, min(t + chunk_sec, total_sec)))
        t += chunk_sec
    # total_sec < chunk_sec の短い音声でも1チャンク返したい場合
    if include_last and not out and total_sec > 0:
        out.append((0.0, min(chunk_sec, total_sec)))
    return out


def ms_quantize(timesec: float, step_ms: int = 10) -> int:
    """秒をms刻みの整数インデックスに量子化"""
    return int(round(timesec * 1000 / step_ms))

class AMTDataset(Dataset):
    """
    pairs: [(wav_or_cache_path, midi_path, program_id), ...]

    __getitem__ は 1曲について複数チャンクを列挙して返す:
      chunks = [(mel, token_ids, (s_sec, e_sec)), ...]
    """
    def __init__(
        self,
        pairs: List[Tuple[str, str, int]],
        *,
        mode: str = "train",
        sr: int = DEFAULT_SR,
        hop: int = 256,
        step_ms: int = 10,
        chunk_sec: float = 2.048,
        max_chunks_per_song=8,
        include_last: bool = True,
        n_fft: int = 2048,
        n_mels: int = 256,
    ):
        self.pairs = pairs
        self.sr = sr
        self.hop = hop
        self.step_ms = step_ms
        self.chunk_sec = chunk_sec
        self.include_last = include_last
        self.mode = mode
        self.max_chunks_per_song = max_chunks_per_song

        self.feat = LogMelExtractor(LogMelCfg(sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels))

        # MIDIパースを高速化したい場合の簡易キャッシュ（プロセス内のみ）
        self._midi_cache: dict[str, list[tuple[float, float, int]]] = {}

        # token側の最大フレーム（Timeトークンの定義に合わせて 0..204 などに固定したいならここ）
        frame_max_template = int(round(chunk_sec * 1000 / step_ms))  # 2.048s & 10ms => ~205
        self.frame_max_token = max(0, frame_max_template - 1)         # 204

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_notes(self, midi_path: str):
        if midi_path in self._midi_cache:
            return self._midi_cache[midi_path]
        pm = pretty_midi.PrettyMIDI(midi_path)
        notes = [(n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes]
        self._midi_cache[midi_path] = notes
        return notes

    def __getitem__(self, i: int):
        wav_path, midi_path, pid = self.pairs[i]

        # ---- load wave (wav or cache) ----
        y, _ = load_audio_mono(wav_path, sr=self.sr)
        total_sec = float(len(y)) / float(self.sr)

        # ---- load MIDI notes ----
        notes = self._load_notes(midi_path)

        chunks = []
        if self.mode == "train":
            chunk_list = chunk_indices(total_sec, self.chunk_sec, include_last=True)
            if self.max_chunks_per_song is not None:
                if len(chunk_list) > self.max_chunks_per_song:
                    chunk_list = random.sample(chunk_list, self.max_chunks_per_song)
                    # 時系列順が良ければソート（任意）
                    chunk_list = sorted(chunk_list, key=lambda x: x[0])
        else:
            # 完全決定論的：0秒から最後まで
            chunk_list = chunk_indices(
                total_sec,
                self.chunk_sec,
                include_last=True
            )

        for s_sec, e_sec in chunk_list:
            ss = int(round(s_sec * self.sr))
            ee = int(round(e_sec * self.sr))
            if ee <= ss:
                continue
            y_seg = y[ss:ee]

            # center=False なので、最低 n_fft サンプルないとフレームが0になる
            # 短い末尾チャンクを捨てたくないなら pad する（ここでは pad して1フレーム以上確保）
            n_fft = self.feat.cfg.n_fft
            if len(y_seg) < n_fft:
                pad = n_fft - len(y_seg)
                y_seg = np.pad(y_seg, (0, pad), mode="constant")

            mel = self.feat(y_seg)  # [T_chunk_mel, n_mels]

            # ---- MIDI -> events (0..204に収める) ----
            ev = []
            ties = []
            frame_max = self.frame_max_token  # 204

            for on, off, p in notes:
                if off <= s_sec or on >= e_sec:
                    continue

                # チャンク基準に変換して10ms刻みへ
                on_q  = max(0, min(ms_quantize(on  - s_sec, self.step_ms), frame_max))
                off_q = max(0, min(ms_quantize(off - s_sec, self.step_ms), frame_max))

                # 前チャンクから継続しているノート（tie）
                if on < s_sec:
                    tie_off = ms_quantize(min(off, e_sec) - s_sec, self.step_ms)
                    tie_off = max(0, min(tie_off, frame_max))
                    ties.append((p, tie_off))
                    on_q = 0

                ev.append((on_q, off_q, p))

            # ここはあなたの既存実装を使う前提
            token_ids = encode_events(ev, pid, ties)

            chunks.append((mel, token_ids, (s_sec, e_sec)))

        return chunks
