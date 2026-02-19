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
from my_mt3.tokenizer import encode_events, INPUT_FRAMES, Vocab
import random

DEFAULT_SR = 16000
# -------------------------
# Dataset (chunk enumeration)
# -------------------------

def chunk_indices(total_sec: float, chunk_sec: float = 2.048, include_last: bool = True):
    t, out, eps = 0.0, [], 5e-3
    while t + chunk_sec <= total_sec + eps:
        out.append((t, min(t + chunk_sec, total_sec)))
        t += chunk_sec

    # 端数が残るなら最後に追加
    if include_last and t < total_sec - eps:
        out.append((t, total_sec))

    # total_sec < chunk_sec の短い音声でも1チャンク返す
    if include_last and not out and total_sec > 0:
        out.append((0.0, min(chunk_sec, total_sec)))
    return out



def ms_quantize(timesec: float, step_ms: int = 10) -> int:
    """秒をms刻みの整数インデックスに量子化"""
    return int(round(timesec * 1000 / step_ms))



class AMTDataset(Dataset):
    """
    pairs: [(wav_or_cache_path, midi_path, program_id), ...]

    1 item (1曲) -> chunks = [(mel[input_frames,n_mels], token_ids, (s_sec,e_sec)), ...]
    train: ランダムに max_chunks_per_song 個の窓をサンプル
    val/test: 全曲を決定論的に走査
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, int]],
        *,
        mode: str = "train",
        sr: int = 16000,
        hop: int = 256,
        step_ms: int = 10,
        input_frames: int = INPUT_FRAMES,
        max_chunks_per_song: int | None = 8,
        stride_frames: int | None = None,   # val/test のスライド間隔（Noneなら input_frames）
        include_last: bool = True,
        n_fft: int = 2048,
        n_mels: int = 256,
        vocab: Vocab,
    ):
        self.vocab = vocab
        self.pairs = pairs
        self.mode = mode
        self.sr = sr
        self.hop = hop
        self.step_ms = step_ms
        self.input_frames = int(input_frames)
        self.max_chunks_per_song = max_chunks_per_song
        self.include_last = include_last

        # val/test の stride: デフォは window と同じ（non-overlap）
        self.stride_frames = int(stride_frames) if stride_frames is not None else int(input_frames)

        self.feat = LogMelExtractor(LogMelCfg(sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels))

        # center=False の STFT で input_frames の mel を得るのに必要な波形サンプル数
        self.need_samples = (self.input_frames - 1) * self.hop + n_fft

        # このウィンドウの秒数（token側 frame_max を決めるために使う）
        self.window_sec = self.need_samples / float(self.sr)

        # 10ms刻みトークンの最大 index（0..frame_max_token）
        # 例: window_sec=8.2s, step_ms=10ms -> 820 -> frame_max=819
        frame_max_template = int(round(self.window_sec * 1000.0 / self.step_ms))
        self.frame_max_token = max(0, frame_max_template - 1)

        # MIDIパース簡易キャッシュ（worker内）
        self._midi_cache: dict[str, list[tuple[float, float, int]]] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_notes(self, midi_path: str):
        if midi_path in self._midi_cache:
            return self._midi_cache[midi_path]
        pm = pretty_midi.PrettyMIDI(midi_path)
        notes = [(n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes]
        self._midi_cache[midi_path] = notes
        return notes

    def _make_start_samples(self, total_samples: int):
        """
        window（need_samples）を切り出す開始サンプルssを列挙。
        """
        max_start = max(0, total_samples - self.need_samples)

        if self.mode == "train":
            # ランダムにK個（曲が短い場合もss=0でOK）
            if self.max_chunks_per_song is None:
                # None のときは全列挙に近くなるので注意（非推奨）
                return list(range(0, max_start + 1, self.need_samples))

            K = int(self.max_chunks_per_song)
            if max_start == 0:
                starts = [0] * K
            else:
                starts = [random.randint(0, max_start) for _ in range(K)]
            starts = sorted(starts)  # 任意：時系列順
            return starts

        # val/test: 決定論的に全曲をスライド
        stride_samples = self.stride_frames * self.hop
        starts = list(range(0, max_start + 1, stride_samples))

        # include_last=Trueなら末尾を必ずカバー（端が余る場合に最後を追加）
        if self.include_last and len(starts) > 0:
            last = starts[-1]
            if last != max_start and (max_start - last) > 0:
                starts.append(max_start)
        elif self.include_last and len(starts) == 0:
            starts = [0]

        return starts

    def __getitem__(self, i: int):
        wav_path, midi_path, pid = self.pairs[i]

        # ---- audio ----
        y, _ = load_audio_mono(wav_path, sr=self.sr)
        total_samples = int(len(y))

        # ---- midi notes ----
        notes = self._load_notes(midi_path)

        # ---- window starts ----
        start_samples = self._make_start_samples(total_samples)

        chunks = []
        for ss in start_samples:
            ee = ss + self.need_samples
            y_seg = y[ss:ee]

            # 末尾不足はpadして固定長に
            if len(y_seg) < self.need_samples:
                y_seg = np.pad(y_seg, (0, self.need_samples - len(y_seg)), mode="constant")

            # mel: 必ず [input_frames, n_mels]（center=False + need_samples固定）
            mel = self.feat(y_seg)

            # 念のため長さ保証（理論上一致する）
            if mel.shape[0] != self.input_frames:
                # 万一ズレたら切る/パディング（保険）
                if mel.shape[0] > self.input_frames:
                    mel = mel[: self.input_frames]
                else:
                    padT = self.input_frames - mel.shape[0]
                    mel = np.pad(mel, ((0, padT), (0, 0)), mode="constant")

            s_sec = ss / float(self.sr)
            e_sec = (ss + self.need_samples) / float(self.sr)

            # ---- MIDI -> events ----
            ev = []
            ties = []
            frame_max = self.frame_max_token

            for on, off, p in notes:
                if off <= s_sec or on >= e_sec:
                    continue

                on_q  = max(0, min(ms_quantize(on  - s_sec, self.step_ms), frame_max))
                off_q = max(0, min(ms_quantize(off - s_sec, self.step_ms), frame_max))

                if on < s_sec:
                    tie_off = ms_quantize(min(off, e_sec) - s_sec, self.step_ms)
                    tie_off = max(0, min(tie_off, frame_max))
                    ties.append((p, tie_off))
                    on_q = 0

                ev.append((on_q, off_q, p))

            token_ids = encode_events(ev, pid, ties, frame_max_token=self.frame_max_token, vocab=self.vocab)
            chunks.append((mel, token_ids, (s_sec, e_sec)))

        return chunks