from __future__ import annotations
from torch.utils.data import Dataset
from my_mt3.audio import load_audio_mono, LogMelExtractor, LogMelCfg
from my_mt3.tokenizer import INPUT_FRAMES
import random
import numpy as np
import torch

class AMTRealDataset(Dataset):
    def __init__(self, wav_paths, *, sr=16000, hop=256, input_frames=INPUT_FRAMES,
                 n_fft=2048, n_mels=256, consecutive_pair: bool = False):
        self.wav_paths = wav_paths
        self.sr = sr
        self.hop = hop
        self.input_frames = int(input_frames)
        self.feat = LogMelExtractor(LogMelCfg(sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels))
        self.need_samples = (self.input_frames - 1) * self.hop + n_fft
        self.consecutive_pair = bool(consecutive_pair)

    def __len__(self): return len(self.wav_paths)

    def __getitem__(self, i):
        wav_path = self.wav_paths[i]
        y, _ = load_audio_mono(wav_path, sr=self.sr)
        total_samples = int(len(y))
        if self.consecutive_pair:
            max_start = max(0, total_samples - (2 * self.need_samples))
        else:
            max_start = max(0, total_samples - self.need_samples)
        ss = 0 if max_start == 0 else random.randint(0, max_start)
        y_seg = y[ss:ss + self.need_samples]
        if len(y_seg) < self.need_samples:
            y_seg = np.pad(y_seg, (0, self.need_samples - len(y_seg)), mode="constant")
        mel = self.feat(y_seg)  # [input_frames, n_mels]
        if mel.shape[0] != self.input_frames:
            mel = mel[: self.input_frames] if mel.shape[0] > self.input_frames else np.pad(mel, ((0, self.input_frames - mel.shape[0]), (0, 0)))
        mel_t = torch.tensor(mel, dtype=torch.float32)
        if not self.consecutive_pair:
            return mel_t, i, ss

        ss_next = ss + self.need_samples
        y_seg_next = y[ss_next:ss_next + self.need_samples]
        if len(y_seg_next) < self.need_samples:
            y_seg_next = np.pad(y_seg_next, (0, self.need_samples - len(y_seg_next)), mode="constant")
        mel_next = self.feat(y_seg_next)
        if mel_next.shape[0] != self.input_frames:
            mel_next = (
                mel_next[: self.input_frames]
                if mel_next.shape[0] > self.input_frames
                else np.pad(mel_next, ((0, self.input_frames - mel_next.shape[0]), (0, 0)))
            )
        mel_next_t = torch.tensor(mel_next, dtype=torch.float32)
        return mel_t, mel_next_t, i, ss, ss_next
