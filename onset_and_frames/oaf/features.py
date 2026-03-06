from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import soundfile as sf

from .config import FeatureConfig


def _cache_key(wav_path: str) -> str:
    s = str(wav_path).replace("\\", "/")
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
    stem = Path(s).stem
    return f"{stem}.{h}"


def load_audio_mono(path: Union[str, Path], target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """Load an audio file (mono) and return (waveform, sr).

    Supports .npy wave caches (memmap) as well as regular audio files.
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        y = np.load(str(p), mmap_mode="r")
        # mmap_mode="r" yields a read-only array; copy to make it writable for PyTorch.
        y = np.asarray(y, dtype=np.float32).copy()
        return torch.from_numpy(y), target_sr

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav)

    if sr != target_sr:
        wav_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
    return wav_t, sr


def wave_cache_path(wav_path: str, *, cache_dir: str, sr: int = 16000) -> Path:
    """Return the expected .npy cache path for a wav file (without creating it)."""
    return Path(cache_dir) / f"{_cache_key(wav_path)}.sr{sr}.npy"


def ensure_wave_cache(
    wav_path: str,
    *,
    cache_dir: str,
    sr: int = 16000,
) -> str:
    """Resample wav to target sr and save as .npy cache. DDP-safe (atomic write).

    Compatible with my_mt3.audio.ensure_wave_cache — shares the same
    cache directory layout so both systems can reuse cached files.
    """
    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(parents=True, exist_ok=True)

    key = _cache_key(wav_path)
    out_path = cache_dir_p / f"{key}.sr{sr}.npy"

    if out_path.exists():
        return str(out_path)

    wav_t, _ = load_audio_mono(wav_path, target_sr=sr)
    y = wav_t.numpy().astype(np.float32)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(cache_dir_p),
        prefix=out_path.stem + ".",
        suffix=".tmp.npy",
    )
    os.close(fd)
    np.save(tmp_path, y)
    try:
        os.replace(tmp_path, out_path)
    except (FileNotFoundError, FileExistsError):
        if out_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        else:
            raise

    return str(out_path)


class LogMelSpec(torch.nn.Module):
    """Compute log-mel spectrogram (as in Onsets and Frames).

    Paper setting: sr=16k, n_fft=2048, hop=512, n_mels=229, log amplitude.
    """

    def __init__(self, cfg: FeatureConfig):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Args:
            wav: (T,) or (B,T)
        Returns:
            log_mel: (B, n_mels, frames) float32
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel(wav)  # (B, n_mels, frames)
        log_mel = torch.log(mel + self.cfg.log_eps)
        return log_mel

    def extra_repr(self) -> str:
        return str(asdict(self.cfg))
