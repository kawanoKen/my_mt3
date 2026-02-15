# my_mt3/audio_io.py
from __future__ import annotations

from pathlib import Path
import os
import hashlib
import numpy as np
import torchaudio
import torch
from dataclasses import dataclass

DEFAULT_SR = 16000


def load_wav_mono_resample(path: str, sr: int = DEFAULT_SR) -> np.ndarray:
    """wav -> mono -> resample(sr) -> np.float32 [T]"""
    wav, file_sr = torchaudio.load(path)  # [C, T]
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)

    if file_sr != sr:
        wav = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=sr)(wav)

    return wav.squeeze(0).contiguous().cpu().numpy().astype(np.float32)


def load_audio_mono(path: str, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """
    - .npy: 波形キャッシュ（[T] float32）を memmapで読む
    - それ以外: wav を mono+resample
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        y = np.load(str(p), mmap_mode="r")
        return np.asarray(y, dtype=np.float32), sr
    return load_wav_mono_resample(str(p), sr=sr), sr


def _cache_key(wav_path: str) -> str:
    # 同名衝突回避：パスをhash化して短いキーに
    s = str(wav_path).replace("\\", "/")
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
    stem = Path(s).stem
    return f"{stem}.{h}"


def ensure_wave_cache(
    wav_path: str,
    *,
    cache_dir: str,
    sr: int = DEFAULT_SR,
) -> str:
    cache_dir_p = Path(cache_dir)
    # 親ディレクトリを必ず作っておく（DDP/並列時の競合を最小化）
    cache_dir_p.mkdir(parents=True, exist_ok=True)

    key = _cache_key(wav_path)
    out_path = cache_dir_p / f"{key}.sr{sr}.npy"

    if out_path.exists():
        return str(out_path)

    y = load_wav_mono_resample(wav_path, sr=sr)

    # ★重要: tmp も ".npy" で終わらせる（np.save が勝手に拡張子を付けないように）
    tmp_path = str(out_path) + ".tmp.npy"

    # tmp を同一ディレクトリに作成してから原子的に置換
    # 念のため親ディレクトリを再作成（並列時の競合対策）
    Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp_path, y)
    try:
        os.replace(tmp_path, out_path)  # 原子的に置換
    except FileNotFoundError:
        # まれに並列で親が消えた/まだ無い/保存先が消えた等。再作成して再試行
        cache_dir_p.mkdir(parents=True, exist_ok=True)
        # tmp が消えていたら再保存
        if not os.path.exists(tmp_path):
            np.save(tmp_path, y)
        os.replace(tmp_path, out_path)
    except FileExistsError:
        # 他プロセスが先に作成済みならOK
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return str(out_path)

@dataclass
class LogMelCfg:
    sr: int = 16000
    n_fft: int = 2048
    hop: int = 256
    n_mels: int = 256
    power: float = 2.0
    f_min: float = 0.0
    f_max: float | None = None
    mel_scale: str = "htk"

class LogMelExtractor:
    """
    center=False（左端アライン）log-mel
    """
    def __init__(self, cfg: LogMelCfg):
        self.cfg = cfg
        f_max = cfg.f_max if cfg.f_max is not None else cfg.sr / 2.0

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop,
            n_mels=cfg.n_mels,
            power=cfg.power,
            center=False,        # ★重要
            pad_mode="reflect",
            f_min=cfg.f_min,
            f_max=f_max,
            norm=None,
            mel_scale=cfg.mel_scale,
        )
        self.amp2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        wav = torch.from_numpy(y.copy()).float().unsqueeze(0)  # [1, T]
        spec = self.mel(wav)                            # [1, n_mels, T']
        logmel = self.amp2db(spec).squeeze(0)           # [n_mels, T']
        return logmel.transpose(0, 1).contiguous().numpy().astype(np.float32)  # [T', n_mels]

