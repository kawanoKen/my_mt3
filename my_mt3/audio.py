# amtx/audio.py  (torchaudio版)
import numpy as np
import torch
import torchaudio

# ===== 基本パラメータ（MVP既定） =====
DEFAULT_SR = 22050

def load_wav_mono(path: str, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """
    WAV読み込み -> mono化 -> 目的サンプルレートへリサンプル。
    返り値は (waveform(float32, shape[T]), sr)
    """
    wav, file_sr = torchaudio.load(path)          # wav: [C, T], float32/float64
    if wav.dim() == 2 and wav.size(0) > 1:        # stereo -> mono (平均)
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:                          # [T] -> [1, T]
        wav = wav.unsqueeze(0)

    if file_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=sr)
        wav = resampler(wav)

    wav = wav.squeeze(0).contiguous()             # [T]
    return wav.numpy().astype(np.float32), sr

def wav_to_logmel(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_fft: int = 2048,
    hop: int = 256,
    n_mels: int = 256,
    power: float = 2.0,
) -> np.ndarray:
    """
    波形(y) -> log-Melスペクトログラム。
    返り値は [T, n_mels] の float32（librosa版と互換の転置）
    """
    # numpy -> torch [1, T]
    wav = torch.from_numpy(y).float().unsqueeze(0)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=power,          # librosaの power=2.0 と同義（パワースペクトログラム）
        center=True,
        pad_mode="reflect",
        f_min=0.0,
        f_max=sr / 2.0,
        norm=None,            # librosaのデフォルトに近づけるなら None を維持
        mel_scale="htk",      # librosaに合わせたいなら "htk"（好みで "slaney" でも可）
    )

    amp2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None)
    spec = mel_spec(wav)                # [1, n_mels, T_spec]
    logmel = amp2db(spec).squeeze(0)    # [n_mels, T_spec]

    # 転置して [T, n_mels] に揃える
    return logmel.transpose(0, 1).contiguous().numpy().astype(np.float32)

def chunk_indices(total_sec, chunk_sec=2.048, include_last=True):
    t, out, eps = 0.0, [], 5e-3
    while t + chunk_sec <= total_sec + eps:
        out.append((t, min(t + chunk_sec, total_sec)))
        t += chunk_sec
    if include_last and not out and total_sec > 0:
        out.append((0.0, min(chunk_sec, total_sec)))
    return out

def ms_quantize(timesec: float, step_ms: int = 10) -> int:
    """秒をms刻みの整数インデックスに量子化"""
    return int(round(timesec * 1000 / step_ms))
