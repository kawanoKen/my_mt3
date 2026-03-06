from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AugmentConfig:
    freq_mask_max_width: int = 30
    noise_std: float = 0.01


def augment_spectrogram(log_mel: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    """Apply the augmentation pipeline used in the SSL paper.

    Steps:
      1) Frequency masking: set a random contiguous frequency band (<=30 bins) to the mean value
      2) Additive Gaussian noise (std=0.01)

    Args:
        log_mel: (B, F, T)
    """
    if log_mel.dim() != 3:
        raise ValueError(f"Expected (B,F,T), got {tuple(log_mel.shape)}")

    x = log_mel.clone()
    B, F, T = x.shape

    for b in range(B):
        width = int(torch.randint(low=0, high=cfg.freq_mask_max_width + 1, size=(1,)).item())
        if width > 0 and width < F:
            f0 = int(torch.randint(low=0, high=F - width + 1, size=(1,)).item())
            mean_val = float(x[b].mean().item())
            x[b, f0 : f0 + width, :] = mean_val

    if cfg.noise_std > 0:
        x = x + torch.randn_like(x) * cfg.noise_std

    return x
