from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class PseudoLabelConfig:
    tau_lo: float = 0.05
    tau_up: float = 0.95
    use_distribution_matching: bool = True
    # optional RNG seed for reproducibility
    seed: Optional[int] = None


def threshold_to_pseudo(pred: torch.Tensor, tau_lo: float, tau_up: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert soft predictions to pseudo labels + confidence mask.

    Implements φ(x, τ_lo, τ_up):
      1 if x >= τ_up
      0 if x <= τ_lo
      ignore otherwise

    Returns:
        pseudo: float tensor (B,P,T) in {0,1}
        mask:  bool tensor (B,P,T) True where pseudo is defined (not NaN)
    """
    if pred.min() < 0 or pred.max() > 1:
        # not strictly required, but helps catch accidentally passing logits
        raise ValueError("Expected probabilities in [0,1]")

    mask_hi = pred >= tau_up
    mask_lo = pred <= tau_lo
    mask = mask_hi | mask_lo
    pseudo = mask_hi.to(torch.float32)
    return pseudo, mask


def distribution_match_mask(
    pseudo: torch.Tensor,
    mask: torch.Tensor,
    desired_ratio: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Undersample excess zeros/ones to match desired ones/zeros ratio.

    This implements the distribution matching operator Δ from the SSL paper using undersampling:
      - compute current ratio in the pseudo-label batch
      - randomly turn excess entries into ignore (mask=False) so that
        ones/zeros matches the reference ratio

    Args:
        pseudo: (B,P,T) {0,1}
        mask: (B,P,T) bool (True: keep)
        desired_ratio: reference ones/zeros ratio (>0, usually small)
    Returns:
        new_mask (B,P,T) bool
    """
    if desired_ratio <= 0:
        return mask

    gen = None
    if seed is not None:
        gen = torch.Generator(device=mask.device)
        gen.manual_seed(seed)

    new_mask = mask.clone()

    ones_mask = new_mask & (pseudo > 0.5)
    zeros_mask = new_mask & (pseudo <= 0.5)

    ones = int(ones_mask.sum().item())
    zeros = int(zeros_mask.sum().item())
    if ones == 0 or zeros == 0:
        return new_mask

    cur_ratio = ones / zeros

    if cur_ratio < desired_ratio:
        # too many zeros -> drop zeros
        zeros_to_keep = int(ones / desired_ratio)
        zeros_to_keep = max(0, min(zeros_to_keep, zeros))
        drop = zeros - zeros_to_keep
        if drop > 0:
            idx = zeros_mask.nonzero(as_tuple=False)  # (N,3)
            perm = torch.randperm(idx.shape[0], generator=gen, device=idx.device)
            drop_idx = idx[perm[:drop]]
            new_mask[drop_idx[:, 0], drop_idx[:, 1], drop_idx[:, 2]] = False
    elif cur_ratio > desired_ratio:
        # too many ones -> drop ones
        ones_to_keep = int(desired_ratio * zeros)
        ones_to_keep = max(0, min(ones_to_keep, ones))
        drop = ones - ones_to_keep
        if drop > 0:
            idx = ones_mask.nonzero(as_tuple=False)
            perm = torch.randperm(idx.shape[0], generator=gen, device=idx.device)
            drop_idx = idx[perm[:drop]]
            new_mask[drop_idx[:, 0], drop_idx[:, 1], drop_idx[:, 2]] = False

    return new_mask


def make_pseudo_labels(
    outputs_clean: Dict[str, torch.Tensor],
    cfg: PseudoLabelConfig,
    desired_ratios: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Create pseudo labels (+ masks) for multiple heads."""
    pseudo: Dict[str, torch.Tensor] = {}
    mask: Dict[str, torch.Tensor] = {}

    for key, pred in outputs_clean.items():
        p, m = threshold_to_pseudo(pred, cfg.tau_lo, cfg.tau_up)
        if cfg.use_distribution_matching and desired_ratios is not None and key in desired_ratios:
            m = distribution_match_mask(p, m, desired_ratio=float(desired_ratios[key]), seed=cfg.seed)
        pseudo[key] = p
        mask[key] = m

    return pseudo, mask
