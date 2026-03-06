from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@torch.amp.autocast("cuda", enabled=False)
def bce_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy averaged over all elements."""
    return F.binary_cross_entropy(pred.float(), target.float(), reduction="mean")


@torch.amp.autocast("cuda", enabled=False)
def bce_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked BCE normalized by total number of bins (B*P*T), as in the SSL paper."""
    pred = pred.float()
    target = target.float()
    loss = F.binary_cross_entropy(pred, target, reduction="none")
    loss = loss * mask.to(loss.dtype)
    return loss.sum() / (pred.numel())


def mse_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE normalized by total bins."""
    loss = (pred - target) ** 2
    loss = loss * mask.to(loss.dtype)
    return loss.sum() / pred.numel()


def compute_frame_weights_oaf(
    onset: torch.Tensor,
    frame: torch.Tensor,
    c: float = 5.0,
    decay: str = "inv_exp",
) -> torch.Tensor:
    """Compute the per-bin weights described in the O&F paper for the frame loss.

    onset/frame: (P,T) binary labels for a single example.

    The paper specifies:
      - weight=c on onset frames t1..t2
      - decaying weights after the onset until note end t3
      - weight=1 elsewhere

    Note: The exact decay formula formatting in the PDF can be ambiguous. We provide a reasonable
    interpretation that **emphasizes early frames** (decay='inv_exp' => 1/c, 1/c^2, ... after onset).
    """
    if onset.dim() != 2 or frame.dim() != 2:
        raise ValueError("onset and frame must be (P,T)")

    P, T = onset.shape
    w = torch.ones((P, T), dtype=torch.float32, device=onset.device)

    onset_b = onset > 0.5
    frame_b = frame > 0.5

    for p in range(P):
        t = 0
        while t < T:
            # find the next onset
            if onset_b[p, t]:
                t1 = t
                # t2: last consecutive onset label frame
                t2 = t1
                while t2 + 1 < T and onset_b[p, t2 + 1]:
                    t2 += 1

                # t3: end of the active note segment (based on frame activity)
                t3 = t2
                while t3 + 1 < T and frame_b[p, t3 + 1]:
                    t3 += 1

                # apply weights
                w[p, t1 : t2 + 1] = c
                if t3 > t2:
                    for k, tt in enumerate(range(t2 + 1, t3 + 1), start=1):
                        if decay == "inv_exp":
                            w[p, tt] = c ** (-k)
                        elif decay == "linear":
                            w[p, tt] = max(1.0, c - float(k))
                        else:
                            # fallback: keep weight 1
                            w[p, tt] = 1.0

                t = t3 + 1
            else:
                t += 1

    return w


@torch.amp.autocast("cuda", enabled=False)
def frame_bce_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted BCE for frame activity; normalized by number of bins."""
    pred = pred.float()
    target = target.float()
    loss = F.binary_cross_entropy(pred, target, reduction="none")
    loss = loss * weights.float()
    return loss.sum() / pred.numel()


@dataclass
class SupervisedLossConfig:
    use_weighted_frame_loss: bool = False  # O&F 2018 uses it; SSL paper disables it
    frame_weight_c: float = 5.0
    frame_weight_decay: str = "inv_exp"
    lambda_on: float = 1.0
    lambda_off: float = 1.0
    lambda_fr: float = 1.0
    lambda_vel: float = 0.0


def supervised_loss(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    cfg: SupervisedLossConfig,
    velocity_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute supervised loss.

    outputs keys: onset, frame, offset(optional), velocity(optional)
    labels keys: onset, frame, offset
    """
    loss = 0.0

    loss_on = bce_mean(outputs["onset"], labels["onset"])
    loss += cfg.lambda_on * loss_on

    if cfg.use_weighted_frame_loss:
        # compute weights per example
        B, P, T = labels["frame"].shape
        ws = []
        for b in range(B):
            w = compute_frame_weights_oaf(
                onset=labels["onset"][b],
                frame=labels["frame"][b],
                c=cfg.frame_weight_c,
                decay=cfg.frame_weight_decay,
            )
            ws.append(w)
        w_batch = torch.stack(ws, dim=0)  # (B,P,T)
        loss_fr = frame_bce_weighted(outputs["frame"], labels["frame"], w_batch)
    else:
        loss_fr = bce_mean(outputs["frame"], labels["frame"])
    loss += cfg.lambda_fr * loss_fr

    if "offset" in outputs and "offset" in labels:
        loss_off = bce_mean(outputs["offset"], labels["offset"])
        loss += cfg.lambda_off * loss_off

    if cfg.lambda_vel > 0.0 and "velocity" in outputs and velocity_labels is not None:
        # only enforce velocity where onset label is 1 (paper)
        mask = labels["onset"] > 0.5
        loss_vel = mse_masked(outputs["velocity"], velocity_labels, mask)
        loss += cfg.lambda_vel * loss_vel

    return loss


@dataclass
class UnsupervisedLossConfig:
    lambda_u: float = 0.05        # overall SSL weight (paper)
    lambda_on: float = 1.0
    lambda_off: float = 0.0       # paper default disables offset in unsup loss
    lambda_fr: float = 1.0


def unsupervised_loss(
    outputs_aug: Dict[str, torch.Tensor],
    pseudo: Dict[str, torch.Tensor],
    mask: Dict[str, torch.Tensor],
    cfg: UnsupervisedLossConfig,
) -> torch.Tensor:
    """Compute FixMatch-style unsupervised loss for O&F.

    outputs_aug: predictions for augmented unlabeled input (B,P,T) in [0,1]
    pseudo: pseudo labels (0/1) for clean unlabeled input
    mask: bool mask indicating which bins are confident (not NaN)
    """
    lu = 0.0
    lu += cfg.lambda_on * bce_masked(outputs_aug["onset"], pseudo["onset"], mask["onset"])
    lu += cfg.lambda_fr * bce_masked(outputs_aug["frame"], pseudo["frame"], mask["frame"])
    if cfg.lambda_off > 0.0 and ("offset" in outputs_aug) and ("offset" in pseudo):
        lu += cfg.lambda_off * bce_masked(outputs_aug["offset"], pseudo["offset"], mask["offset"])
    return lu
