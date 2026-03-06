from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import FeatureConfig, LabelConfig
from .midi_utils import NoteEvent


@dataclass
class LabelTensors:
    onset: torch.Tensor   # (P,T) float32 {0,1}
    frame: torch.Tensor   # (P,T) float32 {0,1}
    offset: torch.Tensor  # (P,T) float32 {0,1}
    velocity: Optional[torch.Tensor] = None  # (P,T) float32 in [0,1] on onset bins only


def _time_to_frame(t: float, dt: float) -> int:
    return int(np.floor(t / dt))


def _time_to_frame_ceil(t: float, dt: float) -> int:
    return int(np.ceil(t / dt))


def notes_to_labels(
    notes: List[NoteEvent],
    n_frames: int,
    feat_cfg: FeatureConfig,
    lab_cfg: LabelConfig,
    compute_velocity: bool = False,
) -> LabelTensors:
    """Convert continuous-time notes to framewise labels.

    - frame: 1 if note is active at any point within the frame
    - onset: note is truncated to onset_length_sec before quantization (paper)
    - offset: we mark a short window before note end (common extended O&F variant)
    """
    dt = feat_cfg.hop_length / feat_cfg.sample_rate  # 32ms by default
    P = lab_cfg.midi_max - lab_cfg.midi_min + 1

    onset = torch.zeros((P, n_frames), dtype=torch.float32)
    frame = torch.zeros((P, n_frames), dtype=torch.float32)
    offset = torch.zeros((P, n_frames), dtype=torch.float32)
    velocity = torch.zeros((P, n_frames), dtype=torch.float32) if compute_velocity else None

    max_vel = max([n.velocity for n in notes], default=127)
    max_vel = max(max_vel, 1)

    for n in notes:
        if n.pitch < lab_cfg.midi_min or n.pitch > lab_cfg.midi_max:
            continue
        p = n.pitch - lab_cfg.midi_min
        start = float(n.start)
        end = float(max(n.end, n.start + 1e-6))

        # frame activity
        t0 = _time_to_frame(start, dt)
        t1 = _time_to_frame_ceil(end, dt)
        t0 = max(t0, 0)
        t1 = min(t1, n_frames)
        if t1 > t0:
            frame[p, t0:t1] = 1.0

        # onset: truncate to onset_length_sec before quantization
        onset_end = start + min(end - start, lab_cfg.onset_length_sec)
        o0 = _time_to_frame(start, dt)
        o1 = _time_to_frame_ceil(onset_end, dt)
        o0 = max(o0, 0)
        o1 = min(o1, n_frames)
        if o1 > o0:
            onset[p, o0:o1] = 1.0
            if compute_velocity and velocity is not None:
                v = float(n.velocity) / float(max_vel)
                velocity[p, o0:o1] = v

        # offset: short window ending at note end
        off_start = max(start, end - lab_cfg.offset_length_sec)
        f0 = _time_to_frame(off_start, dt)
        f1 = _time_to_frame_ceil(end, dt)
        f0 = max(f0, 0)
        f1 = min(f1, n_frames)
        if f1 > f0:
            offset[p, f0:f1] = 1.0

    return LabelTensors(onset=onset, frame=frame, offset=offset, velocity=velocity)


@dataclass
class LabelMarginals:
    """Marginal distribution stats for distribution matching (ones/zeros ratio)."""
    ones: float
    zeros: float

    @property
    def ratio(self) -> float:
        return float(self.ones) / max(float(self.zeros), 1.0)


def estimate_marginals_from_labels(label_list: List[torch.Tensor]) -> LabelMarginals:
    """Estimate marginal ones/zeros across many label matrices (P,T)."""
    ones = 0.0
    zeros = 0.0
    for lab in label_list:
        ones += float((lab > 0.5).sum().item())
        zeros += float((lab <= 0.5).sum().item())
    return LabelMarginals(ones=ones, zeros=zeros)
