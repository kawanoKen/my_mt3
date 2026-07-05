from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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


def _sec_to_frame_round(t_sec: float, feat_cfg: FeatureConfig) -> int:
    """Official O&F-style quantization: round(seconds * sr / hop)."""
    return int(round(float(t_sec) * float(feat_cfg.sample_rate) / float(feat_cfg.hop_length)))


def _clip_interval(start: int, end: int, n_frames: int) -> tuple[int, int]:
    start = max(0, min(int(start), int(n_frames)))
    end = max(0, min(int(end), int(n_frames)))
    return start, end


def notes_to_labels(
    notes: List[NoteEvent],
    n_frames: int,
    feat_cfg: FeatureConfig,
    lab_cfg: LabelConfig,
    compute_velocity: bool = False,
) -> LabelTensors:
    """Convert note events to O&F-style framewise labels.

    This version follows the common official Onsets and Frames convention more
    closely than the previous floor/ceil window implementation:

      - onset/offset times are quantized by round(t * sample_rate / hop_length)
      - onset label width is one hop by default
      - offset label width is one hop by default
      - frame activity spans [onset_frame, offset_frame)

    With the default 16 kHz / 512-hop setting, one hop is 32 ms. Therefore
    onset_length_sec=0.032 and offset_length_sec=0.032 map to exactly one
    positive frame, rather than usually becoming two positive frames.
    """
    P = int(lab_cfg.midi_max - lab_cfg.midi_min + 1)
    onset = torch.zeros((P, n_frames), dtype=torch.float32)
    frame = torch.zeros((P, n_frames), dtype=torch.float32)
    offset = torch.zeros((P, n_frames), dtype=torch.float32)
    velocity = torch.zeros((P, n_frames), dtype=torch.float32) if compute_velocity else None

    # Official constants correspond to ONSET_LENGTH // HOP_LENGTH = 1 and
    # OFFSET_LENGTH // HOP_LENGTH = 1 under the default config. Keep this
    # configurable, but use integer hop counts rather than floor/ceil seconds.
    dt = float(feat_cfg.hop_length) / float(feat_cfg.sample_rate)
    onset_hops = max(1, int(round(float(lab_cfg.onset_length_sec) / dt)))
    offset_hops = max(1, int(round(float(lab_cfg.offset_length_sec) / dt)))

    for n in notes:
        if n.pitch < lab_cfg.midi_min or n.pitch > lab_cfg.midi_max:
            continue

        p = int(n.pitch - lab_cfg.midi_min)
        start_f = _sec_to_frame_round(float(n.start), feat_cfg)
        end_f = _sec_to_frame_round(float(n.end), feat_cfg)

        # Ensure at least one active frame for extremely short or rounded notes.
        if end_f <= start_f:
            end_f = start_f + 1

        # Frame activity: [start_f, end_f)
        fs, fe = _clip_interval(start_f, end_f, n_frames)
        if fe > fs:
            frame[p, fs:fe] = 1.0

        # Onset: [start_f, start_f + onset_hops)
        os, oe = _clip_interval(start_f, start_f + onset_hops, n_frames)
        if oe > os:
            onset[p, os:oe] = 1.0
            if velocity is not None:
                # Official-style velocity target is normalized MIDI velocity.
                v = float(max(1, min(127, int(n.velocity)))) / 128.0
                velocity[p, os:oe] = v

        # Offset: [end_f - offset_hops, end_f)
        # Keep the offset marker near the note end. If the note is shorter than
        # offset_hops, this naturally overlaps the active frame/onset.
        off_s, off_e = _clip_interval(end_f - offset_hops, end_f, n_frames)
        if off_e > off_s:
            offset[p, off_s:off_e] = 1.0

    return LabelTensors(onset=onset, frame=frame, offset=offset, velocity=velocity)


@dataclass
class LabelMarginals:
    """Marginal distribution stats for distribution matching."""
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