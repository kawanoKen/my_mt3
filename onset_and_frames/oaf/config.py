from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 229
    f_min: float = 0.0
    f_max: Optional[float] = None  # None -> sr/2
    log_eps: float = 1e-6


@dataclass
class LabelConfig:
    # Piano range: 88 keys (A0=21 ... C8=108)
    midi_min: int = 21
    midi_max: int = 108
    onset_length_sec: float = 0.032  # 32ms (paper default)
    offset_length_sec: float = 0.032  # for the extended model; if unknown, keep same as onset
    use_sustain: bool = True
    sustain_cc: int = 64
    sustain_threshold: int = 64  # CC value >= this => sustain ON


@dataclass
class ModelConfig:
    n_mels: int = 229
    n_pitches: int = 88
    # conv stack
    conv_channels: Tuple[int, ...] = (32, 32, 64, 64)
    conv_dropout: float = 0.25
    # sequence model
    lstm_hidden: int = 128
    # whether to include the offset head (for the 2019+ "extended" variant used in the SSL paper)
    use_offset_head: bool = True
    # whether to include the velocity head (independent branch; optional)
    use_velocity_head: bool = False
    # projection dim after conv flattening (per frame)
    proj_dim: int = 512
    model_complexity: int = 48


@dataclass
class TrainConfig:
    batch_size_labeled: int = 8
    batch_size_unlabeled: int = 8

    # segmenting (training uses fixed-length crops)
    segment_seconds: float = 20.48
    # if True, try to avoid starting a segment in the middle of a note for labeled data
    avoid_note_splits: bool = False

    # optimizer / schedule
    lr: float = 2e-5  # SSL paper default
    lr_decay_gamma: float = 0.98
    lr_decay_every_steps: int = 5000
    grad_clip_norm: float = 1.0

    # iterations
    iters_supervised: int = 50_000
    iters_ssl: int = 50_000

    # logging / checkpoints
    log_every: int = 50
    ckpt_every: int = 1000
    valid_every: int = 2000
    num_workers: int = 4

    device: str = "cuda"  # or "cpu"
    amp: bool = True


@dataclass
class DecodeConfig:
    tau_on: float = 0.5
    tau_fr: float = 0.5
    tau_off: float = 0.5
    # if True, enforce "no new note without onset" gating
    onset_gating: bool = True
