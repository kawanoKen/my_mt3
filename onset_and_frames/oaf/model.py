from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class ConvStack(nn.Module):
    """Convolutional frontend that preserves time resolution and downsamples frequency.

    Input: (B, 1, F, T)
    Output: per-frame features (B, T, proj_dim)

    Notes:
      The O&F paper specifies a "conv stack" but does not fully detail the layer layout.
      This implementation uses a common pattern for AMT: several 3x3 conv blocks + pooling over frequency only.
    """

    def __init__(self, n_mels: int, channels: Tuple[int, ...], proj_dim: int, dropout: float = 0.25):
        super().__init__()
        layers = []
        in_ch = 1
        for ch in channels:
            layers += [
                nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                # pool only over frequency to keep frame rate intact
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ]
            in_ch = ch
        self.conv = nn.Sequential(*layers)

        # infer flattened dim after conv by running a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, 10)  # (B,C,F,T)
            y = self.conv(dummy)
            _, c, f, _t = y.shape
            flat_dim = c * f
        self.proj = nn.Linear(flat_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, n_mels, T)
        Returns:
            feat: (B, T, proj_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected (B,F,T) got {tuple(x.shape)}")
        x = x.unsqueeze(1)  # (B,1,F,T)
        y = self.conv(x)    # (B,C,F',T)
        y = y.permute(0, 3, 1, 2).contiguous()  # (B,T,C,F')
        B, T, C, Fp = y.shape
        y = y.view(B, T, C * Fp)
        y = self.proj(y)
        y = F.relu(y)
        return y


class OnsetsAndFrames(nn.Module):
    """Onsets and Frames model (onset + frame, optional offset/velocity heads).

    Outputs are framewise probabilities for each of the 88 piano keys.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.onset_cnn = ConvStack(cfg.n_mels, cfg.conv_channels, cfg.proj_dim, cfg.conv_dropout)
        self.onset_lstm = nn.LSTM(
            input_size=cfg.proj_dim, hidden_size=cfg.lstm_hidden, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.onset_fc = nn.Linear(2 * cfg.lstm_hidden, cfg.n_pitches)

        self.frame_cnn = ConvStack(cfg.n_mels, cfg.conv_channels, cfg.proj_dim, cfg.conv_dropout)
        self.frame_fc1 = nn.Linear(cfg.proj_dim, cfg.n_pitches)
        self.frame_lstm = nn.LSTM(
            input_size=2 * cfg.n_pitches, hidden_size=cfg.lstm_hidden, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.frame_fc2 = nn.Linear(2 * cfg.lstm_hidden, cfg.n_pitches)

        if cfg.use_offset_head:
            self.offset_cnn = ConvStack(cfg.n_mels, cfg.conv_channels, cfg.proj_dim, cfg.conv_dropout)
            self.offset_lstm = nn.LSTM(
                input_size=cfg.proj_dim, hidden_size=cfg.lstm_hidden, num_layers=1,
                batch_first=True, bidirectional=True
            )
            self.offset_fc = nn.Linear(2 * cfg.lstm_hidden, cfg.n_pitches)
        else:
            self.offset_cnn = None
            self.offset_lstm = None
            self.offset_fc = None

        if cfg.use_velocity_head:
            self.velocity_cnn = ConvStack(cfg.n_mels, cfg.conv_channels, cfg.proj_dim, cfg.conv_dropout)
            self.velocity_fc = nn.Linear(cfg.proj_dim, cfg.n_pitches)
        else:
            self.velocity_cnn = None
            self.velocity_fc = None

    def forward(self, log_mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:
            log_mel: (B, n_mels, T)
        Returns:
            dict with keys: onset, frame, (offset), (velocity)
            Each is (B, n_pitches, T) in [0,1]
        """
        # onset
        onset_feat = self.onset_cnn(log_mel)  # (B,T,D)
        onset_seq, _ = self.onset_lstm(onset_feat)  # (B,T,2H)
        onset_logits = self.onset_fc(onset_seq)     # (B,T,P)
        onset = torch.sigmoid(onset_logits).transpose(1, 2)  # (B,P,T)

        # frame
        frame_feat = self.frame_cnn(log_mel)  # (B,T,D)
        frame_logits1 = self.frame_fc1(frame_feat)         # (B,T,P)
        frame_pre = torch.sigmoid(frame_logits1)           # (B,T,P)
        frame_in = torch.cat([frame_pre, onset.transpose(1, 2)], dim=-1)  # (B,T,2P)
        frame_seq, _ = self.frame_lstm(frame_in)  # (B,T,2H)
        frame_logits2 = self.frame_fc2(frame_seq)  # (B,T,P)
        frame = torch.sigmoid(frame_logits2).transpose(1, 2)  # (B,P,T)

        out: Dict[str, torch.Tensor] = {"onset": onset, "frame": frame}

        if self.cfg.use_offset_head and self.offset_cnn is not None:
            off_feat = self.offset_cnn(log_mel)
            off_seq, _ = self.offset_lstm(off_feat)
            off_logits = self.offset_fc(off_seq)
            offset = torch.sigmoid(off_logits).transpose(1, 2)
            out["offset"] = offset

        if self.cfg.use_velocity_head and self.velocity_cnn is not None:
            v_feat = self.velocity_cnn(log_mel)   # (B,T,D)
            v_logits = self.velocity_fc(v_feat)   # (B,T,P)
            velocity = torch.sigmoid(v_logits).transpose(1, 2)  # (B,P,T) in [0,1]
            out["velocity"] = velocity

        return out

    def extra_repr(self) -> str:
        return str(asdict(self.cfg))
