from __future__ import annotations

"""Official-style Onsets and Frames model used by groupmm/onsets_frames_semisup.

This file is intended as a drop-in replacement for the current `oaf/model.py`.
It keeps the current project API:

    model = OnsetsAndFrames(ModelConfig(...))
    out = model(log_mel)  # log_mel: (B, F, T)
    out["onset"], out["offset"], out["frame"]  # each: (B, P, T)

Internally, the architecture follows the official implementation:
  - ConvStack: two 3x3 convs, frequency pooling, dropout, fc
  - onset_stack: ConvStack + BiLSTM + Linear + Sigmoid
  - offset_stack: ConvStack + BiLSTM + Linear + Sigmoid
  - frame_stack: ConvStack + Linear + Sigmoid
  - combined_stack: BiLSTM over [onset.detach(), offset.detach(), activation]
  - velocity_stack: ConvStack + Linear

The official implementation expects mel as (B, T, F).  The surrounding code in
this reproduction currently uses (B, F, T), so this wrapper transposes inputs and
outputs accordingly.
"""

from dataclasses import asdict
from typing import Dict, Optional

import torch
from torch import nn

try:  # package import, when used as oaf.model
    from .config import ModelConfig
except Exception:  # pragma: no cover - allows quick standalone smoke tests
    from config import ModelConfig  # type: ignore


class BiLSTM(nn.Module):
    """BiLSTM module matching the official onsets_frames_semisup implementation.

    In eval mode, it processes long sequences in chunks to reduce memory usage,
    following the upstream implementation.
    """

    inference_chunk_length = 512

    def __init__(self, input_features: int, recurrent_features: int):
        super().__init__()
        self.rnn = nn.LSTM(
            input_features,
            recurrent_features,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.rnn.flatten_parameters()

        if self.training:
            return self.rnn(x)[0]

        # Evaluation mode: support longer sequences that may not fit in memory.
        batch_size, sequence_length, _input_features = x.shape
        hidden_size = self.rnn.hidden_size
        num_directions = 2 if self.rnn.bidirectional else 1

        h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device, dtype=x.dtype)
        output = torch.zeros(
            batch_size,
            sequence_length,
            num_directions * hidden_size,
            device=x.device,
            dtype=x.dtype,
        )

        slices = range(0, sequence_length, self.inference_chunk_length)

        # Forward direction.
        for start in slices:
            end = min(start + self.inference_chunk_length, sequence_length)
            output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

        # Reverse direction.
        if self.rnn.bidirectional:
            h.zero_()
            c.zero_()
            for start in reversed(range(0, sequence_length, self.inference_chunk_length)):
                end = min(start + self.inference_chunk_length, sequence_length)
                result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

        return output


class ConvStack(nn.Module):
    """Official-style ConvStack for Onsets and Frames.

    Args:
        input_features: number of mel bins, normally 229.
        output_features: model size, normally model_complexity * 16 = 768.

    Input shape:
        mel: (B, T, F)

    Output shape:
        features: (B, T, output_features)
    """

    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)

        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2: pool frequency only; time resolution is preserved
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3: pool frequency only again
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() != 3:
            raise ValueError(f"Expected mel of shape (B,T,F), got {tuple(mel.shape)}")

        # Official implementation uses `view`; `reshape` is safer after transpose.
        x = mel.reshape(mel.size(0), 1, mel.size(1), mel.size(2))  # (B,1,T,F)
        x = self.cnn(x)                                           # (B,C,T,F')
        x = x.transpose(1, 2).flatten(-2)                         # (B,T,C*F')
        x = self.fc(x)                                            # (B,T,D)
        return x


class OnsetsAndFrames(nn.Module):
    """Official-style Onsets and Frames with current-project compatible API.

    The upstream paper/repo uses `input_features=229`, `output_features=88`, and
    `model_complexity=48`.  `model_complexity` is read from cfg if present;
    otherwise it defaults to 48.

    Forward input:
        log_mel: (B, F, T) as produced by the current `features.py`.
                 (B, T, F) is also accepted.

    Forward output:
        dict with probabilities in current format:
            onset:  (B, P, T)
            offset: (B, P, T)
            frame:  (B, P, T)
        If cfg.use_velocity_head is True, also returns:
            velocity: (B, P, T), raw velocity logits/values as in upstream model.

    Notes:
        The paper/reproduction normally ignores velocity estimation. Therefore,
        this compatibility implementation constructs the velocity stack only when
        cfg.use_velocity_head is True. This avoids DDP unused-parameter errors
        when training without velocity loss.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        input_features = int(cfg.n_mels)
        output_features = int(cfg.n_pitches)
        model_complexity = int(getattr(cfg, "model_complexity", 48))
        model_size = model_complexity * 16

        def sequence_model(input_size: int, output_size: int) -> BiLSTM:
            return BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

        # Upstream name: activation_pred from frame_stack, then final frame_pred
        # from combined_stack.
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

        if bool(getattr(cfg, "use_velocity_head", False)):
            self.velocity_stack: Optional[nn.Module] = nn.Sequential(
                ConvStack(input_features, model_size),
                nn.Linear(model_size, output_features),
            )
        else:
            self.velocity_stack = None

    def _to_official_layout(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Convert current (B,F,T) log-mel to official (B,T,F) layout."""
        if log_mel.dim() != 3:
            raise ValueError(f"Expected log_mel of shape (B,F,T) or (B,T,F), got {tuple(log_mel.shape)}")

        n_mels = int(self.cfg.n_mels)
        if log_mel.shape[1] == n_mels:
            # Current project layout: (B,F,T) -> official layout: (B,T,F)
            return log_mel.transpose(1, 2).contiguous()
        if log_mel.shape[2] == n_mels:
            # Already official layout: (B,T,F)
            return log_mel.contiguous()

        raise ValueError(
            f"Could not infer mel layout from shape {tuple(log_mel.shape)} with n_mels={n_mels}. "
            "Expected (B,F,T) or (B,T,F)."
        )

    @staticmethod
    def _to_current_layout(pred: torch.Tensor) -> torch.Tensor:
        """Convert official (B,T,P) predictions to current (B,P,T)."""
        return pred.transpose(1, 2).contiguous()

    def forward(self, log_mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        mel = self._to_official_layout(log_mel)  # (B,T,F)

        onset_pred = self.onset_stack(mel)       # (B,T,P)
        offset_pred = self.offset_stack(mel)     # (B,T,P)
        activation_pred = self.frame_stack(mel)  # (B,T,P)

        # Official behavior: detach onset/offset before combined frame stack.
        combined_pred = torch.cat(
            [onset_pred.detach(), offset_pred.detach(), activation_pred],
            dim=-1,
        )
        frame_pred = self.combined_stack(combined_pred)  # (B,T,P)

        out: Dict[str, torch.Tensor] = {
            "onset": self._to_current_layout(onset_pred),
            "offset": self._to_current_layout(offset_pred),
            "frame": self._to_current_layout(frame_pred),
        }

        if self.velocity_stack is not None:
            velocity_pred = self.velocity_stack(mel)  # (B,T,P), upstream uses no sigmoid
            out["velocity"] = self._to_current_layout(velocity_pred)

        return out

    def extra_repr(self) -> str:
        try:
            return str(asdict(self.cfg))
        except Exception:
            return repr(self.cfg)
