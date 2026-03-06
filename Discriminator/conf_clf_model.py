from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfClfCfg:
    vocab_size: int
    max_len: int = 512  # including [CLS]
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1

    pad_id: int = 0
    cls_id: int = 1


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0).to(x.dtype)


class TransformerConfidenceClf(nn.Module):
    """
    Transformer Encoder + [CLS] binary classifier + optional MLM head.
    Input: token ids (B, L) with [CLS] at position 0.
    Output (forward):     logits (B,)  — sigmoid(logits) = confidence in [0,1].
    Output (forward_mlm): logits (B, L, vocab_size) — per-position vocab prediction.
    """
    def __init__(self, cfg: ConfClfCfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.dropout = nn.Dropout(cfg.dropout)

        # [CLS] binary classification head
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 1),
        )

        # MLM head: predict original token at each masked position
        self.mlm_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.vocab_size),
        )

        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.head[-1].bias.fill_(-1.0)

    def encode(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared encoder: tokens (B,L) -> hidden (B,L,D)."""
        assert tokens.dim() == 2
        B, L = tokens.shape
        assert L <= self.cfg.max_len, f"input length {L} exceeds max_len {self.cfg.max_len}"

        x = self.tok_emb(tokens)
        x = self.pos_enc(x)
        x = self.dropout(x)

        if attn_mask is None:
            src_key_padding_mask = (tokens == self.cfg.pad_id)
        else:
            src_key_padding_mask = ~attn_mask

        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Binary classification: returns logits (B,)."""
        h = self.encode(tokens, attn_mask)
        cls_h = h[:, 0, :]
        return self.head(cls_h).squeeze(-1)

    def forward_mlm(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """MLM: returns per-position logits (B, L, vocab_size)."""
        h = self.encode(tokens, attn_mask)
        return self.mlm_head(h)

    @torch.no_grad()
    def score(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.forward(tokens, attn_mask=attn_mask)
        return torch.sigmoid(logits)
