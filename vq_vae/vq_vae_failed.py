from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Residual Vector Quantizer with EMA updates
# ----------------------------

class ResidualVQEMA(nn.Module):
    """
    Residual VQ with k codebooks (layers), each having n_codes entries of dim d.
    Codebook is updated by EMA using assigned residual vectors (like VQ-VAE-EMA).
    """
    def __init__(
        self,
        d: int,
        n_codes: int = 512,
        n_layers: int = 3,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d = d
        self.n_codes = n_codes
        self.n_layers = n_layers
        self.decay = decay
        self.eps = eps

        # codebooks: (k, n_codes, d)
        codebook = torch.randn(n_layers, n_codes, d) / math.sqrt(d)
        self.register_buffer("codebook", codebook)

        # EMA state
        self.register_buffer("ema_cluster_size", torch.zeros(n_layers, n_codes))
        self.register_buffer("ema_codebook_sum", torch.zeros(n_layers, n_codes, d))

    @torch.no_grad()
    def _ema_update(self, layer: int, enc: torch.Tensor, indices: torch.Tensor):
        """
        enc: (B, D) residual vectors assigned to codes
        indices: (B,) nearest code index
        """
        onehot = F.one_hot(indices, num_classes=self.n_codes).type_as(enc)  # (B, n_codes)
        cluster_size = onehot.sum(dim=0)  # (n_codes,)
        code_sum = onehot.t() @ enc       # (n_codes, D)

        self.ema_cluster_size[layer].mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_codebook_sum[layer].mul_(self.decay).add_(code_sum, alpha=1 - self.decay)

        # Laplace smoothing
        n = self.ema_cluster_size[layer].sum()
        cluster_size_smoothed = (self.ema_cluster_size[layer] + self.eps) / (n + self.n_codes * self.eps) * n
        new_codebook = self.ema_codebook_sum[layer] / cluster_size_smoothed.unsqueeze(-1)
        self.codebook[layer].copy_(new_codebook)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, D) continuous latent
        returns:
          c: (B, D) quantized latent sum (anchor latent)
          z_q_st: (B, D) straight-through quantized latent (for gradients)
          commit_loss: scalar tensor
        """
        assert z.dim() == 2 and z.size(-1) == self.d

        r = z
        c_sum = torch.zeros_like(z)
        all_indices = []

        for i in range(self.n_layers):
            cb = self.codebook[i]

            r_before = r  # ★追加：更新前の残差を保持

            r2 = (r_before ** 2).sum(dim=1, keepdim=True)
            cb2 = (cb ** 2).sum(dim=1).unsqueeze(0)
            dist = r2 + cb2 - 2 * (r_before @ cb.t())
            indices = dist.argmin(dim=1)

            c_i = cb[indices]
            c_sum = c_sum + c_i
            r = r_before - c_i  # residual update

            if self.training:
                self._ema_update(i, r_before.detach(), indices.detach())  # ★r_before を使う


        # straight-through estimator: gradients pass to z, value is c_sum
        z_q_st = z + (c_sum - z).detach()
        commit_loss = F.mse_loss(z, c_sum.detach())

        return c_sum, z_q_st, commit_loss


# ----------------------------
# Simple Temporal Conv1D Encoder/Decoder
# ----------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Encoder1D(nn.Module):
    def __init__(self, in_ch: int, d: int = 256):
        super().__init__()
        self.b1 = ConvBlock1D(in_ch, 256, k=3, s=1, p=1)
        self.down1 = nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1)  # /2
        self.b2 = ConvBlock1D(256, 512, k=3, s=1, p=1)
        self.down2 = nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1)  # /4
        self.proj = nn.Conv1d(512, d, kernel_size=1)

    def forward(self, x):  # x: (B, K, T)
        h = self.b1(x)
        h = F.relu(self.down1(h), inplace=True)
        h = self.b2(h)
        h = F.relu(self.down2(h), inplace=True)
        z = self.proj(h)  # (B, d, T//4)
        return z


class Decoder1D(nn.Module):
    def __init__(self, d: int = 256, out_ch: int = 9):
        super().__init__()
        self.pre = nn.Conv1d(d, 512, kernel_size=1)
        self.b1 = ConvBlock1D(512, 512, k=3, s=1, p=1)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.b2 = ConvBlock1D(512, 256, k=3, s=1, p=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.out = nn.Conv1d(256, out_ch, kernel_size=3, stride=1, padding=1)  # logits

    def forward(self, z):  # z: (B, d, T//4)
        h = F.relu(self.pre(z), inplace=True)
        h = self.b1(h)
        h = self.up1(h)
        h = self.b2(h)
        h = self.up2(h)
        y_logits = self.out(h)  # (B, K, T)
        return y_logits


# ----------------------------
# Drum Denoising VQ-AE
# ----------------------------
@dataclass
class DrumVQCfg:
    n_drums: int = 9
    latent_dim: int = 256
    n_codes: int = 512
    n_layers: int = 3
    ema_decay: float = 0.99
    beta_commit: float = 0.25
    gamma_anchor: float = 0.5  # set 0 to disable anchor loss


class DrumDenoiseVQVAE(nn.Module):
    def __init__(self, cfg: DrumVQCfg):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder1D(in_ch=cfg.n_drums, d=cfg.latent_dim)
        self.vq = ResidualVQEMA(
            d=cfg.latent_dim,
            n_codes=cfg.n_codes,
            n_layers=cfg.n_layers,
            decay=cfg.ema_decay,
        )
        self.dec = Decoder1D(d=cfg.latent_dim, out_ch=cfg.n_drums)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, K) in {0,1} float
        returns dict with:
          y_logits: denoised logits from continuous latent
          y_anchor_logits: anchor logits from quantized latent
          commit_loss
        """
        # to (B, K, T)
        x_ch = x.transpose(1, 2).contiguous()

        z_seq = self.enc(x_ch)                 # (B, d, T//4)
        # quantize per time step => flatten (B*(T//4), d)
        B, d, Tq = z_seq.shape
        z_flat = z_seq.permute(0, 2, 1).reshape(B * Tq, d)

        c_flat, z_q_st_flat, commit_loss = self.vq(z_flat)

        # reshape back (B, d, T//4)
        z_q_st = z_q_st_flat.view(B, Tq, d).permute(0, 2, 1).contiguous()
        c = c_flat.view(B, Tq, d).permute(0, 2, 1).contiguous()

        y_logits = self.dec(z_q_st)            # (B, K, T)
        y_anchor_logits = self.dec(c)          # (B, K, T)

        # back to (B, T, K)
        y_logits = y_logits.transpose(1, 2).contiguous()
        y_anchor_logits = y_anchor_logits.transpose(1, 2).contiguous()

        # distance (logits)
        with torch.no_grad():
            dist_ylogit_anchor = torch.sqrt(torch.mean((y_logits - y_anchor_logits) ** 2))

        return {
            "y_logits": y_logits,
            "y_anchor_logits": y_anchor_logits,
            "commit_loss": commit_loss,
            "dist_ylogit_anchor": dist_ylogit_anchor,
        }

    def loss(self, out: dict, y: torch.Tensor):
        """
        y: (B, T, K) clean target in {0,1} float
        """
        y_logits = out["y_logits"]
        y_anchor_logits = out["y_anchor_logits"]
        commit_loss = out["commit_loss"]

        rec = F.binary_cross_entropy_with_logits(y_logits, y)
        anch = F.binary_cross_entropy_with_logits(y_anchor_logits, y)

        total = rec + self.cfg.beta_commit * commit_loss + self.cfg.gamma_anchor * anch
        logs = {
            "rec": rec.detach(),
            "anch": anch.detach(),
            "commit": commit_loss.detach(),
        }
        # optional distance from forward
        if "dist_ylogit_anchor" in out:
            logs["dist_ylogit_anchor"] = out["dist_ylogit_anchor"].detach()
        return total, logs


# ----------------------------
# Example noise function (noisy MIDI -> roll)
# ----------------------------
@torch.no_grad()
def corrupt_drum_roll(
    y: torch.Tensor,
    p_drop: float = 0.3,
    p_add: float = 0.5,
    jitter: int = 2,
    p_mask_block: float = 0.25,
    mask_block_len: int = 16,
):
    """
    y: (B, T, K) clean roll in {0,1}
    returns x: corrupted roll
    """
    B, T, K = y.shape
    x = y.clone()

    # drop true hits
    drop = (torch.rand_like(x) < p_drop) & (x > 0.5)
    x[drop] = 0.0

    # add spurious hits
    add = (torch.rand_like(x) < p_add) & (x < 0.5)
    x[add] = 1.0

    # timing jitter (shift whole time axis per-hit approx; cheap version: shift channels)
    if jitter > 0:
        shift = torch.randint(-jitter, jitter + 1, (B, K), device=y.device)
        xj = torch.zeros_like(x)
        for b in range(B):
            for k in range(K):
                s = int(shift[b, k].item())
                if s == 0:
                    xj[b, :, k] = x[b, :, k]
                elif s > 0:
                    xj[b, s:, k] = x[b, : T - s, k]
                else:
                    xj[b, : T + s, k] = x[b, -s:, k]
        x = xj

    # random block mask in time
    if p_mask_block > 0:
        for b in range(B):
            if torch.rand(()) < p_mask_block:
                t0 = torch.randint(0, max(1, T - mask_block_len), (1,), device=y.device).item()
                x[b, t0 : t0 + mask_block_len, :] = 0.0

    return x


# ----------------------------
# Minimal training step
# ----------------------------
def train_step(model: DrumDenoiseVQVAE, opt: torch.optim.Optimizer, y_clean: torch.Tensor):
    model.train()
    module = model.module if hasattr(model, "module") else model
    x_noisy = corrupt_drum_roll(y_clean)

    out = model(x_noisy)
    loss, logs = module.loss(out, y_clean)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    return loss.item(), {k: float(v) for k, v in logs.items()}


