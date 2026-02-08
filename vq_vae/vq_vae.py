from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Residual Vector Quantizer (EMA)
# ----------------------------
class ResidualVQEMA(nn.Module):
    """
    Residual VQ with k codebooks (layers), each having n_codes entries of dim d.
    Codebook is updated by EMA using assigned residual vectors.
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
        enc: (B, D) vectors assigned to codes (use r_i, i.e., "before subtract")
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

    @torch.no_grad()
    def update_codebook_only(self, z: torch.Tensor) -> None:
        """
        Phase-2 用：codebook を EMA で更新するだけ（loss無し、勾配無し）。
        z: (B, D) continuous latent (detach 済み推奨)
        """
        assert z.dim() == 2 and z.size(-1) == self.d
        r = z
        for i in range(self.n_layers):
            cb = self.codebook[i]

            r_before = r
            r2 = (r_before ** 2).sum(dim=1, keepdim=True)
            cb2 = (cb ** 2).sum(dim=1).unsqueeze(0)
            dist = r2 + cb2 - 2 * (r_before @ cb.t())
            indices = dist.argmin(dim=1)

            c_i = cb[indices]
            r = r_before - c_i

            self._ema_update(i, r_before, indices)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推論・TTAで使えるように：z を residual VQ して
        c_sum と indices を返す（EMA更新はしない）。
        z: (B,D)
        """
        assert z.dim() == 2 and z.size(-1) == self.d
        r = z
        c_sum = torch.zeros_like(z)
        all_idx = []
        for i in range(self.n_layers):
            cb = self.codebook[i]
            r2 = (r ** 2).sum(dim=1, keepdim=True)
            cb2 = (cb ** 2).sum(dim=1).unsqueeze(0)
            dist = r2 + cb2 - 2 * (r @ cb.t())
            idx = dist.argmin(dim=1)
            all_idx.append(idx)
            c_i = cb[idx]
            c_sum = c_sum + c_i
            r = r - c_i
        indices = torch.stack(all_idx, dim=1)  # (B, n_layers)
        return c_sum, indices


# ----------------------------
# Temporal Conv1D Encoder/Decoder
# ----------------------------
class ResBlock1D(nn.Module):
    def __init__(self, ch: int, *, k: int = 3, dropout: float = 0.0):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, padding=p)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, padding=p)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = F.relu(self.conv1(x), inplace=True)
        h = self.drop(h)
        h = self.conv2(h)
        return F.relu(x + h, inplace=True)

class DownBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int = 4, s: int = 2, p: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class UpBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return F.relu(self.conv(x), inplace=True)


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

class ResNetEncoder1D(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int = 256, width1: int = 256, width2: int = 512, n_blocks: int = 2):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, width1, kernel_size=3, padding=1)
        self.blocks1 = nn.Sequential(*[ResBlock1D(width1) for _ in range(n_blocks)])
        self.down1 = DownBlock1D(width1, width1)   # /2
        self.blocks2 = nn.Sequential(*[ResBlock1D(width1) for _ in range(n_blocks)])
        self.down2 = DownBlock1D(width1, width2)   # /4
        self.blocks3 = nn.Sequential(*[ResBlock1D(width2) for _ in range(n_blocks)])
        self.proj = nn.Conv1d(width2, latent_dim, kernel_size=1)

    def forward(self, x):  # (B,K,T)
        h = F.relu(self.stem(x), inplace=True)
        h = self.blocks1(h)
        h = self.down1(h)
        h = self.blocks2(h)
        h = self.down2(h)
        h = self.blocks3(h)
        z = self.proj(h)  # (B, latent_dim, T//4)
        return z

class ResNetDecoder1D(nn.Module):
    def __init__(self, latent_dim: int = 256, out_ch: int = 9, width2: int = 512, width1: int = 256, n_blocks: int = 2):
        super().__init__()
        self.pre = nn.Conv1d(latent_dim, width2, kernel_size=1)
        self.blocks1 = nn.Sequential(*[ResBlock1D(width2) for _ in range(n_blocks)])
        self.up1 = UpBlock1D(width2, width1)   # x2
        self.blocks2 = nn.Sequential(*[ResBlock1D(width1) for _ in range(n_blocks)])
        self.up2 = UpBlock1D(width1, width1)   # x4
        self.blocks3 = nn.Sequential(*[ResBlock1D(width1) for _ in range(n_blocks)])
        self.out = nn.Conv1d(width1, out_ch, kernel_size=3, padding=1)

    def forward(self, z):  # (B, latent_dim, T//4)
        h = F.relu(self.pre(z), inplace=True)
        h = self.blocks1(h)
        h = self.up1(h)
        h = self.blocks2(h)
        h = self.up2(h)
        h = self.blocks3(h)
        y = self.out(h)  # (B,K,T)
        return y




class Encoder1D(nn.Module):
    def __init__(self, in_ch: int, d: int = 256):
        super().__init__()
        self.b1 = ResBlock1D(in_ch, 256, k=3, s=1, p=1)
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
# Drum Denoise VQ model (Phase-1 + utilities)
# ----------------------------
@dataclass
class DrumVQCfg:
    n_drums: int = 9
    latent_dim: int = 256
    n_codes: int = 512
    n_layers: int = 3
    ema_decay: float = 0.99
    # Phase-1 uses ONLY rec loss, so these are not used there.
    beta_commit: float = 0.0
    gamma_anchor: float = 0.0


class DrumDenoiseVQVAE(nn.Module):
    def __init__(self, cfg: DrumVQCfg):
        super().__init__()
        self.cfg = cfg
        self.enc = ResNetEncoder1D(in_ch=cfg.n_drums, latent_dim=cfg.latent_dim, width1=512, width2=1024, n_blocks=3)
        self.vq = ResidualVQEMA(
            d=cfg.latent_dim,
            n_codes=cfg.n_codes,
            n_layers=cfg.n_layers,
            decay=cfg.ema_decay,
        )
        self.dec = ResNetDecoder1D(latent_dim=cfg.latent_dim, out_ch=cfg.n_drums, width2=1024, width1=512, n_blocks=3)

    # ---- Phase-1: denoiser only (NO VQ in loss path) ----
    def forward_phase1(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B,T,K) in {0,1} float
        returns: {"y_logits": (B,T,K)}
        """
        x_ch = x.transpose(1, 2).contiguous()     # (B,K,T)
        z_seq = self.enc(x_ch)                    # (B,d,T//4)
        y_logits = self.dec(z_seq)                # (B,K,T)
        y_logits = y_logits.transpose(1, 2).contiguous()  # (B,T,K)
        return {"y_logits": y_logits}

    def loss_phase1(self, out: dict, y: torch.Tensor):
        y_logits = out["y_logits"]
        weight_value = 20.0 
        pos_weight = torch.tensor([weight_value], device=y.device)

        # 引数に pos_weight を追加
        rec = F.binary_cross_entropy_with_logits(
            y_logits, 
            y, 
            pos_weight=pos_weight
        )
        return rec, {"rec": rec.detach()}

    # ---- Utility: encode to z_flat for phase-2 fitting ----
    @torch.no_grad()
    def encode_to_z_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,K)
        returns z_flat: (B*Tq, d)
        """
        x_ch = x.transpose(1, 2).contiguous()
        z_seq = self.enc(x_ch)
        B, d, Tq = z_seq.shape
        z_flat = z_seq.permute(0, 2, 1).reshape(B * Tq, d)
        return z_flat


# ----------------------------
# Noise function (noisy MIDI -> roll)
# ----------------------------
@torch.no_grad()
def corrupt_drum_roll(
    y: torch.Tensor,
    p_drop: float = 0.1,
    p_add: float = 0.4,
    jitter: int = 0,
    p_mask_block: float = 0,
    mask_block_len: int = 16,
) -> torch.Tensor:
    """
    y: (B, T, K) clean roll in {0,1}
    returns x: corrupted roll
    """
    B, T, K = y.shape
    x = y.clone()

    drop = (torch.rand_like(x) < p_drop) & (x > 0.5)
    x[drop] = 0.0

    add = (torch.rand_like(x) < p_add) & (x < 0.5)
    x[add] = 1.0

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

    if p_mask_block > 0:
        for b in range(B):
            if torch.rand(()) < p_mask_block:
                t0 = torch.randint(0, max(1, T - mask_block_len), (1,), device=y.device).item()
                x[b, t0 : t0 + mask_block_len, :] = 0.0

    return x


# ----------------------------
# Phase-1 minimal train step
# ----------------------------
def train_step_phase1(model: DrumDenoiseVQVAE, opt: torch.optim.Optimizer, y_clean: torch.Tensor):
    model.train()
    module = model.module if hasattr(model, "module") else model

    x_noisy = corrupt_drum_roll(y_clean)
    out = module.forward_phase1(x_noisy)
    loss, logs = module.loss_phase1(out, y_clean)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    return loss.item(), {k: float(v) for k, v in logs.items()}
