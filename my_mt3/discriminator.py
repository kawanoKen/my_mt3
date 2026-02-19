import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class Discriminator(nn.Module):
    """
    Paper's 3-layer fully-connected discriminator.
    Input: encoder memory mem [B, T, D]
    Procedure:
      - sample a short chunk (e.g., 0.1 sec worth of frames) along time
      - temporal pooling -> [B, D]
      - 3-layer MLP -> domain logit [B]
    """
    def __init__(self, d: int = 384, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),  # logit
        )

    @torch.no_grad()
    def _sample_start(self, T: int, chunk_frames: int, device):
        """Return random start indices (scalar) for cropping."""
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be >= 1")
        if T <= chunk_frames:
            return 0
        # randint is inclusive low, exclusive high
        return torch.randint(0, T - chunk_frames + 1, (1,), device=device).item()

    def forward(self, mem: torch.Tensor, *, chunk_frames: int = 10, pool: str = "mean"):
        """
        mem: [B, T, D]
        chunk_frames: number of frames corresponding to ~0.1s (paper uses 0.1 sec chunks)
        pool: "mean" or "max"
        returns: logits [B] (domain logit; use BCEWithLogitsLoss)
        """
        if mem.dim() != 3:
            raise ValueError(f"mem must be [B,T,D], got {tuple(mem.shape)}")
        B, T, D = mem.shape
        if chunk_frames > T:
            # fall back to whole sequence
            chunk = mem
        else:
            s = self._sample_start(T, chunk_frames, mem.device)
            chunk = mem[:, s:s + chunk_frames, :]  # [B, K, D]

        if pool == "mean":
            x = chunk.mean(dim=1)           # [B, D]
        elif pool == "max":
            x = chunk.max(dim=1).values     # [B, D]
        else:
            raise ValueError("pool must be 'mean' or 'max'")

        logits = self.net(x).squeeze(-1)    # [B]
        return logits

class _GRL(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grl(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GRL.apply(x, lambd)
