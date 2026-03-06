from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_heads(x: torch.Tensor, nhead: int) -> torch.Tensor:
    # x: [B,S,D] -> [B,H,S,Hd]
    B, S, D = x.shape
    Hd = D // nhead
    return x.view(B, S, nhead, Hd).transpose(1, 2).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # x: [B,H,S,Hd] -> [B,S,D]
    B, H, S, Hd = x.shape
    return x.transpose(1, 2).contiguous().view(B, S, H * Hd)


def _mha_proj_qkv(mha: nn.MultiheadAttention, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: [B,S,D] -> q,k,v: [B,S,D]
    W = mha.in_proj_weight
    b = mha.in_proj_bias
    D = mha.embed_dim
    y = F.linear(x, W, b)  # [B,S,3D]
    return y[..., :D], y[..., D:2*D], y[..., 2*D:]


def _mha_proj_q(mha: nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
    W = mha.in_proj_weight
    b = mha.in_proj_bias
    D = mha.embed_dim
    return F.linear(x, W[:D], b[:D] if b is not None else None)


def _mha_proj_kv_from_mem(mha: nn.MultiheadAttention, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # mem: [B,T,D] -> k,v: [B,T,D] using (K,V) blocks of in_proj
    W = mha.in_proj_weight
    b = mha.in_proj_bias
    D = mha.embed_dim
    Wk = W[D:2*D]; Wv = W[2*D:3*D]
    bk = b[D:2*D] if b is not None else None
    bv = b[2*D:3*D] if b is not None else None
    k = F.linear(mem, Wk, bk)
    v = F.linear(mem, Wv, bv)
    return k, v


@dataclass
class KVCache:
    # self-attn caches (pre-allocated)
    self_k: torch.Tensor   # [L,B,H,MAX,Hd]
    self_v: torch.Tensor   # [L,B,H,MAX,Hd]
    # cross-attn caches (precomputed once per mem)
    cross_k: torch.Tensor  # [L,B,H,Tmem,Hd]
    cross_v: torch.Tensor  # [L,B,H,Tmem,Hd]
    # current decoded length (scalar int)
    step: int = 0



class FastDecoderKV(nn.Module):
    def __init__(self, dec: nn.Module, max_len: int):
        super().__init__()
        assert max_len > 0
        self.dec = dec
        self.max_len = int(max_len)

        # reuse modules directly (weight sharing)
        self.emb = dec.emb
        self.lm = dec.lm
        self.blocks: nn.ModuleList = dec.blocks  # list of nn.TransformerDecoderLayer
        self.register_buffer("pe", dec.pos.pe, persistent=False)  # [max_pos, D]

        # head dims
        mha0 = self.blocks[0].self_attn
        assert mha0.batch_first, "Your code uses batch_first=True; required here."
        self.H = mha0.num_heads
        self.D = mha0.embed_dim
        assert self.D % self.H == 0
        self.Hd = self.D // self.H
        self.L = len(self.blocks)

        # This implementation assumes norm_first=False (default in your original)
        for b in self.blocks:
            if getattr(b, "norm_first", False):
                raise ValueError("This FastDecoderKV assumes norm_first=False. (Your original code uses default False.)")

    @torch.no_grad()
    def init_cache(self, mem: torch.Tensor) -> KVCache:
        """
        Pre-allocate self caches and precompute cross K/V for this mem.
        mem: [B,Tmem,D]
        """
        B, Tm, D = mem.shape
        device = mem.device
        dtype = mem.dtype

        # self caches
        self_k = torch.empty(self.L, B, self.H, self.max_len, self.Hd, device=device, dtype=dtype)
        self_v = torch.empty(self.L, B, self.H, self.max_len, self.Hd, device=device, dtype=dtype)
        # (optional) set to 0 for determinism; not required if you always mask by step
        self_k.zero_()
        self_v.zero_()

        # cross caches per layer (precompute using each layer's cross-attn projection weights)
        cross_k = torch.empty(self.L, B, self.H, Tm, self.Hd, device=device, dtype=dtype)
        cross_v = torch.empty(self.L, B, self.H, Tm, self.Hd, device=device, dtype=dtype)
        for li, layer in enumerate(self.blocks):
            k, v = _mha_proj_kv_from_mem(layer.multihead_attn, mem)  # [B,Tm,D]
            cross_k[li] = _split_heads(k, self.H)  # [B,H,Tm,Hd]
            cross_v[li] = _split_heads(v, self.H)

        return KVCache(self_k=self_k, self_v=self_v, cross_k=cross_k, cross_v=cross_v, step=0)

    def _self_attn_step(
        self,
        layer: nn.TransformerDecoderLayer,
        x: torch.Tensor,          # [B,1,D]
        cache: KVCache,
        li: int
    ) -> torch.Tensor:
        """
        Causal self-attn for one token, updating cache.self_k/v at position cache.step (in-place).
        """
        # project q,k,v for this step
        q, k, v = _mha_proj_qkv(layer.self_attn, x)   # [B,1,D] each
        qh = _split_heads(q, self.H)                  # [B,H,1,Hd]
        kh = _split_heads(k, self.H)                  # [B,H,1,Hd]
        vh = _split_heads(v, self.H)

        t = cache.step
        if t >= self.max_len:
            raise RuntimeError(f"KV cache overflow: step={t} >= max_len={self.max_len}")

        # write one position (in-place)
        cache.self_k[li, :, :, t:t+1, :] = kh
        cache.self_v[li, :, :, t:t+1, :] = vh

        # attend over [0..t] keys/values
        k_all = cache.self_k[li, :, :, :t+1, :]  # [B,H,t+1,Hd]
        v_all = cache.self_v[li, :, :, :t+1, :]

        # SDPA (query length=1, key length=t+1). Causal is already satisfied by truncation.
        # is_causal=False is fine here because keys don't include future positions.
        ctx = F.scaled_dot_product_attention(qh, k_all, v_all, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = _merge_heads(ctx)  # [B,1,D]
        out = layer.self_attn.out_proj(out)
        return out

    def _cross_attn_step(
        self,
        layer: nn.TransformerDecoderLayer,
        x: torch.Tensor,          # [B,1,D]
        cache: KVCache,
        li: int
    ) -> torch.Tensor:
        """
        Cross-attn for one token using precomputed cross_k/v.
        """
        q = _mha_proj_q(layer.multihead_attn, x)  # [B,1,D]
        qh = _split_heads(q, self.H)              # [B,H,1,Hd]

        k = cache.cross_k[li]  # [B,H,Tm,Hd]
        v = cache.cross_v[li]

        ctx = F.scaled_dot_product_attention(qh, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = _merge_heads(ctx)  # [B,1,D]
        out = layer.multihead_attn.out_proj(out)
        return out

    def forward_step(
        self,
        y_last: torch.Tensor,  # [B,1] int64
        cache: KVCache,
    ) -> torch.Tensor:
        """
        One-step decode. Returns logits for next token at current position.
        """
        B = y_last.size(0)
        t = cache.step
        # embed + positional (sin/cos buffer)
        x = self.emb(y_last)  # [B,1,D]
        x = x + self.pe[t:t+1].unsqueeze(0)  # [B,1,D]

        h = x
        for li, layer in enumerate(self.blocks):
            # norm_first=False path (post-norm), matching default TransformerDecoderLayer
            # 1) self-attn
            sa = self._self_attn_step(layer, h, cache, li)
            h = layer.norm1(h + layer.dropout1(sa))

            # 2) cross-attn
            ca = self._cross_attn_step(layer, h, cache, li)
            h = layer.norm2(h + layer.dropout2(ca))

            # 3) FFN
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm3(h + layer.dropout3(ff))

        cache.step += 1
        logits = self.lm(h).squeeze(1)  # [B,V]
        return logits

    @torch.no_grad()
    def greedy_generate_with_probs(
        self,
        mem: torch.Tensor,
        y0: torch.Tensor,               # [B,S0] prefix (PRG含む)
        max_new_tokens: int,
        *,
        eos_id: Optional[int] = None,
        pad_id: Optional[int] = None,
        return_with_prefix: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        KV-cache greedy decode + confidence outputs.

        Returns:
          y:        [B, S] tokens.
          pmax:     [B, S_gen] per generated token max-softmax probability
          margin:   [B, S_gen] per generated token (top1 - top2)
          log_prob: [B, S_gen] per generated token log P(chosen token)
        """
        self.eval()
        B = y0.size(0)
        device = y0.device

        if pad_id is None:
            # if not provided, use eos_id as "fill" after finished; if eos_id also None, fill with last token
            pad_id = eos_id if eos_id is not None else int(y0[:, -1].mode().values.item())

        # ---- cache length safety check ----
        # We will call forward_step for:
        #   (S0-1) priming steps + max_new_tokens generation steps
        need_steps = max(0, y0.size(1) - 1) + int(max_new_tokens)
        if need_steps > self.max_len:
            raise RuntimeError(
                f"max_len_cache too small: need_steps={need_steps} > max_len_cache={self.max_len}. "
                f"Use max_len_cache >= (S0-1)+max_new_tokens."
            )

        cache = self.init_cache(mem)

        # ---- prime cache with prefix tokens except last ----
        if y0.size(1) > 1:
            for t in range(y0.size(1) - 1):
                _ = self.forward_step(y0[:, t:t+1], cache)

        # ---- start from last prefix token ----
        cur = y0[:, -1:]  # [B,1]
        finished = torch.zeros((B,), dtype=torch.bool, device=device)

        # Collect output tokens
        out_tokens: List[torch.Tensor] = [y0] if return_with_prefix else []

        # Collect confidences for generated tokens only
        pmax_list: List[torch.Tensor] = []
        margin_list: List[torch.Tensor] = []
        logprob_list: List[torch.Tensor] = []

        for _ in range(int(max_new_tokens)):
            logits = self.forward_step(cur, cache)               # [B,V]
            probs = F.softmax(logits, dim=-1)                    # [B,V]
            log_p = F.log_softmax(logits, dim=-1)                # [B,V]
            top2 = torch.topk(probs, k=2, dim=-1).values         # [B,2]
            p1 = top2[:, 0]                                      # [B]
            p2 = top2[:, 1]                                      # [B]
            m = p1 - p2

            nxt = torch.argmax(logits, dim=-1, keepdim=True)     # [B,1]
            chosen_logp = log_p.gather(1, nxt).squeeze(1)         # [B]

            if eos_id is not None:
                nxt = torch.where(finished[:, None], torch.full_like(nxt, int(pad_id)), nxt)
                newly_finished = (nxt.squeeze(1) == int(eos_id)) & (~finished)
                finished = finished | newly_finished

            out_tokens.append(nxt)

            pmax_list.append(torch.where(finished, torch.zeros_like(p1), p1))
            margin_list.append(torch.where(finished, torch.zeros_like(m), m))
            logprob_list.append(torch.where(finished, torch.zeros_like(chosen_logp), chosen_logp))

            cur = nxt

            if eos_id is not None and bool(finished.all()):
                break

        y = torch.cat(out_tokens, dim=1) if out_tokens else torch.empty((B, 0), dtype=torch.long, device=device)

        if pmax_list:
            pmax = torch.stack(pmax_list, dim=1)       # [B, S_gen]
            margin = torch.stack(margin_list, dim=1)
            log_prob = torch.stack(logprob_list, dim=1) # [B, S_gen]
        else:
            pmax = torch.empty((B, 0), dtype=torch.float32, device=device)
            margin = torch.empty((B, 0), dtype=torch.float32, device=device)
            log_prob = torch.empty((B, 0), dtype=torch.float32, device=device)

        return y, pmax, margin, log_prob


# ----------------------------
# Replace your pseudo_label_with_kvcache with this
# ----------------------------
@torch.no_grad()
def pseudo_label_with_kvcache(
    teacher: nn.Module,           # MT3Mini (EMA teacher)
    fast_dec: FastDecoderKV,      # create once and reuse
    mel: torch.Tensor,            # [B,T,F]
    *,
    program_id: int,
    vocab,
    max_new_tokens: int,
    return_with_prefix: bool = False,  # False => PRGを除いた生成列を返す（推奨）
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates pseudo labels using EMA teacher encoder + KV-cache decoder reusing teacher.dec weights.

    Returns:
      out:      [B,S] token ids (PRG prefix excluded if return_with_prefix=False; else included)
      pmax:     [B,S_gen] per-token max softmax probability
      margin:   [B,S_gen] per-token (top1 - top2)
      log_prob: [B,S_gen] per-token log P(chosen token)
    """
    teacher.eval()
    fast_dec.eval()

    mem = teacher.enc(mel)

    prg_id = int(vocab.instrument_type[f"PRG_{int(program_id)}"])
    y0 = torch.full((mel.size(0), 1), prg_id, dtype=torch.long, device=mel.device)

    y, pmax, margin, log_prob = fast_dec.greedy_generate_with_probs(
        mem,
        y0=y0,
        max_new_tokens=max_new_tokens,
        eos_id=int(vocab.eos),
        pad_id=int(vocab.pad),
        return_with_prefix=True,
    )

    if return_with_prefix:
        return y, pmax, margin, log_prob
    else:
        return y[:, 1:], pmax, margin, log_prob