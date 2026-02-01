# my_mt3/infer.py
from __future__ import annotations

import numpy as np
import torch
import pretty_midi

from .tokenizer import VOCAB


@torch.no_grad()
def greedy_decode(model, mel, *, max_len: int = 1024, device: str = "cuda", program_id=0):
    """
    Args:
      model: MT3Mini
      mel: np.ndarray [T,F] or torch.Tensor [T,F] or [1,T,F]
      max_len: 最大生成長
      device: "cuda" / "cpu"
    Returns:
      token_ids: List[int]  （生成トークン列。BOSは含めない）
    """
    model.eval()

    # ---- mel -> torch [1,T,F] ----
    if torch.is_tensor(mel):
        mel_t = mel
    else:
        mel_t = torch.from_numpy(mel)

    mel_t = mel_t.to(device=device, dtype=torch.float32, non_blocking=True)
    if mel_t.ndim == 2:
        mel_t = mel_t.unsqueeze(0)
    elif mel_t.ndim != 3:
        raise ValueError(f"mel must be [T,F] or [1,T,F], got {tuple(mel_t.shape)}")

    # ---- encode ----
    mem = model.enc(mel_t)

    # ---- BOS: 最小の program token を使う（安定）----
    if hasattr(VOCAB, "program") and isinstance(VOCAB.program, dict) and len(VOCAB.program) > 0:
        bos_id = int(min(VOCAB.program.values()))
    else:
        bos_id = int(getattr(VOCAB, "bos", None) or getattr(VOCAB, "eos"))

    eos_id = int(VOCAB.eos)
    prg_key = f"PRG_{int(program_id)}"
    bos_id = VOCAB.program.get(prg_key, int(min(VOCAB.program.values())))
    y = torch.full((1, 1), int(bos_id), dtype=torch.long, device=device)

    # y: [1,1]
    y = torch.full((1, 1), bos_id, dtype=torch.long, device=device)

    out = []
    for _ in range(int(max_len)):
        # dec: [B,L,V] -> last step [B,V]
        logits = model.dec(y, mem)[:, -1, :]
        nxt = torch.argmax(logits, dim=-1)  # [1]
        tok = int(nxt.item())
        out.append(tok)

        if tok == eos_id:
            break

        # append: y becomes [1, L+1]
        y = torch.cat([y, nxt.unsqueeze(1)], dim=1)

    return out


def to_midi_from_tokens(token_ids, *, program_id: int = 0, step_ms: int = 10, velocity: int = 80):
    """
    Args:
      token_ids: List[int]
      program_id: 生成MIDIのprogram（単一トラックMVP）
      step_ms: TIM_x の1ステップ(ms)
      velocity: ノートvelocity
    Returns:
      pretty_midi.PrettyMIDI
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=int(program_id))

    cur_ms = 0
    onsets = {}  # pitch -> onset_ms

    eos_id = int(VOCAB.eos)

    for tid in token_ids:
        tid = int(tid)
        if tid == eos_id:
            break

        tok = VOCAB.itos[tid]

        if tok.startswith("TIM_"):
            # TIM_k は「絶対時刻 k*step_ms」扱い（あなたの実装準拠）
            k = int(tok.split("_")[1])
            cur_ms = k * step_ms

        elif tok.startswith("NON_"):
            p = int(tok.split("_")[1])
            onsets[p] = cur_ms

        elif tok.startswith("NOF_"):
            p = int(tok.split("_")[1])
            if p in onsets:
                on_ms = onsets.pop(p)
                if cur_ms > on_ms:  # 0長や逆転を防ぐ
                    inst.notes.append(
                        pretty_midi.Note(
                            velocity=int(velocity),
                            pitch=int(p),
                            start=on_ms / 1000.0,
                            end=cur_ms / 1000.0,
                        )
                    )
        else:
            # MVP: 未知トークンは無視
            continue

    pm.instruments.append(inst)
    return pm
