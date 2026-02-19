# my_mt3/infer.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import pretty_midi
from typing import Optional, Tuple


from my_mt3.tokenizer import Vocab, VOCAB as DEFAULT_VOCAB, VOCAB_PIANO


@torch.no_grad()
def greedy_decode(model, mel, *, max_len: int = 1024, device: str = "cuda", program_id=0, vocab: Vocab | None = None):
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

    # 語彙の解決
    vocab = DEFAULT_VOCAB if vocab is None else vocab

    # ---- BOS/EOS ----
    eos_id = int(vocab.eos)
    prg_key = f"PRG_{int(program_id)}"
    bos_id = int(vocab.program.get(prg_key, int(min(vocab.program.values()))))
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



@torch.no_grad()
def greedy_decode_with_probs(
    model,
    mel: torch.Tensor,              # [B,T,F]
    *,
    program_id: int,
    vocab,
    max_len: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      out:    [B,S] generated tokens (PRGは含まない)
      pmax:   [B,S]
      margin: [B,S]
    """

    device = mel.device
    B = mel.size(0)

    # --- PRG token が decoder の開始 ---
    prg_key = f"PRG_{int(program_id)}"
    prg_id = vocab.program[prg_key]
    y = torch.full((B, 1), prg_id, dtype=torch.long, device=device)

    mem = model.enc(mel)

    out = torch.full((B, max_len), vocab.pad, dtype=torch.long, device=device)
    pmax = torch.zeros((B, max_len), device=device)
    margin = torch.zeros((B, max_len), device=device)

    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(max_len):
        logits = model.dec(y, mem)[:, -1, :]   # [B,V]
        probs = F.softmax(logits, dim=-1)

        top2 = torch.topk(probs, 2, dim=-1).values
        p1 = top2[:, 0]
        p2 = top2[:, 1]
        m = p1 - p2

        nxt = torch.argmax(logits, dim=-1)     # [B]

        # write
        nxt_write = torch.where(finished, torch.full_like(nxt, vocab.pad), nxt)
        out[:, t] = nxt_write
        pmax[:, t] = torch.where(finished, torch.zeros_like(p1), p1)
        margin[:, t] = torch.where(finished, torch.zeros_like(m), m)

        # EOS処理
        just_finished = (nxt == vocab.eos) & (~finished)
        finished = finished | just_finished

        y = torch.cat([y, nxt_write.unsqueeze(1)], dim=1)

        if finished.all():
            break

    return out, pmax, margin




def to_midi_from_tokens(
    token_ids,
    *,
    program_id: int = 0,
    step_ms: int = 10,
    velocity: int = 80,
    default_dur_ms: int = 50,
    vocab: Vocab | None = None,
):
    """
    Args:
      token_ids: List[int]
      program_id: 生成MIDIのprogram（単一トラックMVP）
      step_ms: TIM_x の1ステップ(ms)
      velocity: ノートvelocity
    Returns:
      pretty_midi.PrettyMIDI
    """
    # 語彙の解決
    vocab = DEFAULT_VOCAB if vocab is None else vocab

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=int(program_id))

    cur_ms = 0
    # Note Off を使わない仕様のため、onsets は使わず On の時点で固定長のノートを生成
    # --- 旧実装（Note Off あり）の参考 ---
    # onsets = {}  # pitch -> onset_ms

    eos_id = int(vocab.eos)

    for tid in token_ids:
        tid = int(tid)
        if tid == eos_id:
            break

        tok = vocab.itos[tid]

        if tok.startswith("TIM_"):
            # TIM_k は「絶対時刻 k*step_ms」扱い（あなたの実装準拠）
            k = int(tok.split("_")[1])
            cur_ms = k * step_ms

        elif tok.startswith("NON_"):
            p = int(tok.split("_")[1])
            start_s = cur_ms / 1000.0
            end_s = (cur_ms + int(default_dur_ms)) / 1000.0
            # 0長を避けるため max を入れてもよいが、固定長 > 0 なので不要
            inst.notes.append(
                pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(p),
                    start=start_s,
                    end=end_s,
                )
            )
        # --- 旧実装（Note Off あり）の参考 ---
        # elif tok.startswith("NOF_"):
        #     p = int(tok.split("_")[1])
        #     if p in onsets:
        #         on_ms = onsets.pop(p)
        #         if cur_ms > on_ms:  # 0長や逆転を防ぐ
        #             inst.notes.append(
        #                 pretty_midi.Note(
        #                     velocity=int(velocity),
        #                     pitch=int(p),
        #                     start=on_ms / 1000.0,
        #                     end=cur_ms / 1000.0,
        #                 )
        #             )
        else:
            # MVP: 未知トークンは無視
            continue

    pm.instruments.append(inst)
    return pm


def to_midi_from_tokens_piano(
    token_ids,
    *,
    program_id: int = 0,
    step_ms: int = 10,
    velocity: int = 80,
    default_dur_ms: int = 50,
    vocab: Vocab | None = None,
):
    """
    Piano想定（Note Off 対応）のデコーダ。
    - TIM_k: 絶対時刻を k*step_ms に設定
    - NON_p: pitch=p のノート開始（既にONなら直前ノートを現在時刻までで確定）
    - NOF_p: pitch=p のノート終了（onsets から取り出してノート確定）
    - end_tie は情報を持たないマーカーのため復元では無視（学習のヒント用）
    """
    vocab = VOCAB_PIANO if vocab is None else vocab

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=int(program_id))

    cur_ms = 0
    onsets: dict[int, int] = {}  # pitch -> onset_ms

    eos_id = int(vocab.eos)

    for tid in token_ids:
        tid = int(tid)
        if tid == eos_id:
            break

        tok = vocab.itos[tid]

        if tok.startswith("TIM_"):
            k = int(tok.split("_")[1])
            cur_ms = k * int(step_ms)

        elif tok.startswith("NON_"):
            p = int(tok.split("_")[1])
            if p in onsets:
                on_ms = onsets.pop(p)
                off_ms = cur_ms
                if off_ms <= on_ms:
                    off_ms = on_ms + max(1, int(default_dur_ms))
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(velocity),
                        pitch=p,
                        start=on_ms / 1000.0,
                        end=off_ms / 1000.0,
                    )
                )
            # start new
            onsets[p] = cur_ms

        elif tok.startswith("NOF_"):
            p = int(tok.split("_")[1])
            if p in onsets:
                on_ms = onsets.pop(p)
                off_ms = cur_ms
                if off_ms <= on_ms:
                    off_ms = on_ms + max(1, int(default_dur_ms))
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(velocity),
                        pitch=p,
                        start=on_ms / 1000.0,
                        end=off_ms / 1000.0,
                    )
                )
            # 未対応の NOF は無視
        else:
            # それ以外（end_tie等）は復元情報を持たないため無視
            continue

    # 残っているノートを適当に閉じる（安全策）
    if onsets:
        tail_ms = cur_ms + int(default_dur_ms)
        for p, on_ms in onsets.items():
            off_ms = tail_ms if tail_ms > on_ms else on_ms + max(1, int(default_dur_ms))
            inst.notes.append(
                pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=p,
                    start=on_ms / 1000.0,
                    end=off_ms / 1000.0,
                )
            )

    pm.instruments.append(inst)
    return pm
