# my_mt3/infer.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import pretty_midi
from typing import Optional, Tuple, List


from my_mt3.tokenizer import Vocab, VOCAB as DEFAULT_VOCAB, VOCAB_PIANO
from my_mt3.decode_kv import FastDecoderKV


@torch.no_grad()
def greedy_decode(model, mel, *, max_len: int = 1024, device: str = "cuda", program_id=0, vocab: Vocab | None = None):
    """
    KV-cache accelerated greedy decode.

    Args:
      model: MT3Mini
      mel: np.ndarray [T,F] or torch.Tensor [T,F] or [1,T,F]
      max_len: 最大生成長
      device: "cuda" / "cpu"
    Returns:
      token_ids: List[int]  （生成トークン列。BOSは含めない）
    """
    model.eval()

    if torch.is_tensor(mel):
        mel_t = mel
    else:
        mel_t = torch.from_numpy(mel)

    mel_t = mel_t.to(device=device, dtype=torch.float32, non_blocking=True)
    if mel_t.ndim == 2:
        mel_t = mel_t.unsqueeze(0)
    elif mel_t.ndim != 3:
        raise ValueError(f"mel must be [T,F] or [1,T,F], got {tuple(mel_t.shape)}")

    mem = model.enc(mel_t)

    vocab = DEFAULT_VOCAB if vocab is None else vocab

    eos_id = int(vocab.eos)
    prg_key = f"PRG_{int(program_id)}"
    bos_id = int(vocab.instrument_type.get(prg_key, int(min(vocab.instrument_type.values()))))

    fast_dec = FastDecoderKV(model.dec, max_len=max_len + 1)
    cache = fast_dec.init_cache(mem)

    cur = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    out = []
    for _ in range(int(max_len)):
        logits = fast_dec.forward_step(cur, cache)   # [1, V]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
        tok = int(nxt.item())
        out.append(tok)

        if tok == eos_id:
            break
        cur = nxt

    return out



@torch.no_grad()
def greedy_decode_batch(
    model, mels_bt: torch.Tensor, *, max_len: int, device: str, program_id: int, vocab: Vocab,
) -> List[List[int]]:
    """
    KV-cache accelerated batch greedy decode.
    Args:
      mels_bt: [B, T, F]
    Returns:
      List[List[int]] — token ids per sample (BOS excluded)
    """
    B = mels_bt.size(0)
    mem = model.enc(mels_bt)
    prg_key = f"PRG_{int(program_id)}"
    bos_id = int(vocab.instrument_type.get(prg_key, int(min(vocab.instrument_type.values()))))
    eos_id = int(vocab.eos)

    fast_dec = FastDecoderKV(model.dec, max_len=max_len + 1)
    cache = fast_dec.init_cache(mem)

    cur = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros((B,), dtype=torch.bool, device=device)
    outputs: List[List[int]] = [[] for _ in range(B)]

    for _ in range(int(max_len)):
        logits = fast_dec.forward_step(cur, cache)
        nxt = torch.argmax(logits, dim=-1)
        for b in range(B):
            if not finished[b]:
                tok = int(nxt[b].item())
                outputs[b].append(tok)
                if tok == eos_id:
                    finished[b] = True
        if bool(torch.all(finished).item()):
            break
        cur = nxt.unsqueeze(1)

    return outputs


@torch.no_grad()
def greedy_decode_batch_with_logprobs(
    model, mels_bt: torch.Tensor, *, max_len: int, device: str, program_id: int, vocab: Vocab,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    KV-cache batch greedy decode that also returns per-token log probabilities.
    Returns:
      outputs:   List[List[int]]   — token ids per sample
      logprobs:  List[List[float]] — log P(token|context) per sample
    """
    B = mels_bt.size(0)
    mem = model.enc(mels_bt)
    prg_key = f"PRG_{int(program_id)}"
    bos_id = int(vocab.instrument_type.get(prg_key, int(min(vocab.instrument_type.values()))))
    eos_id = int(vocab.eos)

    fast_dec = FastDecoderKV(model.dec, max_len=max_len + 1)
    cache = fast_dec.init_cache(mem)

    cur = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros((B,), dtype=torch.bool, device=device)
    outputs: List[List[int]] = [[] for _ in range(B)]
    logprobs: List[List[float]] = [[] for _ in range(B)]

    for _ in range(int(max_len)):
        logits = fast_dec.forward_step(cur, cache)
        log_p = F.log_softmax(logits, dim=-1)
        nxt = torch.argmax(logits, dim=-1)
        for b in range(B):
            if not finished[b]:
                tok = int(nxt[b].item())
                outputs[b].append(tok)
                logprobs[b].append(float(log_p[b, tok].item()))
                if tok == eos_id:
                    finished[b] = True
        if bool(torch.all(finished).item()):
            break
        cur = nxt.unsqueeze(1)

    return outputs, logprobs


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
    prg_id = vocab.instrument_type[prg_key]
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


class ChunkDecodeResult:
    """to_midi_from_tokens_piano の戻り値。

    Attributes:
        pm: チャンク内で完結したノートを含む PrettyMIDI
        tie_pitches: tie section で宣言された pitch の集合
        tie_offsets_ms: tie pitch のうちチャンク内で offset が確定したもの {pitch: offset_ms}
        open_onsets_ms: チャンク末尾で offset 未確定のノート {pitch: onset_ms}
    """
    __slots__ = ("pm", "tie_pitches", "tie_offsets_ms", "open_onsets_ms")

    def __init__(
        self,
        pm: pretty_midi.PrettyMIDI,
        tie_pitches: set[int],
        tie_offsets_ms: dict[int, int],
        open_onsets_ms: dict[int, int],
    ):
        self.pm = pm
        self.tie_pitches = tie_pitches
        self.tie_offsets_ms = tie_offsets_ms
        self.open_onsets_ms = open_onsets_ms


def to_midi_from_tokens_piano(
    token_ids,
    *,
    program_id: int = 0,
    step_ms: int = 10,
    velocity: int = 80,
    default_dur_ms: int = 50,
    vocab: Vocab | None = None,
) -> ChunkDecodeResult:
    """
    Piano想定（Note Off 対応）のデコーダ。

    tie section (<end_tie> NON_p1 NON_p2 ... の後 TIM_ で終了) を解析し、
    前チャンクから持ち越されたピッチを tie_pitches として返す。
    tie pitch のノートは pm には含めず tie_offsets_ms / open_onsets_ms で返す。
    チャンク末で offset 未確定のノートは force-close せず open_onsets_ms で返す。
    """
    vocab = VOCAB_PIANO if vocab is None else vocab

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=int(program_id))

    cur_ms = 0
    onsets: dict[int, int] = {}  # pitch -> onset_ms
    tie_pitches: set[int] = set()
    tie_offsets_ms: dict[int, int] = {}
    in_tie_section = False

    eos_id = int(vocab.eos)
    end_tie_id = int(vocab.end_tie) if vocab.end_tie is not None else -1

    for tid in token_ids:
        tid = int(tid)
        if tid == eos_id:
            break

        tok = vocab.itos[tid]

        if tid == end_tie_id:
            in_tie_section = True
            continue

        if tok.startswith("TIM_"):
            in_tie_section = False
            k = int(tok.split("_")[1])
            cur_ms = k * int(step_ms)
            continue

        if tok.startswith("NON_"):
            p = int(tok.split("_")[1])
            if in_tie_section:
                tie_pitches.add(p)
                onsets[p] = 0
                continue
            # 既存 onset を閉じる
            if p in onsets:
                on_ms = onsets.pop(p)
                off_ms = cur_ms
                if off_ms <= on_ms:
                    off_ms = on_ms + max(1, int(default_dur_ms))
                if p in tie_pitches and on_ms == 0:
                    tie_offsets_ms[p] = off_ms
                else:
                    inst.notes.append(pretty_midi.Note(
                        velocity=int(velocity), pitch=p,
                        start=on_ms / 1000.0, end=off_ms / 1000.0,
                    ))
            onsets[p] = cur_ms
            continue

        if tok.startswith("NOF_"):
            p = int(tok.split("_")[1])
            if p in onsets:
                on_ms = onsets.pop(p)
                off_ms = cur_ms
                if off_ms <= on_ms:
                    off_ms = on_ms + max(1, int(default_dur_ms))
                if p in tie_pitches and on_ms == 0:
                    tie_offsets_ms[p] = off_ms
                else:
                    inst.notes.append(pretty_midi.Note(
                        velocity=int(velocity), pitch=p,
                        start=on_ms / 1000.0, end=off_ms / 1000.0,
                    ))

    open_onsets_ms = dict(onsets)

    pm.instruments.append(inst)
    return ChunkDecodeResult(
        pm=pm,
        tie_pitches=tie_pitches,
        tie_offsets_ms=tie_offsets_ms,
        open_onsets_ms=open_onsets_ms,
    )
