# my_mt3/train_DA.py


import math
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_mt3.model import MT3Mini, EMATeacher
from my_mt3.decode_kv import FastDecoderKV, pseudo_label_with_kvcache
from my_mt3.tokenizer import Vocab, INPUT_FRAMES
from my_mt3.dataset import AMTDataset
from my_mt3.dataset_unlabeled import AMTRealDataset
from my_mt3.discriminator import Discriminator
from my_mt3.audio import ensure_wave_cache, DEFAULT_SR
import os
import re
import itertools
import json

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Function
from my_mt3.train import _maybe_cache_pairs_map, make_collate
from my_mt3.augment import AugmentConfig, augment_spectrogram
from my_mt3.analysis_attribution import apply_source_mask_band

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

@dataclass
class NoteSpan:
    # token indices in the *target sequence* (y_tg side) that correspond to this note
    tok_ids: List[int]   # e.g., [k_on, k_off, ...]  (indices in y sequence excluding BOS or aligned as you decide)

def decode_notes_to_spans(token_ids: List[int], vocab) -> List[NoteSpan]:
    """
    token_ids: 生成されたトークン列。
      - teacherのgreedy decodeが BOS を先頭に付けるなら [BOS, ...]
      - 付けないなら [...]. どちらでも動くようにしてある。
    vocab: あなたの Vocab（program/time/note_on/note_off/eos/bos を持つ）

    返り値:
      List[NoteSpan] where NoteSpan.tok_ids are indices in y_tg (= token_ids[(start+1):])
    """

    # ---- BOSの有無を吸収 ----
    start = 1 if (hasattr(vocab, "bos") and len(token_ids) > 0 and token_ids[0] == vocab.bos) else 0

    # y_tg は token_ids[(start+1):] に対応するので、
    # token_ids[i] の y_tg index は i - (start+1)
    def to_tg_index(i: int) -> int:
        return i - (start + 1)

    # ---- reverse maps: id -> pitch ----
    # vocab.note_on: Dict[pitch -> id]
    id2on: Dict[int, int] = {tid: p for p, tid in vocab.note_on.items()}
    has_off = getattr(vocab, "note_off", None) is not None
    id2off: Dict[int, int] = {}
    if has_off:
        id2off = {tid: p for p, tid in vocab.note_off.items()}

    eos_id = getattr(vocab, "eos", None)

    spans: List[NoteSpan] = []

    if not has_off:
        # ---- NoteOff無し: NOTE_ON 1個 = 1 note ----
        for i in range(start, len(token_ids)):
            tid = token_ids[i]
            if eos_id is not None and tid == eos_id:
                break
            if tid in id2on:
                tg = to_tg_index(i)
                if tg >= 0:
                    spans.append(NoteSpan(tok_ids=[tg]))
        return spans

    # ---- NoteOffあり: on/off をペアリングし、[on_idx, off_idx] を span とする ----
    open_on: Dict[int, int] = {}  # pitch -> tg_index_of_on
    for i in range(start, len(token_ids)):
        tid = token_ids[i]
        if eos_id is not None and tid == eos_id:
            break

        if tid in id2on:
            p = id2on[tid]
            tg = to_tg_index(i)
            if tg >= 0:
                open_on[p] = tg

        elif tid in id2off:
            p = id2off[tid]
            tg = to_tg_index(i)
            if tg >= 0 and p in open_on:
                spans.append(NoteSpan(tok_ids=[open_on[p], tg]))
                del open_on[p]

    # 未クローズのノートは NOTE_ON 単体として扱う（安全側）
    for p, on_tg in open_on.items():
        spans.append(NoteSpan(tok_ids=[on_tg]))

    return spans



def build_note_confidences(note_spans: List[NoteSpan], log_prob_1d: torch.Tensor):
    """
    log_prob_1d: [S-1]  per-token log P(token) (BOSの次のステップから)
    return:
      scores: np array [N_notes]  — ノートごとの平均 log prob (= log P / T)
    """
    scores = []
    for ns in note_spans:
        idx = [i for i in ns.tok_ids if 0 <= i < log_prob_1d.numel()]
        if len(idx) == 0:
            scores.append(-float("inf"))
            continue
        scores.append(float(log_prob_1d[idx].mean().item()))
    return np.array(scores)


@torch.no_grad()
def _teacher_forced_token_logp(
    *,
    model,
    mel_1: torch.Tensor,   # [1,T,F]
    y_in_1: torch.Tensor,  # [1,S]
    y_tg_1: torch.Tensor,  # [1,S]
) -> torch.Tensor:
    """Per-position log-probability for a fixed target sequence."""
    mem = model.enc(mel_1)
    logits = model.dec(y_in_1, mem)[0]  # [S,V]
    logp = torch.log_softmax(logits, dim=-1)
    tgt = y_tg_1[0].long()
    return logp.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)  # [S]


def _token_time_frame_map(
    token_ids: List[int],
    *,
    vocab: Vocab,
    sr: int,
    hop: int,
    step_ms: int,
) -> List[Optional[int]]:
    id2time: Dict[int, int] = {tid: t for t, tid in vocab.time.items()}
    cur_frame: Optional[int] = None
    out: List[Optional[int]] = []
    for tid in token_ids:
        t = int(tid)
        if t in id2time:
            t_sec = (float(id2time[t]) * float(step_ms)) / 1000.0
            cur_frame = int(round(t_sec * float(sr) / float(hop)))
        out.append(cur_frame)
    return out


def _build_note_mask_effect_confidences(
    *,
    spans: List[NoteSpan],
    token_ids: List[int],
    base_logp_1d: torch.Tensor,
    masked_logp_by_frame: Dict[int, torch.Tensor],
    token_frame_map: List[Optional[int]],
    note_on_ids: set[int],
    use_log_of_abs: bool,
    eps: float = 1e-8,
) -> np.ndarray:
    scores: List[float] = []
    for ns in spans:
        onset_tok_idx: Optional[int] = None
        for t_idx in ns.tok_ids:
            if 0 <= int(t_idx) < len(token_ids) and int(token_ids[int(t_idx)]) in note_on_ids:
                onset_tok_idx = int(t_idx)
                break
        if onset_tok_idx is None:
            onset_tok_idx = int(ns.tok_ids[0]) if ns.tok_ids else -1
        if onset_tok_idx < 0 or onset_tok_idx >= int(base_logp_1d.numel()) or onset_tok_idx >= len(token_frame_map):
            scores.append(-float("inf"))
            continue
        fr = token_frame_map[onset_tok_idx]
        if fr is None or int(fr) not in masked_logp_by_frame:
            scores.append(-float("inf"))
            continue
        masked_lp = masked_logp_by_frame[int(fr)]
        if onset_tok_idx >= int(masked_lp.numel()):
            scores.append(-float("inf"))
            continue
        delta_abs = float(abs(float(masked_lp[onset_tok_idx].item()) - float(base_logp_1d[onset_tok_idx].item())))
        if use_log_of_abs:
            scores.append(float(np.log(delta_abs + float(eps))))
        else:
            scores.append(float(delta_abs))
    return np.asarray(scores, dtype=float)


def _set_model_trainable_only_unsup_cross_attn(model: nn.Module) -> Dict[str, bool]:
    """
    Temporarily keep grads only for decoder cross-attention weights.
    Returns previous requires_grad state map for restoration.
    """
    prev: Dict[str, bool] = {}
    for name, p in model.named_parameters():
        prev[name] = bool(p.requires_grad)
        keep = (".dec.blocks." in name) and (".multihead_attn." in name)
        p.requires_grad_(keep)
    return prev


def _restore_model_requires_grad(model: nn.Module, prev: Dict[str, bool]) -> None:
    for name, p in model.named_parameters():
        if name in prev:
            p.requires_grad_(bool(prev[name]))

def make_pseudo_token_mask_from_notes(
    note_spans: List[NoteSpan],
    scores: np.ndarray,
    seq_len_no_bos: int,
    *,
    top_frac: float = 0.2,
    bot_frac: float = 0.2,
):
    """
    scores: ノートごとの信頼度 (mean log prob)。高いほど信頼度が高い。
    returns mask: torch.bool [seq_len_no_bos], True=compute CE
    """
    n = len(note_spans)
    if n == 0:
        return torch.zeros((seq_len_no_bos,), dtype=torch.bool)

    order = np.argsort(scores)  # ascending (worst first)

    k_top = max(1, int(round(n * top_frac)))
    k_bot = max(1, int(round(n * bot_frac)))

    idx_bot = set(order[:k_bot].tolist())
    idx_top = set(order[-k_top:].tolist())
    keep_notes = idx_top | idx_bot

    mask = torch.zeros((seq_len_no_bos,), dtype=torch.bool)
    for i, ns in enumerate(note_spans):
        if i not in keep_notes:
            continue
        for t in ns.tok_ids:
            if 0 <= t < seq_len_no_bos:
                mask[t] = True
    return mask


def apply_mask_to_targets(y_tg: torch.Tensor, token_mask: torch.Tensor, ignore_index: int):
    """
    y_tg: [B, S-1]
    token_mask: [B, S-1] bool, True=keep
    """
    y = y_tg.clone()
    y[~token_mask] = ignore_index
    return y


def _collect_timewise_onset_sequences_from_target(
    *,
    target_tokens: List[int],
    vocab: Vocab,
    max_groups: int = 0,
    min_onsets_per_group: int = 1,
) -> List[List[int]]:
    """Build [TIM_t, NOTE_ON...] sequences from one target row."""
    time_ids = set(vocab.time.values())
    note_on_ids = set(vocab.note_on.values())
    pad_id = int(vocab.pad)

    groups: List[List[int]] = []
    cur_time: Optional[int] = None
    cur_onsets: List[int] = []

    def _flush() -> None:
        nonlocal cur_time, cur_onsets
        if cur_time is None:
            return
        if len(cur_onsets) >= int(min_onsets_per_group):
            groups.append([int(cur_time), *[int(t) for t in cur_onsets]])
        cur_time = None
        cur_onsets = []

    for tok in target_tokens:
        tid = int(tok)
        if tid == pad_id:
            continue
        if tid in time_ids:
            _flush()
            cur_time = tid
            continue
        if (cur_time is not None) and (tid in note_on_ids):
            cur_onsets.append(tid)

    _flush()

    if int(max_groups) > 0:
        groups = groups[: int(max_groups)]
    return groups


def _build_timewise_onset_tf_batch(
    *,
    y_tg: torch.Tensor,
    vocab: Vocab,
    max_groups_per_sample: int = 0,
    min_onsets_per_group: int = 1,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Returns:
      sample_indices: [N_aux]  index of source sample in current batch
      y_in_aux:       [N_aux, S_aux]
      y_tg_aux:       [N_aux, S_aux]
    """
    if y_tg.ndim != 2:
        return None

    pad_id = int(vocab.pad)
    seqs: List[List[int]] = []
    src_idx: List[int] = []

    for b_idx, row in enumerate(y_tg.detach().cpu().tolist()):
        groups = _collect_timewise_onset_sequences_from_target(
            target_tokens=[int(t) for t in row],
            vocab=vocab,
            max_groups=int(max_groups_per_sample),
            min_onsets_per_group=int(min_onsets_per_group),
        )
        for g in groups:
            if len(g) >= 3:
                seqs.append(g)
                src_idx.append(int(b_idx))

    if len(seqs) == 0:
        return None

    max_len = max(len(s) - 1 for s in seqs)
    if max_len <= 0:
        return None

    dev = y_tg.device
    y_in_aux = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=dev)
    y_tg_aux = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=dev)

    for i, s in enumerate(seqs):
        yin = s[:-1]
        ytg = s[1:]
        L = min(max_len, len(yin))
        if L > 0:
            y_in_aux[i, :L] = torch.tensor(yin[:L], dtype=torch.long, device=dev)
            y_tg_aux[i, :L] = torch.tensor(ytg[:L], dtype=torch.long, device=dev)

    sample_indices = torch.tensor(src_idx, dtype=torch.long, device=dev)
    return sample_indices, y_in_aux, y_tg_aux


@dataclass
class PseudoNoteEvent:
    pitch: int
    onset: float
    offset: float
    on_tok_idx: int
    off_tok_idx: int


def _decode_pseudo_notes_with_token_indices(
    token_ids: List[int],
    vocab: Vocab,
    *,
    step_ms: int = 10,
    pad_id: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> List[PseudoNoteEvent]:
    id2time: Dict[int, int] = {tid: t for t, tid in vocab.time.items()}
    id2on: Dict[int, int] = {tid: p for p, tid in vocab.note_on.items()}
    id2off: Dict[int, int] = {}
    if getattr(vocab, "note_off", None) is not None:
        id2off = {tid: p for p, tid in vocab.note_off.items()}

    t_sec = 0.0
    step_sec = float(step_ms) / 1000.0
    open_on: Dict[int, List[Tuple[float, int]]] = {}
    events: List[PseudoNoteEvent] = []

    for i, tid in enumerate(token_ids):
        if eos_id is not None and tid == eos_id:
            break
        if pad_id is not None and tid == pad_id:
            break

        if tid in id2time:
            t_sec = id2time[tid] * step_sec
            continue

        if tid in id2on:
            p = id2on[tid]
            open_on.setdefault(p, []).append((t_sec, i))
            continue

        if tid in id2off:
            p = id2off[tid]
            if p not in open_on or not open_on[p]:
                continue
            on_t, on_idx = open_on[p].pop(0)
            off_t = max(t_sec, on_t + 1e-3)
            events.append(
                PseudoNoteEvent(
                    pitch=int(p),
                    onset=float(on_t),
                    offset=float(off_t),
                    on_tok_idx=int(on_idx),
                    off_tok_idx=int(i),
                )
            )

    events.sort(key=lambda x: x.onset)
    return events


def _match_notes_mireval_style(
    est_notes: List[PseudoNoteEvent],
    ref_int: np.ndarray,
    ref_pitch: np.ndarray,
    *,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    offset_min_tolerance: float = 0.05,
) -> List[int]:
    if len(est_notes) == 0 or len(ref_int) == 0:
        return []

    ref_by_pitch: Dict[int, List[int]] = {}
    for i, p in enumerate(ref_pitch.tolist()):
        ref_by_pitch.setdefault(int(p), []).append(i)
    for p in ref_by_pitch.keys():
        ref_by_pitch[p].sort(key=lambda idx: float(ref_int[idx, 0]))

    used_ref: set[int] = set()
    matched_est_idx: List[int] = []

    for i_est, e in enumerate(est_notes):
        cand = ref_by_pitch.get(int(e.pitch), [])
        if not cand:
            continue
        best_idx = -1
        best_score = float("inf")
        for r_idx in cand:
            if r_idx in used_ref:
                continue
            r_on = float(ref_int[r_idx, 0])
            r_off = float(ref_int[r_idx, 1])
            if abs(e.onset - r_on) > onset_tolerance:
                continue
            off_tol = max(offset_min_tolerance, offset_ratio * max(r_off - r_on, 1e-6))
            if abs(e.offset - r_off) > off_tol:
                continue
            score = abs(e.onset - r_on) + abs(e.offset - r_off)
            if score < best_score:
                best_score = score
                best_idx = r_idx
        if best_idx >= 0:
            used_ref.add(best_idx)
            matched_est_idx.append(i_est)

    return matched_est_idx


def oracle_note_token_mask(
    out: torch.Tensor,
    real_idxs: torch.Tensor,
    real_starts: torch.Tensor,
    *,
    oracle_midi_cache: Dict[int, object],
    vocab: Vocab,
    sr: int,
    need_samples: int,
    step_ms: int = 10,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    offset_min_tolerance: float = 0.05,
    device: torch.device,
) -> torch.Tensor:
    from my_mt3.eval import extract_notes_in_range

    B, S = out.shape
    mask = torch.zeros((B, S), dtype=torch.bool, device=device)
    pad_id = int(vocab.pad)
    eos_id = int(vocab.eos)
    window_sec = need_samples / float(sr)

    for b in range(B):
        idx = int(real_idxs[b].item())
        ss = int(real_starts[b].item())
        ref_pm = oracle_midi_cache.get(idx)
        if ref_pm is None:
            continue
        t0 = ss / float(sr)
        t1 = t0 + window_sec
        ref_int, ref_pitch, _ref_vel = extract_notes_in_range(ref_pm, t0, t1, program=0)
        if len(ref_int) == 0:
            continue

        est_notes = _decode_pseudo_notes_with_token_indices(
            out[b].tolist(),
            vocab,
            step_ms=step_ms,
            pad_id=pad_id,
            eos_id=eos_id,
        )
        if not est_notes:
            continue

        matched = _match_notes_mireval_style(
            est_notes,
            ref_int,
            ref_pitch,
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
        )
        for est_i in matched:
            ev = est_notes[est_i]
            if 0 <= ev.on_tok_idx < S:
                mask[b, ev.on_tok_idx] = True
            if 0 <= ev.off_tok_idx < S:
                mask[b, ev.off_tok_idx] = True

    return mask


def pseudo_chunk_filter(
    out: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    pseudo_threshold: float = -0.5,
    pseudo_topn: int = 0,
) -> torch.Tensor:
    """
    Compute per-sample chunk confidence and return a boolean mask (B,).

    Filtering is applied in two stages (both can be active simultaneously):
      1. Threshold gate:  keep samples with conf >= pseudo_threshold
      2. Top-N selection: from the survivors, keep at most the N most confident

    When pseudo_topn <= 0 the top-N stage is skipped (threshold only).
    """
    B = out.size(0)
    confs = torch.full((B,), -float("inf"), device=device)
    for b in range(B):
        valid = (out[b] != pad_id) & (out[b] != eos_id)
        n_valid = valid.sum().item()
        if n_valid == 0:
            continue
        confs[b] = log_prob[b][valid[:log_prob.size(1)]].sum().item() / n_valid

    mask = confs >= pseudo_threshold

    if pseudo_topn > 0 and mask.any():
        confs_filtered = confs.clone()
        confs_filtered[~mask] = -float("inf")
        k = min(pseudo_topn, int(mask.sum().item()))
        _, top_idx = confs_filtered.topk(k)
        mask.fill_(False)
        mask[top_idx] = True

    return mask


def _canonicalize_pseudo_sequence_tokens(
    tokens: List[int],
    logps: List[float],
    *,
    vocab: Vocab,
    pad_id: int,
    eos_id: int,
) -> tuple[List[int], List[float]]:
    """
    Canonicalize pseudo token order within each TIME group:
      1) same TIME: note_on pitch low->high
      2) then note_off pitch low->high
      3) remove duplicate token ids in the same TIME group (keep highest logp)
    """
    id2time: Dict[int, int] = {tid: t for t, tid in vocab.time.items()}
    id2on: Dict[int, int] = {tid: p for p, tid in vocab.note_on.items()}
    id2off: Dict[int, int] = {}
    if getattr(vocab, "note_off", None) is not None:
        id2off = {tid: p for p, tid in vocab.note_off.items()}

    out_toks: List[int] = []
    out_lps: List[float] = []
    group_items: List[Tuple[int, float, int, int, int]] = []
    # item: (tid, lp, type_ord, pitch, orig_idx)
    in_time_group = False

    def _flush_group() -> None:
        nonlocal group_items, out_toks, out_lps
        if not group_items:
            return
        # dedup by token id (keep max logp)
        best_by_tid: Dict[int, Tuple[int, float, int, int, int]] = {}
        for item in group_items:
            tid, lp, type_ord, pitch, orig_idx = item
            prev = best_by_tid.get(int(tid))
            if prev is None or float(lp) > float(prev[1]):
                best_by_tid[int(tid)] = item
        items = list(best_by_tid.values())
        items.sort(key=lambda x: (int(x[2]), int(x[3]), int(x[4])))
        for tid, lp, *_ in items:
            out_toks.append(int(tid))
            out_lps.append(float(lp))
        group_items = []

    for i, tid in enumerate(tokens):
        tid = int(tid)
        lp = float(logps[i]) if i < len(logps) else 0.0
        if tid == pad_id:
            break
        if tid == eos_id:
            _flush_group()
            out_toks.append(int(tid))
            out_lps.append(float(lp))
            break
        if tid in id2time:
            _flush_group()
            out_toks.append(int(tid))
            out_lps.append(float(lp))
            in_time_group = True
            continue

        if in_time_group and (tid in id2on):
            group_items.append((int(tid), float(lp), 0, int(id2on[tid]), int(i)))
            continue
        if in_time_group and (tid in id2off):
            group_items.append((int(tid), float(lp), 1, int(id2off[tid]), int(i)))
            continue

        # non note token: keep relative order after on/off in this TIME group
        if in_time_group:
            group_items.append((int(tid), float(lp), 2, 10**9, int(i)))
        else:
            out_toks.append(int(tid))
            out_lps.append(float(lp))

    _flush_group()
    return out_toks, out_lps


def canonicalize_pseudo_batch_order(
    out: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    vocab: Vocab,
    pad_id: int,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply canonical ordering to each pseudo-labeled sequence in a batch."""
    B, S = out.shape
    out_new = torch.full_like(out, pad_id)
    lp_new = torch.zeros_like(log_prob)

    for b in range(B):
        toks = [int(t) for t in out[b].tolist()]
        lps = [float(x) for x in log_prob[b].tolist()]
        toks_c, lps_c = _canonicalize_pseudo_sequence_tokens(
            toks, lps, vocab=vocab, pad_id=pad_id, eos_id=eos_id
        )
        n = min(S, len(toks_c))
        if n > 0:
            out_new[b, :n] = torch.tensor(toks_c[:n], dtype=out.dtype, device=out.device)
            lp_new[b, :n] = torch.tensor(lps_c[:n], dtype=log_prob.dtype, device=log_prob.device)

    return out_new, lp_new


def oracle_chunk_filter(
    out: torch.Tensor,
    real_idxs: torch.Tensor,
    real_starts: torch.Tensor,
    *,
    oracle_midi_cache: Dict[int, object],
    vocab: Vocab,
    sr: int,
    need_samples: int,
    step_ms: int = 10,
    oracle_metric: str = "note_f",
    oracle_threshold: float = 0.5,
    device: torch.device,
) -> torch.Tensor:
    """Oracle filter: 疑似ラベルを正解 MIDI と照合し、指定指標が閾値以上のチャンクを選択。"""
    import pretty_midi
    from my_mt3.infer import to_midi_from_tokens_piano
    from my_mt3.eval import extract_notes_in_range, evaluate_notes_direct

    B = out.size(0)
    mask = torch.zeros(B, dtype=torch.bool, device=device)
    pad_id = int(vocab.pad)
    eos_id = int(vocab.eos)
    window_sec = need_samples / float(sr)

    for b in range(B):
        tokens_b = out[b].tolist()
        # skip empty
        n_valid = sum(1 for t in tokens_b if t != pad_id and t != eos_id)
        if n_valid == 0:
            continue

        idx = int(real_idxs[b].item())
        ss = int(real_starts[b].item())
        t0 = ss / float(sr)
        t1 = t0 + window_sec

        ref_pm = oracle_midi_cache.get(idx)
        if ref_pm is None:
            continue

        # est: pseudo-label tokens → PrettyMIDI
        res = to_midi_from_tokens_piano(tokens_b, program_id=0, step_ms=step_ms, vocab=vocab)
        est_notes = []
        for inst in res.pm.instruments:
            for n in inst.notes:
                est_notes.append((n.start, n.end, n.pitch, n.velocity))
        if not est_notes:
            continue
        est_int = np.array([[s, e] for s, e, _, _ in est_notes], dtype=float)
        est_pitch = np.array([p for _, _, p, _ in est_notes], dtype=int)
        est_vel = np.array([v for _, _, _, v in est_notes], dtype=int)
        order = np.argsort(est_int[:, 0])
        est_int, est_pitch, est_vel = est_int[order], est_pitch[order], est_vel[order]

        ref_int, ref_pitch, ref_vel = extract_notes_in_range(ref_pm, t0, t1, program=0)
        if len(ref_int) == 0 and len(est_int) == 0:
            mask[b] = True
            continue
        if len(ref_int) == 0 or len(est_int) == 0:
            continue

        try:
            m = evaluate_notes_direct(ref_int, ref_pitch, ref_vel, est_int, est_pitch, est_vel)
        except Exception:
            continue

        if m.get(oracle_metric, 0.0) >= oracle_threshold:
            mask[b] = True

    return mask


def _save_pseudo_debug_sample(
    *,
    out_tokens: List[int],
    log_prob_row: torch.Tensor,
    selected_token_mask_row: torch.Tensor,
    chunk_selected: bool,
    save_root: str,
    sample_idx: int,
    epoch: int,
    batch_idx: int,
    in_batch_idx: int,
    vocab: Vocab,
    gt_intervals: np.ndarray,
    gt_pitches: np.ndarray,
    window_sec: float,
    step_ms: int = 10,
    chunk_keep_ratio_batch: float | None = None,
    token_keep_ratio_batch: float | None = None,
):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    os.makedirs(save_root, exist_ok=True)
    stem = f"sample_{sample_idx:04d}_ep{epoch:05d}_b{batch_idx:05d}_i{in_batch_idx:02d}"
    txt_path = os.path.join(save_root, f"{stem}.txt")
    png_path = os.path.join(save_root, f"{stem}.png")

    pad_id = int(vocab.pad)
    eos_id = int(vocab.eos)
    tokens_trim = []
    for t in out_tokens:
        tokens_trim.append(int(t))
        if int(t) == eos_id:
            break
    if len(tokens_trim) == 0:
        tokens_trim = [int(t) for t in out_tokens]

    lp_cpu = log_prob_row.detach().to("cpu")
    sel_mask = selected_token_mask_row.detach().to("cpu").bool()
    n_tok = min(len(tokens_trim), int(lp_cpu.numel()))
    valid_ids = torch.tensor(tokens_trim[:n_tok], dtype=torch.long)
    valid_mask = (valid_ids != pad_id) & (valid_ids != eos_id)
    chunk_conf = float(lp_cpu[:n_tok][valid_mask].mean().item()) if valid_mask.any() else float("-inf")
    sel_valid_mask = sel_mask[:n_tok] & valid_mask

    def _dist_stats(x: torch.Tensor):
        if x.numel() == 0:
            return {
                "mean": float("nan"),
                "p10": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
            }
        x = x.float()
        return {
            "mean": float(x.mean().item()),
            "p10": float(torch.quantile(x, 0.10).item()),
            "p50": float(torch.quantile(x, 0.50).item()),
            "p90": float(torch.quantile(x, 0.90).item()),
        }

    lp_all_stats = _dist_stats(lp_cpu[:n_tok][valid_mask])
    lp_sel_stats = _dist_stats(lp_cpu[:n_tok][sel_valid_mask])
    token_keep_ratio_row = float(sel_valid_mask.float().mean().item()) if valid_mask.any() else float("nan")

    pseudo_events = _decode_pseudo_notes_with_token_indices(
        out_tokens,
        vocab,
        step_ms=step_ms,
        pad_id=pad_id,
        eos_id=eos_id,
    )
    selected_events: List[PseudoNoteEvent] = []
    for ev in pseudo_events:
        on_sel = (0 <= int(ev.on_tok_idx) < sel_mask.numel()) and bool(sel_mask[int(ev.on_tok_idx)].item())
        off_sel = (0 <= int(ev.off_tok_idx) < sel_mask.numel()) and bool(sel_mask[int(ev.off_tok_idx)].item())
        if on_sel or off_sel:
            selected_events.append(ev)

    summary = {
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "in_batch_idx": int(in_batch_idx),
        "chunk_selected": bool(chunk_selected),
        "chunk_conf_mean_logprob": float(chunk_conf),
        "n_tokens": int(len(tokens_trim)),
        "n_gt_notes": int(len(gt_intervals)),
        "n_pseudo_notes_chunk": int(len(pseudo_events)),
        "n_pseudo_notes_selected": int(len(selected_events)),
        "selected_token_count": int(sel_mask.sum().item()),
        "token_keep_ratio_row": float(token_keep_ratio_row),
        "logprob_all_mean": lp_all_stats["mean"],
        "logprob_all_p10": lp_all_stats["p10"],
        "logprob_all_p50": lp_all_stats["p50"],
        "logprob_all_p90": lp_all_stats["p90"],
        "logprob_sel_mean": lp_sel_stats["mean"],
        "logprob_sel_p10": lp_sel_stats["p10"],
        "logprob_sel_p50": lp_sel_stats["p50"],
        "logprob_sel_p90": lp_sel_stats["p90"],
    }
    if chunk_keep_ratio_batch is not None:
        summary["chunk_keep_ratio_batch"] = float(chunk_keep_ratio_batch)
    if token_keep_ratio_batch is not None:
        summary["token_keep_ratio_batch"] = float(token_keep_ratio_batch)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# pseudo debug sample\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n\n# tokens\n")
        f.write("token_ids=" + ",".join(str(int(t)) for t in tokens_trim) + "\n")
        f.write("# selected token indices (y_tg positions)\n")
        sel_idx = torch.where(sel_mask)[0].tolist()
        f.write("selected_token_indices=" + ",".join(str(int(i)) for i in sel_idx) + "\n")
        f.write("# pseudo notes (all)\n")
        for i, ev in enumerate(pseudo_events):
            f.write(
                f"all[{i}] pitch={int(ev.pitch)} onset={float(ev.onset):.3f} offset={float(ev.offset):.3f} "
                f"on_tok={int(ev.on_tok_idx)} off_tok={int(ev.off_tok_idx)}\n"
            )
        f.write("# pseudo notes (selected)\n")
        for i, ev in enumerate(selected_events):
            f.write(
                f"sel[{i}] pitch={int(ev.pitch)} onset={float(ev.onset):.3f} offset={float(ev.offset):.3f} "
                f"on_tok={int(ev.on_tok_idx)} off_tok={int(ev.off_tok_idx)}\n"
            )
        f.write("# gt notes (local chunk time)\n")
        for i in range(len(gt_intervals)):
            f.write(
                f"gt[{i}] pitch={int(gt_pitches[i])} onset={float(gt_intervals[i, 0]):.3f} "
                f"offset={float(gt_intervals[i, 1]):.3f}\n"
            )

    fig, ax = plt.subplots(figsize=(12, 4))

    def _draw_bars(intervals: np.ndarray, pitches: np.ndarray, *, color: str, label: str, alpha: float):
        first = True
        for i in range(len(intervals)):
            s = float(intervals[i, 0])
            e = float(intervals[i, 1])
            p = int(pitches[i])
            w = max(0.01, e - s)
            rect = Rectangle(
                (s, p - 0.42),
                w,
                0.84,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=alpha,
                label=(label if first else None),
            )
            ax.add_patch(rect)
            first = False

    if len(gt_intervals) > 0:
        _draw_bars(gt_intervals, gt_pitches, color="tab:blue", label="GT MIDI", alpha=0.45)

    if len(pseudo_events) > 0:
        est_int = np.array([[ev.onset, ev.offset] for ev in pseudo_events], dtype=float)
        est_pitch = np.array([ev.pitch for ev in pseudo_events], dtype=int)
        _draw_bars(est_int, est_pitch, color="gray", label="Pseudo chunk notes", alpha=0.35)

    if len(selected_events) > 0:
        sel_int = np.array([[ev.onset, ev.offset] for ev in selected_events], dtype=float)
        sel_pitch = np.array([ev.pitch for ev in selected_events], dtype=int)
        _draw_bars(sel_int, sel_pitch, color="tab:red", label="Selected pseudo notes", alpha=0.85)

    ax.set_xlim(0.0, max(0.1, float(window_sec)))
    ax.set_ylim(20, 108)
    ax.set_xlabel("Time (s, local chunk)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title(
        f"Pseudo Debug ep={epoch} batch={batch_idx} idx={in_batch_idx} "
        f"(chunk_selected={int(chunk_selected)})"
    )
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def make_collate_real():
    def _collate(batch):
        # batch: List[(Tensor[T,F], int, int)]
        mels = [b[0] for b in batch]
        idxs = torch.tensor([b[1] for b in batch], dtype=torch.long)
        starts = torch.tensor([b[2] for b in batch], dtype=torch.long)
        mels_padded = nn.utils.rnn.pad_sequence(mels, batch_first=True)  # [B,T,F]
        return mels_padded, idxs, starts
    return _collate

def _pm_to_note_arrays(pm) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    intervals: List[List[float]] = []
    pitches: List[int] = []
    velocities: List[int] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            intervals.append([float(n.start), float(n.end)])
            pitches.append(int(n.pitch))
            velocities.append(int(n.velocity))
    if len(intervals) == 0:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )
    arr_i = np.asarray(intervals, dtype=float)
    arr_p = np.asarray(pitches, dtype=int)
    arr_v = np.asarray(velocities, dtype=int)
    order = np.argsort(arr_i[:, 0])
    return arr_i[order], arr_p[order], arr_v[order]


@torch.no_grad()
def eval_loop_ddp(
    model_ddp,
    dl,
    crit,
    device,
    *,
    compute_token_acc: bool = False,
    compute_mir_eval: bool = False,
    vocab: Optional[Vocab] = None,
):
    """
    全rankで val を回し、loss を global average する
    """
    model_ddp.eval()
    total_loss = torch.zeros(1, device=device)
    n_batches = torch.zeros(1, device=device)
    token_correct = torch.zeros(1, device=device)
    token_count = torch.zeros(1, device=device)
    pad_id = int(crit.ignore_index)

    mir_onset_pitch_f_sum = torch.zeros(1, device=device)
    mir_onset_pitch_p_sum = torch.zeros(1, device=device)
    mir_onset_pitch_r_sum = torch.zeros(1, device=device)
    mir_note_f_sum = torch.zeros(1, device=device)
    mir_note_p_sum = torch.zeros(1, device=device)
    mir_note_r_sum = torch.zeros(1, device=device)
    mir_count = torch.zeros(1, device=device)

    if compute_mir_eval:
        if vocab is None:
            raise ValueError("vocab must be provided when compute_mir_eval=True")
        from my_mt3.eval import evaluate_notes_direct
        from my_mt3.infer import to_midi_from_tokens_piano

    for mels, y_in, y_tg in dl:
        mels, y_in, y_tg = mels.to(device, non_blocking=True), y_in.to(device, non_blocking=True), y_tg.to(device, non_blocking=True)
        logits = model_ddp(mels, y_in)
        loss = crit(logits.reshape(-1, logits.size(-1)), y_tg.reshape(-1))
        total_loss += loss.detach()
        n_batches += 1
        if compute_token_acc:
            pred = logits.argmax(dim=-1)
            valid = (y_tg != pad_id)
            token_correct += ((pred == y_tg) & valid).sum()
            token_count += valid.sum()
        if compute_mir_eval:
            pred = logits.argmax(dim=-1)
            for b in range(y_tg.size(0)):
                valid_b = (y_tg[b] != pad_id)
                if int(valid_b.sum().item()) == 0:
                    continue
                first_tok = int(y_in[b, 0].item())
                if first_tok == pad_id:
                    continue
                gt_seq = [first_tok] + y_tg[b][valid_b].tolist()
                est_seq = [first_tok] + pred[b][valid_b].tolist()
                pm_gt = to_midi_from_tokens_piano(gt_seq, program_id=0, step_ms=10, vocab=vocab).pm
                pm_est = to_midi_from_tokens_piano(est_seq, program_id=0, step_ms=10, vocab=vocab).pm
                gt_int, gt_pitch, gt_vel = _pm_to_note_arrays(pm_gt)
                est_int, est_pitch, est_vel = _pm_to_note_arrays(pm_est)
                try:
                    met = evaluate_notes_direct(
                        gt_int, gt_pitch, gt_vel,
                        est_int, est_pitch, est_vel,
                    )
                except Exception:
                    continue
                mir_onset_pitch_f_sum += float(met.get("onset_pitch_f", 0.0))
                mir_onset_pitch_p_sum += float(met.get("onset_pitch_p", 0.0))
                mir_onset_pitch_r_sum += float(met.get("onset_pitch_r", 0.0))
                mir_note_f_sum += float(met.get("note_f", 0.0))
                mir_note_p_sum += float(met.get("note_p", 0.0))
                mir_note_r_sum += float(met.get("note_r", 0.0))
                mir_count += 1.0

    # allreduce
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)
    if compute_token_acc:
        dist.all_reduce(token_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    if compute_mir_eval:
        dist.all_reduce(mir_onset_pitch_f_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mir_onset_pitch_p_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mir_onset_pitch_r_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mir_note_f_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mir_note_p_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mir_note_r_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mir_count, op=dist.ReduceOp.SUM)

    val_loss = (total_loss / torch.clamp(n_batches, min=1)).item()
    val_token_acc = (token_correct / torch.clamp(token_count, min=1)).item() if compute_token_acc else 0.0
    denom = torch.clamp(mir_count, min=1.0)
    val_onset_pitch_f = (mir_onset_pitch_f_sum / denom).item() if compute_mir_eval else 0.0
    val_onset_pitch_p = (mir_onset_pitch_p_sum / denom).item() if compute_mir_eval else 0.0
    val_onset_pitch_r = (mir_onset_pitch_r_sum / denom).item() if compute_mir_eval else 0.0
    val_note_f = (mir_note_f_sum / denom).item() if compute_mir_eval else 0.0
    val_note_p = (mir_note_p_sum / denom).item() if compute_mir_eval else 0.0
    val_note_r = (mir_note_r_sum / denom).item() if compute_mir_eval else 0.0
    return {
        "val_loss": float(val_loss),
        "val_token_acc": float(val_token_acc),
        "val_onset_pitch_f": float(val_onset_pitch_f),
        "val_onset_pitch_p": float(val_onset_pitch_p),
        "val_onset_pitch_r": float(val_onset_pitch_r),
        "val_note_f": float(val_note_f),
        "val_note_p": float(val_note_p),
        "val_note_r": float(val_note_r),
    }


def train_loop_distributed_DA_confusion(
    pairs,
    *,
    vocab: Vocab,
    # ---- Domain Adaptation (DC専用) ----
    use_dc: bool = True,
    pairs_real: dict | None = None ,            # {"train": [wav_path, ...]} 必須
    lambda_adv: float = 0.01,         # 論文既定
    lr_t: float = 2e-4,               # Transformer(E,D)
    lr_c: float = 1e-4,               # Discriminator
    chunk_frames: int | None = None,  # 0.1s相当。未指定なら hop, sr から自動決定
    disc_hidden: int = 256,
    # ---- SSL (pseudo) ----
    use_pseudo: bool = True,
    pseudo_start_epoch: int = 3,
    ema_decay: float = 0.999,
    unsup_weight: float = 1.0,
    pseudo_max_len: int = 1024,
    pseudo_threshold: float = -0.5,
    pseudo_topn: int = 0,
    # ---- 事前学習重み ----
    pretrained_ckpt: str | None = None,
    resume_ckpt: str | None = None,
    # ---- Oracle filter (実験用) ----
    oracle_filter: bool = False,
    oracle_metric: str = "note_f",
    oracle_threshold: float = 0.5,
    oracle_midi_paths: list | None = None,
    oracle_note_target_only: bool = False,
    oracle_note_without_chunk: bool = False,
    pseudo_note_target_only: bool = False,
    pseudo_note_onset_only: bool = False,
    pseudo_note_threshold: float = -0.5,
    pseudo_note_prob_threshold: Optional[float] = None,
    pseudo_note_mask_threshold: Optional[float] = None,
    pseudo_note_conf_mode: str = "single",
    pseudo_note_score_metric: str = "logprob_mean",
    pseudo_note_mask_score_metric: str = "abs_mask_delta",
    pseudo_note_mask_width_ratio: float = 0.2,
    pseudo_note_mask_fill: str = "zero",
    pseudo_note_without_chunk: bool = False,
    pseudo_repair_order: bool = False,
    pseudo_debug_n: int = 0,
    pseudo_debug_dir: str | None = None,
    pseudo_debug_start_epoch: int = 0,  # deprecated: debug starts at pseudo_start_epoch
    # ---- Auxiliary teacher forcing (timewise onset) ----
    timewise_onset_tf_weight: float = 0.0,
    timewise_onset_tf_max_groups: int = 0,
    timewise_onset_tf_min_onsets: int = 1,
    pseudo_unsup_cross_attn_only: bool = False,
    # ---- Augmentation ----
    use_augment: bool = True,
    # ---- 既存 ----
    epochs=5,
    bs=8,
    input_frames: int = INPUT_FRAMES,
    lr_warmup_epochs: int = 0,
    lr_min_ratio: float = 0.1,
    val_every: int = 2000,
    save_every=10,
    save_dir="checkpoints",
    use_cache: bool = True,
    cache_dir: str = "cache/wave_sr16000",
    sr: int = DEFAULT_SR,
    num_workers: int = 2,
):
    # ---- init DDP ----
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)

    rank = dist.get_rank()
    world = dist.get_world_size()

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        if use_cache and cache_dir:
            cache_dir_abs = os.path.abspath(cache_dir)
            os.makedirs(cache_dir_abs, exist_ok=True)
            cache_dir = cache_dir_abs
    dist.barrier()

    # ---- broadcast cache_dir (existing behavior) ----
    if use_cache and cache_dir:
        cache_dir_bytes = os.path.abspath(cache_dir).encode("utf-8") if rank == 0 else b""
        length_tensor = torch.tensor([len(cache_dir_bytes)], dtype=torch.int32, device=device)
        dist.broadcast(length_tensor, src=0)
        buf = torch.empty((int(length_tensor.item()),), dtype=torch.uint8, device=device)
        if rank == 0:
            buf.copy_(torch.tensor(list(cache_dir_bytes), dtype=torch.uint8, device=device))
        dist.broadcast(buf, src=0)
        cache_dir = bytes(buf.cpu().numpy().tolist()).decode("utf-8")

    # ---- cache synth pairs ----
    pairs = _maybe_cache_pairs_map(pairs, sr=sr, cache_dir=(cache_dir if use_cache else None))

    # ---- auto chunk_frames for 0.1s ----
    # 1 frame sec = hop/sr (AMTDataset hop=256, sr=16000 => 16ms)
    # 0.1 / 0.016 = 6.25 => 6 or 7
    if chunk_frames is None:
        # AMTDatasetのhopは引数で受け取っていないので、あなたのデフォ hop=256 を想定
        # hopを外から変えている場合は chunk_frames を明示指定してください
        hop = 256
        frames_per_sec = sr / float(hop)
        chunk_frames = max(1, int(round(0.1 * frames_per_sec)))  # ≈6

    # ---- datasets (synth) ----
    train_ds = AMTDataset(pairs["train"], mode="train", sr=sr, input_frames=input_frames, vocab=vocab)
    val_ds   = AMTDataset(pairs["validation"], mode="validation", sr=sr, input_frames=input_frames, vocab=vocab)

    # ---- samplers/loaders (synth) ----
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=make_collate(vocab),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=bs,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=make_collate(vocab),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # ---- real loader（DCまたはpseudoで使用） ----
    real_dl = None
    if (use_dc or use_pseudo):
        if pairs_real is None or "train" not in pairs_real:
            raise ValueError("pairs_real={'train':[wav,...]} が必須です（DC/pseudoで使用）")
        real_wavs = pairs_real["train"]
        if real_wavs:
            real_ds = AMTRealDataset(real_wavs, sr=sr, hop=256, input_frames=input_frames, n_fft=2048, n_mels=256,)
            real_sampler = DistributedSampler(real_ds, shuffle=True, drop_last=True)
            real_dl = DataLoader(
                real_ds,
                batch_size=bs,
                sampler=real_sampler,
                shuffle=False,
                collate_fn=make_collate_real(),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=(num_workers > 0),
            )

    # ---- Oracle MIDI cache ----
    oracle_midi_cache: Dict[int, object] = {}
    _need_samples_for_oracle = 0
    if (oracle_filter or int(pseudo_debug_n) > 0) and oracle_midi_paths is not None:
        import pretty_midi as _pm
        _need_samples_for_oracle = (input_frames - 1) * 256 + 2048
        if rank == 0:
            print(f"[oracle] loading {len(oracle_midi_paths)} reference MIDIs ...")
        for i, mp in enumerate(oracle_midi_paths):
            try:
                oracle_midi_cache[i] = _pm.PrettyMIDI(mp)
            except Exception as e:
                if rank == 0:
                    print(f"[oracle] skip idx={i}: {e}")
        if rank == 0:
            print(f"[oracle] cached {len(oracle_midi_cache)} / {len(oracle_midi_paths)} MIDIs")
            print(f"[oracle] metric={oracle_metric}  threshold={oracle_threshold}")
            if oracle_note_target_only:
                msg = "[oracle] note-target-only mode enabled"
                if oracle_note_without_chunk:
                    msg += " (without chunk filter)"
                print(msg)
    if rank == 0 and pseudo_note_target_only:
        msg = "[pseudo-note] token-target-only mode enabled"
        if pseudo_note_without_chunk:
            msg += " (without chunk filter)"
        print(msg)
        _print_conf_mode = str(pseudo_note_conf_mode)
        _print_legacy_metric = str(pseudo_note_score_metric)
        _print_mask_metric = str(pseudo_note_mask_score_metric)
        _print_prob_th = float(pseudo_note_threshold if pseudo_note_prob_threshold is None else pseudo_note_prob_threshold)
        _print_mask_th = float(pseudo_note_threshold if pseudo_note_mask_threshold is None else pseudo_note_mask_threshold)
        print(
            "[pseudo-note] conf mode="
            f"{_print_conf_mode} | legacy_metric={_print_legacy_metric} "
            f"| prob_th={_print_prob_th:.4f} | mask_metric={_print_mask_metric} | mask_th={_print_mask_th:.4f}"
        )
        if (_print_conf_mode in {"mask", "prob_and_mask", "prob_or_mask"}) or (_print_conf_mode == "single" and _print_legacy_metric in {"abs_mask_delta", "log_abs_mask_delta"}):
            print(
                "[pseudo-note] mask-effect params: "
                f"width_ratio={float(pseudo_note_mask_width_ratio):.3f}, fill={str(pseudo_note_mask_fill)}"
            )
    if rank == 0 and pseudo_repair_order:
        print("[pseudo-note] order-repair enabled: same-time pitch low->high, on->off, dedup same token")
    if rank == 0 and float(timewise_onset_tf_weight) > 0.0:
        print(
            "[timewise-tf] enabled: "
            f"weight={float(timewise_onset_tf_weight):.4f}, "
            f"max_groups={int(timewise_onset_tf_max_groups)}, "
            f"min_onsets={int(timewise_onset_tf_min_onsets)}"
        )
    if rank == 0 and bool(pseudo_unsup_cross_attn_only):
        print("[pseudo-unsup] cross-attention-only update enabled")
    pseudo_debug_written = 0
    pseudo_debug_root = str(pseudo_debug_dir) if pseudo_debug_dir else os.path.join(save_dir, "pseudo_debug")
    pseudo_metrics_csv = os.path.join(pseudo_debug_root, "pseudo_metrics.csv")
    if rank == 0 and int(pseudo_debug_n) > 0:
        os.makedirs(pseudo_debug_root, exist_ok=True)
        print(f"[pseudo-debug] enabled: n={int(pseudo_debug_n)} dir={pseudo_debug_root}")
        if not os.path.exists(pseudo_metrics_csv):
            try:
                import csv as _csv
                with open(pseudo_metrics_csv, "w", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow([
                        "epoch", "batch_idx",
                        "chunk_keep_ratio", "token_keep_ratio",
                        "n_chunks", "n_tokens_valid", "n_tokens_selected",
                        "logprob_all_mean", "logprob_all_p10", "logprob_all_p50", "logprob_all_p90",
                        "logprob_sel_mean", "logprob_sel_p10", "logprob_sel_p50", "logprob_sel_p90",
                    ])
            except Exception as e:
                print(f"[pseudo-debug] failed to init metrics csv: {e}")

    # ---- models ----
    model = MT3Mini(vocab_size=len(vocab.itos)).to(device)
    if (pretrained_ckpt is not None) and (resume_ckpt is None):
        ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=True)
        state = ckpt if not isinstance(ckpt, dict) else ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if rank == 0:
            print(f"[pretrained] loaded {pretrained_ckpt}")
            if missing:
                print(f"  missing keys : {missing}")
            if unexpected:
                print(f"  unexpected keys: {unexpected}")
    model_ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    crit_ce = nn.CrossEntropyLoss(ignore_index=vocab.pad)
    bce = nn.BCEWithLogitsLoss()

    # Optimizers / Discriminator
    disc_ddp = None
    if use_dc:
        disc = Discriminator(d=384, hidden=disc_hidden).to(device)
        disc_ddp = DDP(disc, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        opt_c = optim.AdamW(disc_ddp.parameters(), lr=lr_c)
    opt_t = optim.AdamW(model_ddp.parameters(), lr=lr_t)
    total_updates = max(1, epochs * len(train_dl))
    warmup_updates = max(0, int(lr_warmup_epochs) * len(train_dl))
    min_ratio = float(lr_min_ratio)
    min_ratio = min(max(min_ratio, 0.0), 1.0)

    def _lr_lambda(step_idx: int) -> float:
        if warmup_updates > 0 and step_idx < warmup_updates:
            return float(step_idx + 1) / float(warmup_updates)
        if total_updates <= warmup_updates:
            return 1.0
        progress = float(step_idx - warmup_updates) / float(total_updates - warmup_updates)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    sch_t = optim.lr_scheduler.LambdaLR(opt_t, lr_lambda=_lr_lambda)

    if rank == 0:
        msg = f"[DDP-DA] world={world} | synth_train_songs={len(train_ds)} | val_songs={len(val_ds)}"
        if real_dl is not None:
            msg += f" | real_wavs={len(real_ds)}"
        if use_dc:
            msg += f" | DC(lambda_adv={lambda_adv}, chunk_frames={chunk_frames})"
        print(msg)
        # DA損失ログCSVを初期化
        try:
            import csv as _csv
            csv_path = os.path.join(save_dir, "da_losses.csv")
            file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            mode = "a" if (resume_ckpt is not None and file_exists) else "w"
            with open(csv_path, mode, newline="") as f:
                w = _csv.writer(f)
                if mode == "w":
                    w.writerow([
                        "epoch", "train_total", "train_sup", "train_adv", "train_unsup", "train_disc", "val_loss",
                        "val_token_acc",
                        "val_onset_pitch_f", "val_onset_pitch_p", "val_onset_pitch_r",
                        "val_note_f", "val_note_p", "val_note_r",
                        "pseudo_chunks", "pseudo_notes"
                    ])
        except Exception:
            pass
    ema = EMATeacher(model_ddp.module, decay=ema_decay)
    start_epoch = 0

    if resume_ckpt is not None:
        ckpt_obj = torch.load(resume_ckpt, map_location=device, weights_only=False)
        if isinstance(ckpt_obj, dict) and ("model" in ckpt_obj):
            missing, unexpected = model_ddp.module.load_state_dict(ckpt_obj["model"], strict=False)
            if rank == 0:
                print(f"[resume] loaded model from {resume_ckpt}")
                if missing:
                    print(f"  missing keys : {missing}")
                if unexpected:
                    print(f"  unexpected keys: {unexpected}")
            if "optimizer_t" in ckpt_obj:
                opt_t.load_state_dict(ckpt_obj["optimizer_t"])
            if "scheduler_t" in ckpt_obj:
                sch_t.load_state_dict(ckpt_obj["scheduler_t"])
            if "ema_teacher" in ckpt_obj:
                ema.teacher.load_state_dict(ckpt_obj["ema_teacher"], strict=False)
            if use_dc and disc_ddp is not None:
                if "disc" in ckpt_obj:
                    disc_ddp.module.load_state_dict(ckpt_obj["disc"], strict=False)
                if ("optimizer_c" in ckpt_obj) and ("opt_c" in locals()):
                    opt_c.load_state_dict(ckpt_obj["optimizer_c"])
            start_epoch = int(ckpt_obj.get("epoch", -1)) + 1
            start_epoch = max(0, start_epoch)
        else:
            state = ckpt_obj if not isinstance(ckpt_obj, dict) else ckpt_obj.get("model", ckpt_obj)
            model_ddp.module.load_state_dict(state, strict=False)
            ckpt_name = os.path.basename(str(resume_ckpt))
            m = re.search(r"ep(\d+)\.pt$", ckpt_name)
            start_epoch = int(m.group(1)) if m else 0
            if rank == 0:
                print(f"[resume] model-only checkpoint loaded from {resume_ckpt}")
                print("[resume] optimizer/scheduler state not found; using fresh optimizer")
                if m:
                    print(f"[resume] inferred start epoch from filename: {start_epoch}")

        if rank == 0:
            print(f"[resume] start_epoch={start_epoch + 1} / target_epochs={epochs}")

    if rank == 0 and (pretrained_ckpt is not None) and (resume_ckpt is not None):
        print("[resume] --resume_ckpt is set; --pretrained_ckpt is ignored")

    if start_epoch >= int(epochs):
        if rank == 0:
            print(f"[resume] start_epoch ({start_epoch}) >= epochs ({epochs}); nothing to train")
        dist.destroy_process_group()
        return model_ddp

    _note_conf_mode = str(pseudo_note_conf_mode)
    _valid_note_conf_modes = {"single", "prob", "mask", "prob_and_mask", "prob_or_mask"}
    if _note_conf_mode not in _valid_note_conf_modes:
        raise ValueError(
            f"invalid pseudo_note_conf_mode={_note_conf_mode} "
            f"(choose one of {sorted(_valid_note_conf_modes)})"
        )

    _note_score_metric = str(pseudo_note_score_metric)
    _valid_note_score_metrics = {"logprob_mean", "abs_mask_delta", "log_abs_mask_delta"}
    if _note_score_metric not in _valid_note_score_metrics:
        raise ValueError(
            f"invalid pseudo_note_score_metric={_note_score_metric} "
            f"(choose one of {sorted(_valid_note_score_metrics)})"
        )
    _mask_score_metric = str(pseudo_note_mask_score_metric)
    _valid_mask_score_metrics = {"abs_mask_delta", "log_abs_mask_delta"}
    if _mask_score_metric not in _valid_mask_score_metrics:
        raise ValueError(
            f"invalid pseudo_note_mask_score_metric={_mask_score_metric} "
            f"(choose one of {sorted(_valid_mask_score_metrics)})"
        )

    _prob_threshold = float(pseudo_note_threshold if pseudo_note_prob_threshold is None else pseudo_note_prob_threshold)
    _mask_threshold = float(pseudo_note_threshold if pseudo_note_mask_threshold is None else pseudo_note_mask_threshold)

    _mask_fill = "mean" if str(pseudo_note_mask_fill) == "mean" else "zero"
    _step_ms_note_metric = 10

    aug_cfg = AugmentConfig() if use_augment else None
    # ---- loop ----
    for ep in range(start_epoch, epochs):
        train_sampler.set_epoch(ep)
        if (use_dc or use_pseudo) and real_dl is not None:
            real_sampler.set_epoch(ep)

        model_ddp.train()
        if use_dc and disc_ddp is not None:
            disc_ddp.train()

        running_total = 0.0
        running_sup = 0.0
        running_adv = 0.0
        running_unsup = 0.0
        running_disc = 0.0
        running_pseudo_chunks = 0
        running_pseudo_notes = 0
        n_batches = 0
        real_iter = itertools.cycle(real_dl) if real_dl is not None else None

        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}", unit="batch", disable=(rank != 0))
        if use_pseudo:
            fast_dec = FastDecoderKV(ema.teacher.dec, max_len=pseudo_max_len).to(device).eval()

        for batch_idx, (mels_s, y_in_s, y_tg_s) in enumerate(pbar, start=1):
            mels_r, real_idxs, real_starts = (None, None, None)
            if real_iter is not None:
                mels_r, real_idxs, real_starts = next(real_iter)
            if mels_r is not None:
                mels_r = mels_r.to(device, non_blocking=True)

            mels_s = mels_s.to(device, non_blocking=True)
            y_in_s = y_in_s.to(device, non_blocking=True)
            y_tg_s = y_tg_s.to(device, non_blocking=True)

            # ===== encoder outputs =====
            mem_s = model_ddp.module.enc(mels_s)
            mem_r = model_ddp.module.enc(mels_r) if mels_r is not None else None

            # ========= (A) Discriminator step (Eq.1) =========
            loss_disc = torch.zeros((), device=device)
            if use_dc and disc_ddp is not None and mem_r is not None:
                logit_s = disc_ddp(mem_s.detach(), chunk_frames=chunk_frames)
                logit_r = disc_ddp(mem_r.detach(), chunk_frames=chunk_frames)
                loss_disc = bce(logit_s, torch.zeros_like(logit_s)) + bce(logit_r, torch.ones_like(logit_r))
                opt_c.zero_grad(set_to_none=True)
                loss_disc.backward()
                opt_c.step()

            # ========= (B) Student step: supervised + adv + pseudo =========
            if use_dc and disc_ddp is not None:
                for p in disc_ddp.parameters():
                    p.requires_grad_(False)

            # (1) supervised CE on synth
            logits_s = model_ddp.module.dec(y_in_s, mem_s)
            loss_sup = crit_ce(logits_s.reshape(-1, logits_s.size(-1)), y_tg_s.reshape(-1))
            loss_sup_timewise = torch.zeros((), device=device)
            if float(timewise_onset_tf_weight) > 0.0:
                aux_batch = _build_timewise_onset_tf_batch(
                    y_tg=y_tg_s,
                    vocab=vocab,
                    max_groups_per_sample=int(timewise_onset_tf_max_groups),
                    min_onsets_per_group=int(timewise_onset_tf_min_onsets),
                )
                if aux_batch is not None:
                    src_idx_aux, y_in_aux, y_tg_aux = aux_batch
                    mem_aux = mem_s.index_select(0, src_idx_aux)
                    logits_aux = model_ddp.module.dec(y_in_aux, mem_aux)
                    loss_sup_timewise = crit_ce(
                        logits_aux.reshape(-1, logits_aux.size(-1)),
                        y_tg_aux.reshape(-1),
                    )

            # (2) adversarial confusion on synth+real
            loss_adv = torch.zeros((), device=device)
            if use_dc and disc_ddp is not None and mem_r is not None:
                logit_s2 = disc_ddp(mem_s, chunk_frames=chunk_frames)
                logit_r2 = disc_ddp(mem_r, chunk_frames=chunk_frames)
                half_s = 0.5 * torch.ones_like(logit_s2)
                half_r = 0.5 * torch.ones_like(logit_r2)
                loss_adv = bce(logit_s2, half_s) + bce(logit_r2, half_r)

            # (3) pseudo-label loss on real (start from pseudo_start_epoch)
            loss_unsup = torch.zeros((), device=device)

            if use_pseudo and real_dl is not None and (ep + 1) >= pseudo_start_epoch:
                out, _pmax, _margin, log_prob = pseudo_label_with_kvcache(
                    teacher=ema.teacher,
                    fast_dec=fast_dec,
                    mel=mels_r,
                    program_id=0,
                    vocab=vocab,
                    max_new_tokens=pseudo_max_len,
                    return_with_prefix=False,
                )
                if pseudo_repair_order:
                    out, log_prob = canonicalize_pseudo_batch_order(
                        out,
                        log_prob,
                        vocab=vocab,
                        pad_id=int(vocab.pad),
                        eos_id=int(vocab.eos),
                    )

                B = out.size(0)
                prg_id = int(vocab.instrument_type["PRG_0"])
                prg = torch.full((B, 1), prg_id, dtype=torch.long, device=out.device)
                y_in_p = torch.cat([prg, out[:, :-1]], dim=1)
                y_tg_p = out

                pad_id = int(vocab.pad)
                eos_id = int(vocab.eos)

                if oracle_filter and oracle_midi_cache and real_idxs is not None:
                    chunk_mask = oracle_chunk_filter(
                        out, real_idxs, real_starts,
                        oracle_midi_cache=oracle_midi_cache,
                        vocab=vocab,
                        sr=sr,
                        need_samples=_need_samples_for_oracle,
                        oracle_metric=oracle_metric,
                        oracle_threshold=oracle_threshold,
                        device=device,
                    )
                else:
                    chunk_mask = pseudo_chunk_filter(
                        out, log_prob,
                        pad_id=pad_id, eos_id=eos_id, device=device,
                        pseudo_threshold=pseudo_threshold,
                        pseudo_topn=pseudo_topn,
                    )

                use_token_only_without_chunk = bool(pseudo_note_target_only and pseudo_note_without_chunk)
                use_oracle_token_only_without_chunk = bool(
                    oracle_note_target_only and oracle_note_without_chunk and oracle_filter
                )
                can_use_unsup = bool(chunk_mask.any()) or use_token_only_without_chunk or use_oracle_token_only_without_chunk

                if can_use_unsup:
                    running_pseudo_chunks += int(chunk_mask.sum().item())
                    if oracle_note_target_only and oracle_filter and oracle_midi_cache and real_idxs is not None:
                        note_mask = oracle_note_token_mask(
                            out, real_idxs, real_starts,
                            oracle_midi_cache=oracle_midi_cache,
                            vocab=vocab,
                            sr=sr,
                            need_samples=_need_samples_for_oracle,
                            step_ms=10,
                            onset_tolerance=0.05,
                            offset_ratio=0.2,
                            offset_min_tolerance=0.05,
                            device=device,
                        )
                        if oracle_note_without_chunk:
                            final_mask = note_mask
                        else:
                            final_mask = note_mask & chunk_mask.unsqueeze(1)
                        y_tg_masked = torch.full_like(y_tg_p, pad_id)
                        y_tg_masked[final_mask] = y_tg_p[final_mask]
                        note_on_ids = set(vocab.note_on.values())
                        note_count = 0
                        if final_mask.any():
                            selected_tokens = y_tg_p[final_mask].tolist()
                            note_count = sum(1 for t in selected_tokens if t in note_on_ids)
                        running_pseudo_notes += int(note_count)
                    elif pseudo_note_target_only:
                        note_mask = torch.zeros_like(y_tg_p, dtype=torch.bool)
                        note_on_ids = set(vocab.note_on.values())
                        if use_token_only_without_chunk:
                            target_idxs = list(range(out.size(0)))
                        else:
                            target_idxs = torch.where(chunk_mask)[0].tolist()

                        for b_idx in target_idxs:
                            spans = decode_notes_to_spans(out[b_idx].tolist(), vocab)
                            if len(spans) == 0:
                                continue
                            prob_scores = build_note_confidences(spans, log_prob[b_idx])

                            need_mask_scores = (
                                _note_conf_mode in {"mask", "prob_and_mask", "prob_or_mask"}
                                or (_note_conf_mode == "single" and _note_score_metric in {"abs_mask_delta", "log_abs_mask_delta"})
                            )
                            mask_scores = None
                            if need_mask_scores:
                                token_ids_b = [int(t) for t in out[b_idx].tolist()]
                                y_in_b = y_in_p[b_idx].unsqueeze(0)
                                y_tg_b = y_tg_p[b_idx].unsqueeze(0)
                                with torch.no_grad():
                                    base_tf_logp = _teacher_forced_token_logp(
                                        model=ema.teacher,
                                        mel_1=mels_r[b_idx:b_idx + 1],
                                        y_in_1=y_in_b,
                                        y_tg_1=y_tg_b,
                                    )
                                frame_map = _token_time_frame_map(
                                    token_ids_b,
                                    vocab=vocab,
                                    sr=int(sr),
                                    hop=256,
                                    step_ms=int(_step_ms_note_metric),
                                )
                                note_on_ids = set(vocab.note_on.values())
                                onset_pos = []
                                for ns in spans:
                                    found = None
                                    for t_idx in ns.tok_ids:
                                        if 0 <= int(t_idx) < len(token_ids_b) and int(token_ids_b[int(t_idx)]) in note_on_ids:
                                            found = int(t_idx)
                                            break
                                    onset_pos.append(int(found if found is not None else (ns.tok_ids[0] if ns.tok_ids else -1)))
                                needed_frames = sorted(
                                    {
                                        int(frame_map[p])
                                        for p in onset_pos
                                        if (0 <= int(p) < len(frame_map)) and (frame_map[p] is not None)
                                    }
                                )
                                masked_by_frame: Dict[int, torch.Tensor] = {}
                                for fr in needed_frames:
                                    mel_mask = apply_source_mask_band(
                                        mels_r[b_idx:b_idx + 1],
                                        center_frame=int(fr),
                                        width_ratio=float(pseudo_note_mask_width_ratio),
                                        fill=_mask_fill,
                                    )
                                    with torch.no_grad():
                                        masked_by_frame[int(fr)] = _teacher_forced_token_logp(
                                            model=ema.teacher,
                                            mel_1=mel_mask,
                                            y_in_1=y_in_b,
                                            y_tg_1=y_tg_b,
                                        )
                                mask_scores = _build_note_mask_effect_confidences(
                                    spans=spans,
                                    token_ids=token_ids_b,
                                    base_logp_1d=base_tf_logp,
                                    masked_logp_by_frame=masked_by_frame,
                                    token_frame_map=frame_map,
                                    note_on_ids=note_on_ids,
                                    use_log_of_abs=(_mask_score_metric == "log_abs_mask_delta"),
                                )

                            if _note_conf_mode == "single":
                                if _note_score_metric == "logprob_mean":
                                    keep_mask_note = prob_scores >= float(_prob_threshold)
                                else:
                                    keep_mask_note = np.asarray(mask_scores, dtype=float) >= float(_mask_threshold)
                            elif _note_conf_mode == "prob":
                                keep_mask_note = prob_scores >= float(_prob_threshold)
                            elif _note_conf_mode == "mask":
                                if mask_scores is None:
                                    mask_scores = np.full_like(prob_scores, fill_value=-float("inf"), dtype=float)
                                keep_mask_note = mask_scores >= float(_mask_threshold)
                            elif _note_conf_mode == "prob_and_mask":
                                if mask_scores is None:
                                    mask_scores = np.full_like(prob_scores, fill_value=-float("inf"), dtype=float)
                                keep_mask_note = (prob_scores >= float(_prob_threshold)) & (mask_scores >= float(_mask_threshold))
                            else:  # prob_or_mask
                                if mask_scores is None:
                                    mask_scores = np.full_like(prob_scores, fill_value=-float("inf"), dtype=float)
                                keep_mask_note = (prob_scores >= float(_prob_threshold)) | (mask_scores >= float(_mask_threshold))

                            keep_note_idxs = np.where(keep_mask_note)[0].tolist()
                            if not keep_note_idxs:
                                continue
                            for n_idx in keep_note_idxs:
                                for t_idx in spans[n_idx].tok_ids:
                                    if 0 <= int(t_idx) < y_tg_p.size(1):
                                        note_mask[b_idx, int(t_idx)] = True

                        final_mask = note_mask
                        if not use_token_only_without_chunk:
                            final_mask = final_mask & chunk_mask.unsqueeze(1)
                        if pseudo_note_onset_only:
                            onset_mask = torch.zeros_like(y_tg_p, dtype=torch.bool)
                            for note_on_id in note_on_ids:
                                onset_mask |= (y_tg_p == int(note_on_id))
                            final_mask &= onset_mask

                        y_tg_masked = torch.full_like(y_tg_p, pad_id)
                        y_tg_masked[final_mask] = y_tg_p[final_mask]
                        note_count = 0
                        if final_mask.any():
                            selected_tokens = y_tg_p[final_mask].tolist()
                            note_count = sum(1 for t in selected_tokens if t in note_on_ids)
                        running_pseudo_notes += int(note_count)
                    else:
                        y_tg_masked = y_tg_p.clone()
                        y_tg_masked[~chunk_mask] = pad_id
                        kept_idxs = torch.where(chunk_mask)[0].tolist()
                        for b_idx in kept_idxs:
                            running_pseudo_notes += len(decode_notes_to_spans(out[b_idx].tolist(), vocab))

                    selected_token_mask = (y_tg_masked != pad_id)
                    lp_len = int(log_prob.size(1))
                    valid_lp_mask = (out[:, :lp_len] != pad_id) & (out[:, :lp_len] != eos_id)
                    selected_lp_mask = selected_token_mask[:, :lp_len] & valid_lp_mask
                    chunk_keep_ratio = float(chunk_mask.float().mean().item())
                    token_keep_ratio = float(selected_lp_mask.float().mean().item())

                    lp_all = log_prob[valid_lp_mask]
                    lp_sel = log_prob[selected_lp_mask]

                    def _batch_lp_stats(x: torch.Tensor):
                        if x.numel() == 0:
                            return (float("nan"), float("nan"), float("nan"), float("nan"))
                        x = x.float()
                        return (
                            float(x.mean().item()),
                            float(torch.quantile(x, 0.10).item()),
                            float(torch.quantile(x, 0.50).item()),
                            float(torch.quantile(x, 0.90).item()),
                        )

                    all_mean, all_p10, all_p50, all_p90 = _batch_lp_stats(lp_all)
                    sel_mean, sel_p10, sel_p50, sel_p90 = _batch_lp_stats(lp_sel)

                    if rank == 0 and int(pseudo_debug_n) > 0:
                        try:
                            import csv as _csv
                            with open(pseudo_metrics_csv, "a", newline="") as f:
                                w = _csv.writer(f)
                                w.writerow([
                                    int(ep + 1), int(batch_idx),
                                    f"{chunk_keep_ratio:.6f}", f"{token_keep_ratio:.6f}",
                                    int(chunk_mask.numel()), int(valid_lp_mask.sum().item()), int(selected_lp_mask.sum().item()),
                                    f"{all_mean:.6f}", f"{all_p10:.6f}", f"{all_p50:.6f}", f"{all_p90:.6f}",
                                    f"{sel_mean:.6f}", f"{sel_p10:.6f}", f"{sel_p50:.6f}", f"{sel_p90:.6f}",
                                ])
                        except Exception as e:
                            print(f"[pseudo-debug] failed to append metrics: {e}")

                    dbg_active = (rank == 0) and (int(pseudo_debug_n) > 0) and (pseudo_debug_written < int(pseudo_debug_n))
                    if dbg_active:
                        ep1 = int(ep + 1)
                        if ep1 < int(pseudo_start_epoch):
                            dbg_active = False

                    if dbg_active and real_idxs is not None and real_starts is not None:
                        try:
                            from my_mt3.eval import extract_notes_in_range
                            epoch_debug_root = os.path.join(pseudo_debug_root, f"ep_{ep1:05d}")
                            window_sec_dbg = (
                                _need_samples_for_oracle / float(sr)
                                if _need_samples_for_oracle > 0
                                else (((input_frames - 1) * 256 + 2048) / float(sr))
                            )
                            kept_idxs = torch.where(chunk_mask)[0].tolist()
                            if not kept_idxs:
                                kept_idxs = torch.where(selected_token_mask.any(dim=1))[0].tolist()
                            for b_idx in kept_idxs:
                                if pseudo_debug_written >= int(pseudo_debug_n):
                                    break
                                idx = int(real_idxs[b_idx].item())
                                ss = int(real_starts[b_idx].item())
                                t0 = ss / float(sr)
                                t1 = t0 + window_sec_dbg
                                ref_pm = oracle_midi_cache.get(idx)
                                if ref_pm is not None:
                                    gt_int, gt_pitch, _ = extract_notes_in_range(ref_pm, t0, t1, program=0)
                                else:
                                    gt_int = np.zeros((0, 2), dtype=float)
                                    gt_pitch = np.zeros((0,), dtype=int)
                                _save_pseudo_debug_sample(
                                    out_tokens=out[b_idx].tolist(),
                                    log_prob_row=log_prob[b_idx],
                                    selected_token_mask_row=selected_token_mask[b_idx],
                                    chunk_selected=bool(chunk_mask[b_idx].item()),
                                    save_root=epoch_debug_root,
                                    sample_idx=pseudo_debug_written + 1,
                                    epoch=ep1,
                                    batch_idx=batch_idx,
                                    in_batch_idx=int(b_idx),
                                    vocab=vocab,
                                    gt_intervals=gt_int,
                                    gt_pitches=gt_pitch,
                                    window_sec=float(window_sec_dbg),
                                    step_ms=10,
                                    chunk_keep_ratio_batch=chunk_keep_ratio,
                                    token_keep_ratio_batch=token_keep_ratio,
                                )
                                pseudo_debug_written += 1
                        except Exception as e:
                            if rank == 0:
                                print(f"[pseudo-debug] skip due to error: {e}")

                    # teacher: clean mel (used in pseudo_label_with_kvcache above)
                    # student: augmented mel for consistency regularization
                    if aug_cfg is not None:
                        mels_r_aug = augment_spectrogram(mels_r, aug_cfg)
                        mem_r_student = model_ddp.module.enc(mels_r_aug)
                    else:
                        mem_r_student = mem_r

                    if (y_tg_masked != pad_id).any():
                        logits_r = model_ddp.module.dec(y_in_p.to(device), mem_r_student)
                        loss_unsup = crit_ce(logits_r.reshape(-1, logits_r.size(-1)), y_tg_masked.to(device).reshape(-1))

            loss_main = (
                loss_sup
                + float(timewise_onset_tf_weight) * loss_sup_timewise
                + lambda_adv * loss_adv
            )
            loss_total = loss_main + unsup_weight * loss_unsup

            opt_t.zero_grad(set_to_none=True)
            use_cross_only_unsup = bool(
                pseudo_unsup_cross_attn_only
                and float(unsup_weight) != 0.0
                and bool(loss_unsup.requires_grad)
            )
            if use_cross_only_unsup:
                loss_main.backward(retain_graph=True)
                prev_req = _set_model_trainable_only_unsup_cross_attn(model_ddp.module)
                try:
                    (unsup_weight * loss_unsup).backward()
                finally:
                    _restore_model_requires_grad(model_ddp.module, prev_req)
            else:
                loss_total.backward()
            opt_t.step()
            sch_t.step()

            if use_dc and disc_ddp is not None:
                for p in disc_ddp.parameters():
                    p.requires_grad_(True)

            # EMA update（pseudoを使うときに有効）
            if use_pseudo and (ep + 1) >= pseudo_start_epoch:
                ema.update(model_ddp.module)

            # accumulate
            running_total += float(loss_total.item())
            running_sup += float(loss_sup.item())
            running_adv += float(loss_adv.item())
            running_unsup += float(loss_unsup.item())
            running_disc += float(loss_disc.item())
            n_batches += 1
            if rank == 0:
                pbar.set_postfix(
                    loss=f"{loss_total.item():.3f}",
                    sup=f"{loss_sup.item():.3f}",
                    sup_tw=f"{loss_sup_timewise.item():.3f}",
                    adv=f"{loss_adv.item():.3f}",
                    unsup=f"{float(loss_unsup.item()):.3f}",
                    disc=f"{loss_disc.item():.3f}",
                )


        # ---- val (optional) ----
        val_loss = 0.0
        val_token_acc = 0.0
        val_onset_pitch_f = 0.0
        val_onset_pitch_p = 0.0
        val_onset_pitch_r = 0.0
        val_note_f = 0.0
        val_note_p = 0.0
        val_note_r = 0.0
        if (ep + 1) % int(val_every) == 0 or (ep + 1) == epochs:
            val_metrics = eval_loop_ddp(
                model_ddp,
                val_dl,
                crit_ce,
                device,
                compute_token_acc=((ep + 1) == epochs),
                compute_mir_eval=True,
                vocab=vocab,
            )
            val_loss = float(val_metrics["val_loss"])
            val_token_acc = float(val_metrics["val_token_acc"])
            val_onset_pitch_f = float(val_metrics["val_onset_pitch_f"])
            val_onset_pitch_p = float(val_metrics["val_onset_pitch_p"])
            val_onset_pitch_r = float(val_metrics["val_onset_pitch_r"])
            val_note_f = float(val_metrics["val_note_f"])
            val_note_p = float(val_metrics["val_note_p"])
            val_note_r = float(val_metrics["val_note_r"])

        if rank == 0:
            denom = max(1, n_batches)
            avg_total = running_total / denom
            avg_sup = running_sup / denom
            avg_adv = running_adv / denom
            avg_unsup = running_unsup / denom
            avg_disc = running_disc / denom
            print(
                f"[epoch {ep+1}] train_loss={avg_total:.3f} | "
                f"val_loss={val_loss:.3f} | val_token_acc={val_token_acc:.4f} | "
                f"onset_pitch(f/p/r)={val_onset_pitch_f:.4f}/{val_onset_pitch_p:.4f}/{val_onset_pitch_r:.4f} | "
                f"note(f/p/r)={val_note_f:.4f}/{val_note_p:.4f}/{val_note_r:.4f}"
            )
            # CSVへ追記
            try:
                import csv as _csv
                with open(os.path.join(save_dir, "da_losses.csv"), "a", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow([
                        ep+1, f"{avg_total:.6f}", f"{avg_sup:.6f}", f"{avg_adv:.6f}", f"{avg_unsup:.6f}",
                        f"{avg_disc:.6f}", f"{val_loss:.6f}", f"{val_token_acc:.6f}",
                        f"{val_onset_pitch_f:.6f}", f"{val_onset_pitch_p:.6f}", f"{val_onset_pitch_r:.6f}",
                        f"{val_note_f:.6f}", f"{val_note_p:.6f}", f"{val_note_r:.6f}",
                        running_pseudo_chunks, running_pseudo_notes
                    ])
            except Exception:
                pass

            if (ep + 1) % save_every == 0 or (ep + 1) == epochs:
                train_state_path = os.path.join(save_dir, f"train_state_ep{ep+1}.pt")
                train_state = {
                    "epoch": ep,
                    "model": model_ddp.module.state_dict(),
                    "optimizer_t": opt_t.state_dict(),
                    "scheduler_t": sch_t.state_dict(),
                    "ema_teacher": ema.teacher.state_dict(),
                    "use_dc": bool(use_dc),
                }
                if use_dc and disc_ddp is not None:
                    train_state["disc"] = disc_ddp.module.state_dict()
                    if "opt_c" in locals():
                        train_state["optimizer_c"] = opt_c.state_dict()
                torch.save(train_state, train_state_path)
                print(f"✅ saved -> {train_state_path}")

                save_path = os.path.join(save_dir, f"model_ep{ep+1}.pt")
                torch.save(model_ddp.module.state_dict(), save_path)
                print(f"✅ saved -> {save_path}")

                if use_dc and disc_ddp is not None:
                    disc_path = os.path.join(save_dir, f"disc_ep{ep+1}.pt")
                    torch.save(disc_ddp.module.state_dict(), disc_path)
                    print(f"✅ saved -> {disc_path}")

        dist.barrier()

    dist.destroy_process_group()
    return model_ddp
