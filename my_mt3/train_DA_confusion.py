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
import itertools

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Function
from my_mt3.train import _maybe_cache_pairs_map, make_collate
from my_mt3.augment import AugmentConfig, augment_spectrogram

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


def make_collate_real():
    def _collate(batch):
        # batch: List[(Tensor[T,F], int, int)]
        mels = [b[0] for b in batch]
        idxs = torch.tensor([b[1] for b in batch], dtype=torch.long)
        starts = torch.tensor([b[2] for b in batch], dtype=torch.long)
        mels_padded = nn.utils.rnn.pad_sequence(mels, batch_first=True)  # [B,T,F]
        return mels_padded, idxs, starts
    return _collate

@torch.no_grad()
def eval_loop_ddp(model_ddp, dl, crit, device, *, compute_token_acc: bool = False):
    """
    全rankで val を回し、loss を global average する
    """
    model_ddp.eval()
    total_loss = torch.zeros(1, device=device)
    n_batches = torch.zeros(1, device=device)
    token_correct = torch.zeros(1, device=device)
    token_count = torch.zeros(1, device=device)
    pad_id = int(crit.ignore_index)

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

    # allreduce
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)
    if compute_token_acc:
        dist.all_reduce(token_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    val_loss = (total_loss / torch.clamp(n_batches, min=1)).item()
    val_token_acc = (token_correct / torch.clamp(token_count, min=1)).item() if compute_token_acc else 0.0
    return {"val_loss": float(val_loss), "val_token_acc": float(val_token_acc)}


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
    # ---- Oracle filter (実験用) ----
    oracle_filter: bool = False,
    oracle_metric: str = "note_f",
    oracle_threshold: float = 0.5,
    oracle_midi_paths: list | None = None,
    oracle_note_target_only: bool = False,
    # ---- Augmentation ----
    use_augment: bool = True,
    # ---- 既存 ----
    epochs=5,
    bs=8,
    input_frames: int = INPUT_FRAMES,
    lr_warmup_epochs: int = 0,
    lr_min_ratio: float = 0.1,
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
    if oracle_filter and oracle_midi_paths is not None:
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
                print("[oracle] note-target-only mode enabled")

    # ---- models ----
    model = MT3Mini(vocab_size=len(vocab.itos)).to(device)
    if pretrained_ckpt is not None:
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
            with open(os.path.join(save_dir, "da_losses.csv"), "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow([
                    "epoch", "train_total", "train_sup", "train_adv", "train_unsup", "train_disc", "val_loss",
                    "val_token_acc", "pseudo_chunks", "pseudo_notes"
                ])
        except Exception:
            pass
    ema = EMATeacher(model_ddp.module, decay=ema_decay)
    aug_cfg = AugmentConfig() if use_augment else None
    # ---- loop ----
    for ep in range(epochs):
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

        for mels_s, y_in_s, y_tg_s in pbar:
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

                if chunk_mask.any():
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
                        final_mask = note_mask & chunk_mask.unsqueeze(1)
                        y_tg_masked = torch.full_like(y_tg_p, pad_id)
                        y_tg_masked[final_mask] = y_tg_p[final_mask]
                        note_on_ids = set(vocab.note_on.values())
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

            loss_total = loss_sup + lambda_adv * loss_adv + unsup_weight * loss_unsup

            opt_t.zero_grad(set_to_none=True)
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
                    adv=f"{loss_adv.item():.3f}",
                    unsup=f"{float(loss_unsup.item()):.3f}",
                    disc=f"{loss_disc.item():.3f}",
                )


        # ---- val (optional) ----
        val_loss = 0.0
        val_token_acc = 0.0
        if (ep + 1) % 10 == 0 or (ep + 1) == epochs:
            val_metrics = eval_loop_ddp(
                model_ddp,
                val_dl,
                crit_ce,
                device,
                compute_token_acc=((ep + 1) == epochs),
            )
            val_loss = float(val_metrics["val_loss"])
            val_token_acc = float(val_metrics["val_token_acc"])

        if rank == 0:
            denom = max(1, n_batches)
            avg_total = running_total / denom
            avg_sup = running_sup / denom
            avg_adv = running_adv / denom
            avg_unsup = running_unsup / denom
            avg_disc = running_disc / denom
            print(
                f"[epoch {ep+1}] train_loss={avg_total:.3f} | "
                f"val_loss={val_loss:.3f} | val_token_acc={val_token_acc:.4f}"
            )
            # CSVへ追記
            try:
                import csv as _csv
                with open(os.path.join(save_dir, "da_losses.csv"), "a", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow([
                        ep+1, f"{avg_total:.6f}", f"{avg_sup:.6f}", f"{avg_adv:.6f}", f"{avg_unsup:.6f}",
                        f"{avg_disc:.6f}", f"{val_loss:.6f}", f"{val_token_acc:.6f}",
                        running_pseudo_chunks, running_pseudo_notes
                    ])
            except Exception:
                pass

            if (ep + 1) % save_every == 0 or (ep + 1) == epochs:
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
