from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pretty_midi
import torch
from tqdm import tqdm

from my_mt3.audio import load_audio_mono
from my_mt3.dataset import LogMelCfg, LogMelExtractor
from my_mt3.infer import greedy_decode_batch_with_logprobs
from my_mt3.model import MT3Mini
from my_mt3.tokenizer import INPUT_FRAMES, VOCAB, build_vocab
from my_mt3.eval import extract_notes_in_range
from my_mt3.analysis_attribution import apply_source_mask_band
from my_mt3.train_DA_confusion import (
    _build_note_mask_effect_confidences,
    _save_pseudo_debug_sample,
    _teacher_forced_token_logp,
    _token_time_frame_map,
    build_note_confidences,
    canonicalize_pseudo_batch_order,
    decode_notes_to_spans,
    pseudo_chunk_filter,
)
from run_mt3.infer_maestro import collect_pairs_maestro


def _shift_spans_to_out_index(
    spans: List,
    *,
    seq_len: int,
) -> List:
    """
    decode_notes_to_spans() returns indices aligned to training y_tg convention.
    For demo inference, y_tg is directly `out`, so shift by +1.
    """
    out = []
    for ns in spans:
        ids = [int(t) + 1 for t in ns.tok_ids if 0 <= (int(t) + 1) < int(seq_len)]
        if not ids:
            continue
        out.append(type(ns)(tok_ids=ids))
    return out


def _decoder_logits_and_self_attn_avg(
    model: MT3Mini,
    mel_1: torch.Tensor,   # [1,T,F]
    y_in_1: torch.Tensor,  # [1,S]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      logits_1: [S, V]
      w_avg:    [S, S] averaged over decoder layers and heads
    """
    mem = model.enc(mel_1)
    tgt = model.dec.pos(model.dec.emb(y_in_1))
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
    h = tgt
    w_layers: List[torch.Tensor] = []

    for blk in model.dec.blocks:
        if bool(getattr(blk, "norm_first", False)):
            h_norm = blk.norm1(h)
            sa_out, sa_w = blk.self_attn(
                h_norm, h_norm, h_norm,
                attn_mask=tgt_mask,
                key_padding_mask=None,
                need_weights=True,
                average_attn_weights=False,
            )
            h = h + blk.dropout1(sa_out)
            w_layers.append(sa_w.mean(dim=1)[0])  # [S,S]

            h_norm2 = blk.norm2(h)
            ca_out = blk.multihead_attn(
                h_norm2, mem, mem,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=False,
            )[0]
            h = h + blk.dropout2(ca_out)

            h_norm3 = blk.norm3(h)
            ff = blk.linear2(blk.dropout(blk.activation(blk.linear1(h_norm3))))
            h = h + blk.dropout3(ff)
        else:
            sa_out, sa_w = blk.self_attn(
                h, h, h,
                attn_mask=tgt_mask,
                key_padding_mask=None,
                need_weights=True,
                average_attn_weights=False,
            )
            h = blk.norm1(h + blk.dropout1(sa_out))
            w_layers.append(sa_w.mean(dim=1)[0])  # [S,S]

            ca_out = blk.multihead_attn(
                h, mem, mem,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=False,
            )[0]
            h = blk.norm2(h + blk.dropout2(ca_out))

            ff = blk.linear2(blk.dropout(blk.activation(blk.linear1(h))))
            h = blk.norm3(h + blk.dropout3(ff))

    logits_1 = model.dec.lm(h)[0]
    w_avg = torch.stack(w_layers, dim=0).mean(dim=0) if w_layers else torch.zeros((h.size(1), h.size(1)), device=h.device)
    return logits_1, w_avg


def _star_token_scores(
    *,
    model: MT3Mini,
    mel_1: torch.Tensor,  # [1,T,F]
    y_in_1: torch.Tensor,  # [1,S]
    y_tg_1: torch.Tensor,  # [1,S]
    lam: float,
    tau: float,
    prompt_offset: int = 0,
) -> torch.Tensor:
    """
    Compute STAR paper-style token score S_l from confidence C_l and attentive score A_l.
    Returns:
      S: [S]
    """
    eps = 1e-8
    logits, w = _decoder_logits_and_self_attn_avg(model, mel_1, y_in_1)  # [S,V], [S,S]
    logp = torch.log_softmax(logits, dim=-1)
    tgt = y_tg_1[0].long()
    c = torch.exp(logp.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)).clamp_min(eps)  # [S]

    s_len = int(tgt.numel())
    a = torch.zeros((s_len,), dtype=logits.dtype, device=logits.device)
    for l in range(s_len):
        start = int(prompt_offset)
        if l >= start:
            term1 = w[l, start:l + 1].sum()
        else:
            term1 = torch.zeros((), dtype=logits.dtype, device=logits.device)
        if l + 1 < s_len:
            term2 = w[l + 1:s_len, l].sum()
        else:
            term2 = torch.zeros((), dtype=logits.dtype, device=logits.device)
        a[l] = (term1 + term2).clamp_min(eps)

    r1 = (a * a / c).clamp_min(eps)
    r2 = (c * c / a).clamp_min(eps)
    s_conf = (torch.sigmoid(r1 - float(lam)) + torch.sigmoid(r2 - float(lam))) * a
    s_cons = (torch.sigmoid(float(lam) - r1) * torch.sigmoid(float(lam) - r2)) * a * torch.exp((c - a) / float(tau))
    return s_conf + s_cons


def _to_padded_tensors(
    outputs: List[List[int]],
    logprobs: List[List[float]],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = len(outputs)
    max_len = max((len(x) for x in outputs), default=1)
    out = torch.full((bsz, max_len), int(pad_id), dtype=torch.long, device=device)
    lp = torch.zeros((bsz, max_len), dtype=torch.float32, device=device)
    for b in range(bsz):
        n = min(len(outputs[b]), max_len)
        if n > 0:
            out[b, :n] = torch.tensor(outputs[b][:n], dtype=torch.long, device=device)
            lp[b, :n] = torch.tensor(logprobs[b][:n], dtype=torch.float32, device=device)
    return out, lp


def _build_selected_token_mask_like_training(
    *,
    model: MT3Mini,
    mels_bt: torch.Tensor,
    out_bt: torch.Tensor,
    log_prob_bt: torch.Tensor,
    vocab,
    sr: int,
    hop: int,
    step_ms: int,
    pseudo_threshold: float,
    pseudo_topn: int,
    pseudo_note_conf_mode: str,
    pseudo_note_score_metric: str,
    pseudo_note_mask_score_metric: str,
    pseudo_note_prob_threshold: float,
    pseudo_note_mask_threshold: float,
    pseudo_note_mask_width_ratio: float,
    pseudo_note_mask_fill: str,
    pseudo_note_onset_only: bool,
    pseudo_note_without_chunk: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = int(vocab.pad)
    eos_id = int(vocab.eos)

    chunk_mask = pseudo_chunk_filter(
        out_bt,
        log_prob_bt,
        pad_id=pad_id,
        eos_id=eos_id,
        device=out_bt.device,
        pseudo_threshold=float(pseudo_threshold),
        pseudo_topn=int(pseudo_topn),
    )

    bsz = out_bt.size(0)
    prg_id = int(vocab.instrument_type["PRG_0"])
    prg = torch.full((bsz, 1), prg_id, dtype=torch.long, device=out_bt.device)
    y_in_p = torch.cat([prg, out_bt[:, :-1]], dim=1)
    y_tg_p = out_bt

    note_mask = torch.zeros_like(y_tg_p, dtype=torch.bool)
    note_on_ids = set(vocab.note_on.values())

    if bool(pseudo_note_without_chunk):
        target_idxs = list(range(bsz))
    else:
        target_idxs = torch.where(chunk_mask)[0].tolist()

    _note_conf_mode = str(pseudo_note_conf_mode)
    _note_score_metric = str(pseudo_note_score_metric)
    _mask_score_metric = str(pseudo_note_mask_score_metric)
    _mask_fill = "mean" if str(pseudo_note_mask_fill) == "mean" else "zero"

    for b_idx in target_idxs:
        spans_raw = decode_notes_to_spans(out_bt[b_idx].tolist(), vocab)
        spans = _shift_spans_to_out_index(spans_raw, seq_len=int(y_tg_p.size(1)))
        if len(spans) == 0:
            continue

        prob_scores = build_note_confidences(spans, log_prob_bt[b_idx])
        need_mask_scores = (
            _note_conf_mode in {"mask", "prob_and_mask", "prob_or_mask"}
            or (_note_conf_mode == "single" and _note_score_metric in {"abs_mask_delta", "log_abs_mask_delta"})
        )

        mask_scores = None
        if need_mask_scores:
            token_ids_b = [int(t) for t in out_bt[b_idx].tolist()]
            y_in_b = y_in_p[b_idx].unsqueeze(0)
            y_tg_b = y_tg_p[b_idx].unsqueeze(0)

            with torch.no_grad():
                base_tf_logp = _teacher_forced_token_logp(
                    model=model,
                    mel_1=mels_bt[b_idx:b_idx + 1],
                    y_in_1=y_in_b,
                    y_tg_1=y_tg_b,
                )

            frame_map = _token_time_frame_map(
                token_ids_b,
                vocab=vocab,
                sr=int(sr),
                hop=int(hop),
                step_ms=int(step_ms),
            )

            onset_pos: List[int] = []
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
                    mels_bt[b_idx:b_idx + 1],
                    center_frame=int(fr),
                    width_ratio=float(pseudo_note_mask_width_ratio),
                    fill=_mask_fill,
                )
                with torch.no_grad():
                    masked_by_frame[int(fr)] = _teacher_forced_token_logp(
                        model=model,
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
                keep_mask_note = prob_scores >= float(pseudo_note_prob_threshold)
            else:
                keep_mask_note = np.asarray(mask_scores, dtype=float) >= float(pseudo_note_mask_threshold)
        elif _note_conf_mode == "prob":
            keep_mask_note = prob_scores >= float(pseudo_note_prob_threshold)
        elif _note_conf_mode == "mask":
            if mask_scores is None:
                mask_scores = np.full_like(prob_scores, fill_value=-float("inf"), dtype=float)
            keep_mask_note = mask_scores >= float(pseudo_note_mask_threshold)
        elif _note_conf_mode == "prob_and_mask":
            if mask_scores is None:
                mask_scores = np.full_like(prob_scores, fill_value=-float("inf"), dtype=float)
            keep_mask_note = (prob_scores >= float(pseudo_note_prob_threshold)) & (
                mask_scores >= float(pseudo_note_mask_threshold)
            )
        else:  # prob_or_mask
            if mask_scores is None:
                mask_scores = np.full_like(prob_scores, fill_value=-float("inf"), dtype=float)
            keep_mask_note = (prob_scores >= float(pseudo_note_prob_threshold)) | (
                mask_scores >= float(pseudo_note_mask_threshold)
            )

        keep_note_idxs = np.where(keep_mask_note)[0].tolist()
        if not keep_note_idxs:
            continue
        for n_idx in keep_note_idxs:
            for t_idx in spans[n_idx].tok_ids:
                if 0 <= int(t_idx) < y_tg_p.size(1):
                    note_mask[b_idx, int(t_idx)] = True

    final_mask = note_mask
    if not bool(pseudo_note_without_chunk):
        final_mask = final_mask & chunk_mask.unsqueeze(1)

    if bool(pseudo_note_onset_only):
        onset_mask = torch.zeros_like(y_tg_p, dtype=torch.bool)
        for note_on_id in note_on_ids:
            onset_mask |= (y_tg_p == int(note_on_id))
        final_mask &= onset_mask

    return chunk_mask, final_mask


def _build_selected_token_mask_star(
    *,
    model: MT3Mini,
    mels_bt: torch.Tensor,
    out_bt: torch.Tensor,
    log_prob_bt: torch.Tensor,
    vocab,
    sr: int,
    hop: int,
    step_ms: int,
    pseudo_threshold: float,
    pseudo_topn: int,
    star_lambda: float,
    star_tau: float,
    star_prompt_offset: int,
    star_token_threshold: Optional[float],
    star_topk_ratio: float,
    pseudo_note_onset_only: bool,
    pseudo_note_without_chunk: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = int(vocab.pad)
    eos_id = int(vocab.eos)
    chunk_mask = pseudo_chunk_filter(
        out_bt,
        log_prob_bt,
        pad_id=pad_id,
        eos_id=eos_id,
        device=out_bt.device,
        pseudo_threshold=float(pseudo_threshold),
        pseudo_topn=int(pseudo_topn),
    )

    bsz = out_bt.size(0)
    prg_id = int(vocab.instrument_type["PRG_0"])
    note_on_ids = set(vocab.note_on.values())
    note_off_ids = set(vocab.note_off.values()) if vocab.note_off is not None else set()
    note_ids = set(int(x) for x in (note_on_ids | note_off_ids))

    final_mask = torch.zeros_like(out_bt, dtype=torch.bool)
    if bool(pseudo_note_without_chunk):
        target_idxs = list(range(bsz))
    else:
        target_idxs = torch.where(chunk_mask)[0].tolist()

    for b_idx in target_idxs:
        y_tg = out_bt[b_idx]
        y_in = torch.cat(
            [
                torch.tensor([prg_id], dtype=torch.long, device=out_bt.device),
                y_tg[:-1],
            ],
            dim=0,
        ).unsqueeze(0)
        y_tg_1 = y_tg.unsqueeze(0)

        with torch.no_grad():
            s = _star_token_scores(
                model=model,
                mel_1=mels_bt[b_idx:b_idx + 1],
                y_in_1=y_in,
                y_tg_1=y_tg_1,
                lam=float(star_lambda),
                tau=float(star_tau),
                prompt_offset=int(star_prompt_offset),
            )  # [S]

        valid = (y_tg != pad_id) & (y_tg != eos_id)
        is_note_tok = torch.zeros_like(valid, dtype=torch.bool)
        for tid in note_ids:
            is_note_tok |= (y_tg == int(tid))
        cand = valid & is_note_tok
        cand_idx = torch.where(cand)[0]
        if cand_idx.numel() == 0:
            continue

        if star_token_threshold is not None:
            keep = cand & (s >= float(star_token_threshold))
        else:
            k = max(1, int(round(cand_idx.numel() * float(star_topk_ratio))))
            k = min(k, int(cand_idx.numel()))
            vals = s[cand_idx]
            _, topk_local = torch.topk(vals, k=k)
            keep_idx = cand_idx[topk_local]
            keep = torch.zeros_like(cand, dtype=torch.bool)
            keep[keep_idx] = True

        if bool(pseudo_note_onset_only):
            onset_mask = torch.zeros_like(keep, dtype=torch.bool)
            for note_on_id in note_on_ids:
                onset_mask |= (y_tg == int(note_on_id))
            keep &= onset_mask

        final_mask[b_idx] = keep

    if not bool(pseudo_note_without_chunk):
        final_mask = final_mask & chunk_mask.unsqueeze(1)

    return chunk_mask, final_mask


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Demo: infer MAESTRO chunks then apply STAR-like token filtering and plot selected-note piano rolls"
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints_maestro_SSL/model_ep10000.pt",
        help="checkpoint path (default: checkpoints_maestro_SSL/model_ep10000.pt)",
    )
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--out_dir", type=str, default="outputs/star_filter_demo")
    ap.add_argument("--max_songs", type=int, default=3)
    ap.add_argument("--max_chunks_per_song", type=int, default=0, help="0=all chunks")
    ap.add_argument("--save_samples", type=int, default=200, help="max number of debug samples to save")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--n_mels", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--program_id", type=int, default=0)
    ap.add_argument("--step_ms", type=int, default=10)

    # Chunk filter (same semantics as training pseudo_chunk_filter)
    ap.add_argument("--pseudo_threshold", type=float, default=-1.1)
    ap.add_argument("--pseudo_topn", type=int, default=0)

    # Note filter (same semantics as training)
    ap.add_argument(
        "--pseudo_note_conf_mode",
        type=str,
        default="prob_and_mask",
        choices=["single", "prob", "mask", "prob_and_mask", "prob_or_mask"],
    )
    ap.add_argument(
        "--pseudo_note_score_metric",
        type=str,
        default="logprob_mean",
        choices=["logprob_mean", "abs_mask_delta", "log_abs_mask_delta"],
    )
    ap.add_argument(
        "--pseudo_note_mask_score_metric",
        type=str,
        default="abs_mask_delta",
        choices=["abs_mask_delta", "log_abs_mask_delta"],
    )
    ap.add_argument("--pseudo_note_prob_threshold", type=float, default=-0.6)
    ap.add_argument("--pseudo_note_mask_threshold", type=float, default=0.3)
    ap.add_argument("--pseudo_note_mask_width_ratio", type=float, default=0.2)
    ap.add_argument("--pseudo_note_mask_fill", type=str, default="zero", choices=["zero", "mean"])
    ap.add_argument("--pseudo_note_onset_only", action="store_true")
    ap.add_argument("--pseudo_note_without_chunk", action="store_true")
    ap.add_argument("--pseudo_repair_order", action="store_true")
    ap.add_argument("--use_star_indicator", action="store_true", help="use STAR (C/A/S) token filter instead of note-conf filter")
    ap.add_argument("--star_lambda", type=float, default=2.0)
    ap.add_argument("--star_tau", type=float, default=10.0)
    ap.add_argument("--star_prompt_offset", type=int, default=0)
    ap.add_argument("--star_token_threshold", type=float, default=None, help="absolute threshold on STAR S_l (default: use topk ratio)")
    ap.add_argument("--star_topk_ratio", type=float, default=0.2, help="when star_token_threshold is None, keep top ratio among note tokens")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "pseudo_debug_demo"
    debug_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(input_frames=INPUT_FRAMES, instrument_type="piano", include_note_off=True)
    model = MT3Mini(vocab_size=len(VOCAB.itos)).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    pairs = collect_pairs_maestro(args.root, split=args.split, program_id=args.program_id)
    if args.max_songs > 0:
        pairs = pairs[: args.max_songs]
    if len(pairs) == 0:
        raise SystemExit("No songs found for given root/split.")

    feat = LogMelExtractor(LogMelCfg(sr=args.sr, n_fft=args.n_fft, hop=args.hop, n_mels=args.n_mels))
    need_samples = (INPUT_FRAMES - 1) * int(args.hop) + int(args.n_fft)
    window_sec = need_samples / float(args.sr)

    saved = 0
    global_chunk_idx = 0

    for song_idx, (audio_path, midi_path, pid) in enumerate(tqdm(pairs, desc="songs", unit="song"), start=1):
        if saved >= int(args.save_samples):
            break

        stem = Path(audio_path).stem
        ref_pm = pretty_midi.PrettyMIDI(midi_path)
        y, _ = load_audio_mono(audio_path, sr=args.sr)
        total_samples = len(y)
        stride_samples = INPUT_FRAMES * int(args.hop)
        starts = list(range(0, max(0, total_samples - need_samples) + 1, stride_samples))
        if not starts:
            starts = [0]
        if args.max_chunks_per_song > 0:
            starts = starts[: args.max_chunks_per_song]

        mel_list: List[np.ndarray] = []
        for ss in starts:
            y_seg = y[ss:ss + need_samples]
            if len(y_seg) < need_samples:
                y_seg = np.pad(y_seg, (0, need_samples - len(y_seg)))
            mel = feat(y_seg)
            if mel.shape[0] > INPUT_FRAMES:
                mel = mel[:INPUT_FRAMES]
            elif mel.shape[0] < INPUT_FRAMES:
                mel = np.pad(mel, ((0, INPUT_FRAMES - mel.shape[0]), (0, 0)))
            mel_list.append(mel.astype(np.float32, copy=False))

        for b0 in range(0, len(starts), int(args.batch_size)):
            if saved >= int(args.save_samples):
                break
            b1 = min(len(starts), b0 + int(args.batch_size))
            mels_bt = torch.from_numpy(np.stack(mel_list[b0:b1])).to(device=args.device, dtype=torch.float32)

            with torch.no_grad():
                out_list, lp_list = greedy_decode_batch_with_logprobs(
                    model,
                    mels_bt,
                    max_len=int(args.max_len),
                    device=args.device,
                    program_id=int(pid),
                    vocab=vocab,
                )
            out_bt, lp_bt = _to_padded_tensors(out_list, lp_list, pad_id=int(vocab.pad), device=mels_bt.device)
            if bool(args.pseudo_repair_order):
                out_bt, lp_bt = canonicalize_pseudo_batch_order(
                    out_bt,
                    lp_bt,
                    vocab=vocab,
                    pad_id=int(vocab.pad),
                    eos_id=int(vocab.eos),
                )

            if bool(args.use_star_indicator):
                chunk_mask, selected_token_mask = _build_selected_token_mask_star(
                    model=model,
                    mels_bt=mels_bt,
                    out_bt=out_bt,
                    log_prob_bt=lp_bt,
                    vocab=vocab,
                    sr=int(args.sr),
                    hop=int(args.hop),
                    step_ms=int(args.step_ms),
                    pseudo_threshold=float(args.pseudo_threshold),
                    pseudo_topn=int(args.pseudo_topn),
                    star_lambda=float(args.star_lambda),
                    star_tau=float(args.star_tau),
                    star_prompt_offset=int(args.star_prompt_offset),
                    star_token_threshold=args.star_token_threshold,
                    star_topk_ratio=float(args.star_topk_ratio),
                    pseudo_note_onset_only=bool(args.pseudo_note_onset_only),
                    pseudo_note_without_chunk=bool(args.pseudo_note_without_chunk),
                )
            else:
                chunk_mask, selected_token_mask = _build_selected_token_mask_like_training(
                    model=model,
                    mels_bt=mels_bt,
                    out_bt=out_bt,
                    log_prob_bt=lp_bt,
                    vocab=vocab,
                    sr=int(args.sr),
                    hop=int(args.hop),
                    step_ms=int(args.step_ms),
                    pseudo_threshold=float(args.pseudo_threshold),
                    pseudo_topn=int(args.pseudo_topn),
                    pseudo_note_conf_mode=str(args.pseudo_note_conf_mode),
                    pseudo_note_score_metric=str(args.pseudo_note_score_metric),
                    pseudo_note_mask_score_metric=str(args.pseudo_note_mask_score_metric),
                    pseudo_note_prob_threshold=float(args.pseudo_note_prob_threshold),
                    pseudo_note_mask_threshold=float(args.pseudo_note_mask_threshold),
                    pseudo_note_mask_width_ratio=float(args.pseudo_note_mask_width_ratio),
                    pseudo_note_mask_fill=str(args.pseudo_note_mask_fill),
                    pseudo_note_onset_only=bool(args.pseudo_note_onset_only),
                    pseudo_note_without_chunk=bool(args.pseudo_note_without_chunk),
                )

            lp_len = int(lp_bt.size(1))
            valid_lp_mask = (out_bt[:, :lp_len] != int(vocab.pad)) & (out_bt[:, :lp_len] != int(vocab.eos))
            selected_lp_mask = selected_token_mask[:, :lp_len] & valid_lp_mask
            chunk_keep_ratio = float(chunk_mask.float().mean().item())
            token_keep_ratio = float(selected_lp_mask.float().mean().item())

            for local_i in range(b1 - b0):
                if saved >= int(args.save_samples):
                    break
                seq_has_selected = bool(selected_token_mask[local_i].any().item())
                if not seq_has_selected:
                    continue

                chunk_idx = b0 + local_i
                ss = starts[chunk_idx]
                t0 = ss / float(args.sr)
                t1 = t0 + window_sec
                gt_int, gt_pitch, _ = extract_notes_in_range(ref_pm, t0, t1, program=int(pid))

                sample_root = debug_dir / f"song_{song_idx:03d}_{stem}"
                _save_pseudo_debug_sample(
                    out_tokens=[int(t) for t in out_bt[local_i].tolist()],
                    log_prob_row=lp_bt[local_i],
                    selected_token_mask_row=selected_token_mask[local_i],
                    chunk_selected=bool(chunk_mask[local_i].item()),
                    save_root=str(sample_root),
                    sample_idx=saved + 1,
                    epoch=0,
                    batch_idx=global_chunk_idx,
                    in_batch_idx=local_i,
                    vocab=vocab,
                    gt_intervals=gt_int,
                    gt_pitches=gt_pitch,
                    window_sec=window_sec,
                    step_ms=int(args.step_ms),
                    chunk_keep_ratio_batch=chunk_keep_ratio,
                    token_keep_ratio_batch=token_keep_ratio,
                )
                saved += 1
                global_chunk_idx += 1

    print(f"saved debug samples: {saved}")
    print(f"output dir: {debug_dir}")


if __name__ == "__main__":
    main()

