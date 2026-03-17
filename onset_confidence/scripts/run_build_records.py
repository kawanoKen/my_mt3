from __future__ import annotations

import json
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from my_mt3.audio import load_audio_mono
from my_mt3.dataset import LogMelCfg, LogMelExtractor
from my_mt3.eval import extract_notes_in_range
from my_mt3.model import MT3Mini
from my_mt3.tokenizer import INPUT_FRAMES, VOCAB, build_vocab, encode_events
from run_mt3.infer_maestro import collect_pairs_maestro


def load_cfg(path: str) -> dict:
    txt = pathlib.Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception:
        return json.loads(txt)


def parse_gt_onsets(token_ids: list[int], vocab, step_ms: int):
    eos = int(vocab.eos)
    id2time = {tid: t for t, tid in vocab.time.items()}
    id2on = {tid: p for p, tid in vocab.note_on.items()}
    out = []
    onsets_by_time = defaultdict(set)
    cur_t = 0
    for tid in token_ids:
        tid = int(tid)
        if tid == eos:
            break
        if tid in id2time:
            cur_t = int(id2time[tid])
        elif tid in id2on:
            onsets_by_time[cur_t].add(int(id2on[tid]))

    cur_t = 0
    lower = []
    for i, tid in enumerate(token_ids):
        tid = int(tid)
        if tid == eos:
            break
        if tid in id2time:
            cur_t = int(id2time[tid])
            lower = []
            continue
        if tid in id2on:
            p = int(id2on[tid])
            out.append(
                dict(
                    step_index=i,
                    time_idx=cur_t,
                    event_time=(cur_t * step_ms) / 1000.0,
                    pitch=p,
                    onset_token_id=tid,
                    same_time_lower_pitches_gt=list(sorted(set(lower))),
                    same_time_onsets_gt=list(sorted(onsets_by_time[cur_t])),
                )
            )
            lower.append(p)
    return out


def build_context_prob_map(token_ids: list[int], probs_steps: torch.Tensor, vocab, target_onset_ids: list[int]):
    eos = int(vocab.eos)
    id2time = {tid: t for t, tid in vocab.time.items()}
    id2on = {tid: p for p, tid in vocab.note_on.items()}
    cur_t = 0
    lower = []
    mp = {}
    for i, tok in enumerate(token_ids):
        if i >= probs_steps.size(0):
            break
        key = (cur_t, tuple(sorted(set(lower))))
        slot = mp.setdefault(key, {})
        pv = probs_steps[i]
        for tid in target_onset_ids:
            if tid < pv.numel():
                slot[int(tid)] = float(pv[int(tid)].item())
        tok = int(tok)
        if tok == eos:
            break
        if tok in id2time:
            cur_t = int(id2time[tok])
            lower = []
        elif tok in id2on:
            lower.append(int(id2on[tok]))
    return mp


def decode_trace(model, mem, vocab, program_id: int, max_len: int, mode: str, temperature=1.0, topk=0):
    prg = int(vocab.instrument_type[f"PRG_{int(program_id)}"])
    eos = int(vocab.eos)
    y = torch.tensor([[prg]], dtype=torch.long, device=mem.device)
    toks, probs_rows, lps = [], [], []
    for _ in range(max_len):
        logits = model.dec(y, mem)[:, -1, :][0]
        if mode == "sample":
            z = logits / max(float(temperature), 1e-6)
            if topk > 0:
                vals, idx = torch.topk(z, k=min(int(topk), z.numel()))
                masked = torch.full_like(z, float("-inf"))
                masked[idx] = vals
                z = masked
            probs = torch.softmax(z, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.argmax(logits).item())
        toks.append(nxt)
        probs_rows.append(probs.detach().cpu())
        lps.append(float(torch.log(torch.clamp(probs[nxt], min=1e-12)).item()))
        y = torch.cat([y, torch.tensor([[nxt]], device=mem.device)], dim=1)
        if nxt == eos:
            break
    return toks, torch.stack(probs_rows, dim=0), float(sum(lps))


def beam_traces(model, mem, vocab, program_id: int, max_len: int, beam_size: int):
    prg = int(vocab.instrument_type[f"PRG_{int(program_id)}"])
    eos = int(vocab.eos)
    beams = [([prg], [], 0.0, False)]  # yseq, probs_rows(list[tensor]), lp, ended
    for _ in range(max_len):
        cand = []
        for yseq, pr, lp, ended in beams:
            if ended:
                cand.append((yseq, pr, lp, ended))
                continue
            y = torch.tensor([yseq], dtype=torch.long, device=mem.device)
            logits = model.dec(y, mem)[:, -1, :][0]
            logp = F.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            v, idx = torch.topk(logp, k=min(beam_size, logits.numel()))
            for lpi, ti in zip(v.tolist(), idx.tolist()):
                cand.append((yseq + [int(ti)], pr + [probs], lp + float(lpi), int(ti) == eos))
        cand.sort(key=lambda x: x[2], reverse=True)
        beams = cand[:beam_size]
        if all(b[3] for b in beams):
            break
    out = []
    for yseq, pr, lp, _ in beams:
        toks = yseq[1:]
        if not toks or not pr:
            continue
        out.append((toks, torch.stack(pr, dim=0), lp))
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="onset_confidence/conf/default.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)["build"]

    random.seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))
    torch.manual_seed(int(cfg.get("seed", 42)))

    out_csv = pathlib.Path(cfg["out_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(input_frames=INPUT_FRAMES, instrument_type="piano", include_note_off=True)
    model = MT3Mini(vocab_size=len(VOCAB.itos)).to(cfg.get("device", "cuda"))
    sd = torch.load(cfg["ckpt"], map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    pairs = collect_pairs_maestro(cfg["root"], split=cfg.get("split", "validation"), program_id=0)
    pairs = pairs[: int(cfg.get("max_songs", 3))]
    feat = LogMelExtractor(LogMelCfg(sr=16000, n_fft=2048, hop=256, n_mels=256))
    need_samples = (INPUT_FRAMES - 1) * 256 + 2048
    stride = INPUT_FRAMES * 256
    frame_max = len(vocab.time) - 1
    tid2pitch = {int(tid): int(p) for p, tid in vocab.note_on.items()}
    onset_ids = [int(tid) for tid in vocab.note_on.values()]

    rows = []
    for song_i, (apath, mpath, pid) in enumerate(tqdm(pairs, desc="songs")):
        y, _ = load_audio_mono(apath, sr=16000)
        import pretty_midi
        ref_pm = pretty_midi.PrettyMIDI(mpath)
        starts = list(range(0, max(0, len(y) - need_samples) + 1, stride))
        if not starts:
            starts = [0]
        mc = int(cfg.get("max_chunks_per_song", 0))
        if mc > 0:
            starts = starts[:mc]

        for chunk_i, ss in enumerate(starts):
            t0 = ss / 16000.0
            t1 = t0 + need_samples / 16000.0
            seg = y[ss:ss + need_samples]
            if len(seg) < need_samples:
                seg = np.pad(seg, (0, need_samples - len(seg)))
            mel = feat(seg)
            if mel.shape[0] > INPUT_FRAMES:
                mel = mel[:INPUT_FRAMES]
            elif mel.shape[0] < INPUT_FRAMES:
                mel = np.pad(mel, ((0, INPUT_FRAMES - mel.shape[0]), (0, 0)))
            mel_bt = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0).to(cfg.get("device", "cuda"))
            mem = model.enc(mel_bt)

            ref_int, ref_pitch, _ = extract_notes_in_range(ref_pm, t0, t1, program=int(pid))
            events = []
            for i in range(len(ref_int)):
                onq = int(np.clip(int(round((ref_int[i, 0] * 1000.0) / cfg.get("step_ms", 10))), 0, frame_max))
                offq = int(np.clip(int(round((ref_int[i, 1] * 1000.0) / cfg.get("step_ms", 10))), 0, frame_max))
                events.append((onq, offq, int(ref_pitch[i])))
            gt_tokens = encode_events(events, int(pid), ties=[], frame_max_token=frame_max, vocab=vocab)
            if len(gt_tokens) == 0:
                continue
            gt_steps = parse_gt_onsets(gt_tokens, vocab, int(cfg.get("step_ms", 10)))
            if not gt_steps:
                continue

            prg = int(vocab.instrument_type[f"PRG_{int(pid)}"])
            y_in = torch.tensor([[prg] + gt_tokens[:-1]], dtype=torch.long, device=mem.device)
            tf_logits = model.dec(y_in, mem)[0]
            tf_probs = torch.softmax(tf_logits, dim=-1).detach().cpu()

            g_toks, g_probs, _ = decode_trace(model, mem, vocab, int(pid), int(cfg.get("max_len", 512)), mode="greedy")
            g_map = build_context_prob_map(g_toks, g_probs, vocab, onset_ids)

            b_maps, b_lps = [], []
            for toks, probs, lp in beam_traces(model, mem, vocab, int(pid), int(cfg.get("max_len", 512)), int(cfg.get("beam_size", 3))):
                b_maps.append(build_context_prob_map(toks, probs, vocab, onset_ids))
                b_lps.append(lp)

            s_maps = []
            for _ in range(int(cfg.get("sample_n", 3))):
                stoks, sprobs, _ = decode_trace(
                    model,
                    mem,
                    vocab,
                    int(pid),
                    int(cfg.get("max_len", 512)),
                    mode="sample",
                    temperature=float(cfg.get("sample_temperature", 1.0)),
                    topk=int(cfg.get("sample_topk", 0)),
                )
                s_maps.append(build_context_prob_map(stoks, sprobs, vocab, onset_ids))

            for gt in gt_steps:
                step = int(gt["step_index"])
                if step >= tf_probs.size(0):
                    continue
                t_idx = int(gt["time_idx"])
                lower = [int(x) for x in gt["same_time_lower_pitches_gt"]]
                pos_tid = int(gt["onset_token_id"])
                vec = tf_probs[step].numpy()
                top = np.sort(vec)[::-1]
                margin = float(top[0] - top[1]) if len(top) > 1 else float(top[0])
                pmax = float(np.max(vec))

                key = (t_idx, tuple(sorted(set(lower))))
                def beam_score(tid: int):
                    vals, lps = [], []
                    for bm, lp in zip(b_maps, b_lps):
                        if key in bm and tid in bm[key]:
                            vals.append(float(bm[key][tid]))
                            lps.append(float(lp))
                    if not vals:
                        return np.nan
                    z = np.array(lps, dtype=np.float64)
                    z -= np.max(z)
                    w = np.exp(z)
                    w /= max(np.sum(w), 1e-12)
                    return float(np.sum(w * np.array(vals)))

                def sample_score(tid: int):
                    vals = [sm[key][tid] for sm in s_maps if key in sm and tid in sm[key]]
                    return float(np.mean(vals)) if vals else np.nan

                def same_time_score(tid: int, pitch: int):
                    vals = []
                    for sm in s_maps:
                        for (ti, lowers), d in sm.items():
                            if int(ti) != t_idx:
                                continue
                            if not all(int(lp) < int(pitch) for lp in lowers):
                                continue
                            if tid in d:
                                vals.append(float(d[tid]))
                    return float(np.mean(vals)) if vals else np.nan

                # positive
                rows.append(
                    dict(
                        sample_id=pathlib.Path(apath).stem,
                        chunk_id=chunk_i,
                        step_index=step,
                        event_time=float(gt["event_time"]),
                        time_idx=t_idx,
                        pitch=int(gt["pitch"]),
                        onset_token_id=pos_tid,
                        is_correct=1,
                        target_type="positive",
                        gt_prefix_length=step,
                        decoded_prefix_length=-1,
                        gt_token_context="",
                        decoded_token_context="",
                        same_time_lower_pitches_gt=json.dumps(lower),
                        same_time_lower_pitches_decoded=json.dumps(lower),
                        score_tf_local=float(tf_probs[step, pos_tid].item()) if pos_tid < tf_probs.size(1) else np.nan,
                        score_greedy_local=float(g_map.get(key, {}).get(pos_tid, np.nan)),
                        score_beam_marginal=beam_score(pos_tid),
                        score_sample_marginal=sample_score(pos_tid),
                        score_same_time_marginal=same_time_score(pos_tid, int(gt["pitch"])),
                        score_maxprob=pmax,
                        score_margin=margin,
                        prefix_edit_distance=np.nan,
                        prefix_match_rate=np.nan,
                        same_time_prefix_mismatch=np.nan,
                    )
                )

                # hard negatives
                gt_same = set(int(p) for p in gt["same_time_onsets_gt"])
                cand = sorted([(tid, float(tf_probs[step, tid].item())) for tid in onset_ids], key=lambda x: x[1], reverse=True)
                added = 0
                for neg_tid, _ in cand:
                    npitch = tid2pitch[int(neg_tid)]
                    if int(neg_tid) == pos_tid or npitch in gt_same:
                        continue
                    rows.append(
                        dict(
                            sample_id=pathlib.Path(apath).stem,
                            chunk_id=chunk_i,
                            step_index=step,
                            event_time=float(gt["event_time"]),
                            time_idx=t_idx,
                            pitch=npitch,
                            onset_token_id=int(neg_tid),
                            is_correct=0,
                            target_type="negative_hard",
                            gt_prefix_length=step,
                            decoded_prefix_length=-1,
                            gt_token_context="",
                            decoded_token_context="",
                            same_time_lower_pitches_gt=json.dumps(lower),
                            same_time_lower_pitches_decoded=json.dumps(lower),
                            score_tf_local=float(tf_probs[step, neg_tid].item()) if neg_tid < tf_probs.size(1) else np.nan,
                            score_greedy_local=float(g_map.get(key, {}).get(int(neg_tid), np.nan)),
                            score_beam_marginal=beam_score(int(neg_tid)),
                            score_sample_marginal=sample_score(int(neg_tid)),
                            score_same_time_marginal=same_time_score(int(neg_tid), npitch),
                            score_maxprob=pmax,
                            score_margin=margin,
                            prefix_edit_distance=np.nan,
                            prefix_match_rate=np.nan,
                            same_time_prefix_mismatch=np.nan,
                        )
                    )
                    added += 1
                    if added >= int(cfg.get("hard_neg_topk", 5)):
                        break

            pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved CSV: {out_csv} rows={len(rows)}")


if __name__ == "__main__":
    main()
