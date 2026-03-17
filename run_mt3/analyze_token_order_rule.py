from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import torch
import mir_eval
from tqdm import tqdm

from my_mt3.tokenizer import INPUT_FRAMES, VOCAB, build_vocab
from my_mt3.infer import greedy_decode_batch_with_logprobs
from my_mt3.eval import extract_notes_in_range
from run_mt3.infer_maestro import collect_pairs_maestro


def parse_time_groups(token_ids, lps, vocab, step_ms: int):
    eos = int(vocab.eos)
    id2time = {tid: t for t, tid in vocab.time.items()}
    id2on = {tid: p for p, tid in vocab.note_on.items()}
    id2off = {}
    if vocab.note_off is not None:
        id2off = {tid: p for p, tid in vocab.note_off.items()}

    groups = []
    cur_gid = -1
    cur_t_idx = 0
    for i, tid in enumerate(token_ids):
        tid = int(tid)
        if tid == eos:
            break
        lp = float(lps[i]) if i < len(lps) else np.nan
        if tid in id2time:
            cur_t_idx = int(id2time[tid])
            cur_gid += 1
            groups.append(
                {
                    "group_id": cur_gid,
                    "time_idx": cur_t_idx,
                    "time_sec": (cur_t_idx * int(step_ms)) / 1000.0,
                    "time_logp": lp,
                    "items": [],
                }
            )
            continue
        if cur_gid < 0:
            continue
        if tid in id2on:
            groups[cur_gid]["items"].append(
                {"type": "on", "pitch": int(id2on[tid]), "tok_idx": i, "logp": lp}
            )
        elif tid in id2off:
            groups[cur_gid]["items"].append(
                {"type": "off", "pitch": int(id2off[tid]), "tok_idx": i, "logp": lp}
            )
    return groups


def _matched_pred_indices(ref_times, pred_times, tol: float) -> set[int]:
    if len(ref_times) == 0 or len(pred_times) == 0:
        return set()
    matched = mir_eval.util.match_events(ref_times, pred_times, window=tol)
    if isinstance(matched, tuple) and len(matched) == 2:
        return {int(i) for i in matched[1]}
    return {int(j) for _, j in matched}


def event_correct_flags(events: list[dict], ref_int, ref_pitch, onset_tol=0.05, offset_tol=0.05):
    if len(events) == 0:
        return []
    if len(ref_int) == 0:
        return [False] * len(events)

    flags = [False] * len(events)
    ref_on = ref_int[:, 0]
    ref_off = ref_int[:, 1]

    for ev_type, ref_base, tol in (("on", ref_on, onset_tol), ("off", ref_off, offset_tol)):
        idxs = [i for i, e in enumerate(events) if e["type"] == ev_type]
        if not idxs:
            continue
        arr = [events[i] for i in idxs]
        pred_pitch = np.asarray([a["pitch"] for a in arr], dtype=int)
        pred_time = np.asarray([a["time_sec"] for a in arr], dtype=float)
        local_ok = np.zeros((len(arr),), dtype=bool)
        for p in np.unique(pred_pitch):
            m = pred_pitch == p
            pred_t_p = pred_time[m]
            pred_idx_p = np.where(m)[0]
            ref_t_p = ref_base[ref_pitch == p]
            matched = _matched_pred_indices(ref_t_p, pred_t_p, tol)
            for j in matched:
                local_ok[pred_idx_p[j]] = True
        for li, gi in enumerate(idxs):
            flags[gi] = bool(local_ok[li])
    return flags


def analyze_one_chunk(groups, ref_int, ref_pitch):
    token_rows = []
    group_rows = []

    flat_events = []
    for g in groups:
        for pos, it in enumerate(g["items"]):
            flat_events.append(
                {
                    "group_id": g["group_id"],
                    "time_idx": g["time_idx"],
                    "time_sec": g["time_sec"],
                    "time_logp": g["time_logp"],
                    "pos_in_group": pos,
                    **it,
                }
            )
    corr = event_correct_flags(flat_events, ref_int, ref_pitch, onset_tol=0.05, offset_tol=0.05)
    for ev, ok in zip(flat_events, corr):
        ev["correct"] = bool(ok)

    # group-level rule check + token-level violation attribution
    by_gid = {}
    for i, ev in enumerate(flat_events):
        by_gid.setdefault(int(ev["group_id"]), []).append((i, ev))

    for gid, arr in by_gid.items():
        arr = sorted(arr, key=lambda x: x[1]["pos_in_group"])
        kinds = [a[1]["type"] for a in arr]
        pitches_on = [a[1]["pitch"] for a in arr if a[1]["type"] == "on"]
        pitches_off = [a[1]["pitch"] for a in arr if a[1]["type"] == "off"]

        # kind rule: all on before any off
        seen_off = False
        kind_ok = True
        kind_violate_idx = set()
        for idx, ev in arr:
            if ev["type"] == "off":
                seen_off = True
            elif ev["type"] == "on" and seen_off:
                kind_ok = False
                kind_violate_idx.add(idx)

        # pitch ascending (non-decreasing) within each type subsequence
        on_ok = True
        off_ok = True
        last = -10**9
        on_violate_idx = set()
        for idx, ev in arr:
            if ev["type"] != "on":
                continue
            if ev["pitch"] < last:
                on_ok = False
                on_violate_idx.add(idx)
            last = max(last, ev["pitch"])
        last = -10**9
        off_violate_idx = set()
        for idx, ev in arr:
            if ev["type"] != "off":
                continue
            if ev["pitch"] < last:
                off_ok = False
                off_violate_idx.add(idx)
            last = max(last, ev["pitch"])

        group_rows.append(
            {
                "group_id": gid,
                "time_idx": int(arr[0][1]["time_idx"]),
                "time_sec": float(arr[0][1]["time_sec"]),
                "n_items": len(arr),
                "n_on": len(pitches_on),
                "n_off": len(pitches_off),
                "kind_rule_ok": bool(kind_ok),
                "on_pitch_order_ok": bool(on_ok),
                "off_pitch_order_ok": bool(off_ok),
                "group_rule_ok": bool(kind_ok and on_ok and off_ok),
            }
        )

        for idx, ev in arr:
            violate = (idx in kind_violate_idx) or (idx in on_violate_idx) or (idx in off_violate_idx)
            token_rows.append(
                {
                    **ev,
                    "kind_violation": bool(idx in kind_violate_idx),
                    "pitch_order_violation": bool((idx in on_violate_idx) or (idx in off_violate_idx)),
                    "rule_violation": bool(violate),
                    "rule_ok": bool(not violate),
                }
            )

    return token_rows, group_rows


def main():
    from my_mt3.model import MT3Mini
    from my_mt3.audio import load_audio_mono
    from my_mt3.dataset import LogMelCfg, LogMelExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--root", type=str, default="dataset/maestro-v3.0.0")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--out_dir", type=str, default="outputs/token_order_rule")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_songs", type=int, default=3)
    ap.add_argument("--max_chunks_per_song", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--step_ms", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(input_frames=INPUT_FRAMES, instrument_type="piano", include_note_off=True)
    model = MT3Mini(vocab_size=len(VOCAB.itos)).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    pairs = collect_pairs_maestro(args.root, split=args.split, program_id=0)
    if args.max_songs > 0:
        pairs = pairs[: args.max_songs]

    feat = LogMelExtractor(LogMelCfg(sr=16000, n_fft=2048, hop=256, n_mels=256))
    need_samples = (INPUT_FRAMES - 1) * 256 + 2048
    chunk_sec = need_samples / 16000.0

    token_rows_all = []
    group_rows_all = []

    for song_idx, (audio_path, midi_path, pid) in enumerate(tqdm(pairs, desc="songs")):
        stem = Path(audio_path).stem
        ref_pm = pretty_midi.PrettyMIDI(midi_path)
        y, _ = load_audio_mono(audio_path, sr=16000)
        stride = INPUT_FRAMES * 256
        starts = list(range(0, max(0, len(y) - need_samples) + 1, stride))
        if not starts:
            starts = [0]
        if args.max_chunks_per_song > 0:
            starts = starts[: args.max_chunks_per_song]

        mel_list = []
        for ss in starts:
            seg = y[ss:ss + need_samples]
            if len(seg) < need_samples:
                seg = np.pad(seg, (0, need_samples - len(seg)))
            mel = feat(seg)
            if mel.shape[0] > INPUT_FRAMES:
                mel = mel[:INPUT_FRAMES]
            elif mel.shape[0] < INPUT_FRAMES:
                mel = np.pad(mel, ((0, INPUT_FRAMES - mel.shape[0]), (0, 0)))
            mel_list.append(mel.astype(np.float32, copy=False))

        for b0 in range(0, len(starts), args.batch_size):
            b1 = min(len(starts), b0 + args.batch_size)
            mels_bt = torch.from_numpy(np.stack(mel_list[b0:b1])).to(args.device, dtype=torch.float32)
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        tok_batch, lp_batch = greedy_decode_batch_with_logprobs(
                            model, mels_bt, max_len=args.max_len, device=args.device, program_id=int(pid), vocab=vocab
                        )
                else:
                    tok_batch, lp_batch = greedy_decode_batch_with_logprobs(
                        model, mels_bt, max_len=args.max_len, device=args.device, program_id=int(pid), vocab=vocab
                    )

            for li in range(len(tok_batch)):
                chunk_idx = b0 + li
                ss = starts[chunk_idx]
                t0 = ss / 16000.0
                t1 = t0 + chunk_sec
                ref_int, ref_pitch, _ = extract_notes_in_range(ref_pm, t0, t1, program=int(pid))
                groups = parse_time_groups(tok_batch[li], lp_batch[li], vocab, step_ms=args.step_ms)
                tr, gr = analyze_one_chunk(groups, ref_int, ref_pitch)
                for r in tr:
                    token_rows_all.append(
                        {
                            "stem": stem,
                            "song_idx": song_idx,
                            "chunk_idx": chunk_idx,
                            "t0": t0,
                            "t1": t1,
                            **r,
                        }
                    )
                for r in gr:
                    group_rows_all.append(
                        {
                            "stem": stem,
                            "song_idx": song_idx,
                            "chunk_idx": chunk_idx,
                            "t0": t0,
                            "t1": t1,
                            **r,
                        }
                    )

    df_tok = pd.DataFrame(token_rows_all)
    df_grp = pd.DataFrame(group_rows_all)
    tok_csv = out_dir / "token_order_rule_token_level.csv"
    grp_csv = out_dir / "token_order_rule_group_level.csv"
    df_tok.to_csv(tok_csv, index=False)
    df_grp.to_csv(grp_csv, index=False)

    summary = {}
    if not df_tok.empty:
        for typ in ("on", "off"):
            sub = df_tok[df_tok["type"] == typ]
            if sub.empty:
                continue
            ok = sub[sub["rule_ok"] == True]
            ng = sub[sub["rule_violation"] == True]
            summary[f"{typ}_n"] = int(len(sub))
            summary[f"{typ}_rule_ok_rate"] = float(np.mean(sub["rule_ok"].astype(float)))
            summary[f"{typ}_correct_rate_all"] = float(np.mean(sub["correct"].astype(float)))
            summary[f"{typ}_correct_rate_rule_ok"] = float(np.mean(ok["correct"].astype(float))) if len(ok) else np.nan
            summary[f"{typ}_correct_rate_rule_violation"] = float(np.mean(ng["correct"].astype(float))) if len(ng) else np.nan

    if not df_grp.empty:
        summary["group_n"] = int(len(df_grp))
        summary["group_rule_ok_rate"] = float(np.mean(df_grp["group_rule_ok"].astype(float)))
        summary["group_kind_rule_ok_rate"] = float(np.mean(df_grp["kind_rule_ok"].astype(float)))
        summary["group_on_pitch_order_ok_rate"] = float(np.mean(df_grp["on_pitch_order_ok"].astype(float)))
        summary["group_off_pitch_order_ok_rate"] = float(np.mean(df_grp["off_pitch_order_ok"].astype(float)))

    summary_path = out_dir / "token_order_rule_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {tok_csv}")
    print(f"Saved: {grp_csv}")
    print(f"Saved: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
