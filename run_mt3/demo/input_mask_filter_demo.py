from __future__ import annotations

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pretty_midi
import torch
from tqdm import tqdm

from my_mt3.audio import load_audio_mono
from my_mt3.dataset import LogMelCfg, LogMelExtractor
from my_mt3.eval import extract_notes_in_range
from my_mt3.infer import greedy_decode_batch_with_logprobs
from my_mt3.model import MT3Mini
from my_mt3.tokenizer import INPUT_FRAMES, VOCAB, build_vocab
from my_mt3.train_DA_confusion import _save_pseudo_debug_sample, canonicalize_pseudo_batch_order
from run_mt3.demo_star_filter_infer import _build_selected_token_mask_like_training, _to_padded_tensors
from run_mt3.infer_maestro import collect_pairs_maestro


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo: infer -> input-mask note filter -> debug piano roll")
    ap.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints_maestro_SSL/model_ep10000.pt",
        help="checkpoint path (default: checkpoints_maestro_SSL/model_ep10000.pt)",
    )
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--out_dir", type=str, default="outputs/demo/input_mask_filter")
    ap.add_argument("--max_songs", type=int, default=3)
    ap.add_argument("--max_chunks_per_song", type=int, default=0, help="0=all chunks")
    ap.add_argument("--save_samples", type=int, default=100, help="max number of debug samples to save")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--n_mels", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--program_id", type=int, default=0)
    ap.add_argument("--step_ms", type=int, default=10)
    ap.add_argument("--pseudo_threshold", type=float, default=-1.5)
    ap.add_argument("--pseudo_topn", type=int, default=0)
    ap.add_argument("--pseudo_note_mask_threshold", type=float, default=0.3)
    ap.add_argument(
        "--pseudo_note_mask_score_metric",
        type=str,
        default="abs_mask_delta",
        choices=["abs_mask_delta", "log_abs_mask_delta"],
    )
    ap.add_argument("--pseudo_note_mask_width_ratio", type=float, default=0.2)
    ap.add_argument("--pseudo_note_mask_fill", type=str, default="zero", choices=["zero", "mean"])
    ap.add_argument("--pseudo_note_onset_only", action="store_true")
    ap.add_argument("--pseudo_note_without_chunk", action="store_true")
    ap.add_argument("--pseudo_repair_order", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
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
    if not pairs:
        raise SystemExit("No pairs found for root/split.")

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
                    model, mels_bt, max_len=int(args.max_len), device=args.device, program_id=int(pid), vocab=vocab
                )
            out_bt, lp_bt = _to_padded_tensors(out_list, lp_list, pad_id=int(vocab.pad), device=mels_bt.device)
            if bool(args.pseudo_repair_order):
                out_bt, lp_bt = canonicalize_pseudo_batch_order(
                    out_bt, lp_bt, vocab=vocab, pad_id=int(vocab.pad), eos_id=int(vocab.eos)
                )

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
                pseudo_note_conf_mode="mask",
                pseudo_note_score_metric="abs_mask_delta",
                pseudo_note_mask_score_metric=str(args.pseudo_note_mask_score_metric),
                pseudo_note_prob_threshold=-9999.0,
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
                if not bool(selected_token_mask[local_i].any().item()):
                    continue
                chunk_idx = b0 + local_i
                t0 = starts[chunk_idx] / float(args.sr)
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

