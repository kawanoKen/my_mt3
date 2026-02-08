#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import pretty_midi

from vq_vae import DrumVQCfg, DrumDenoiseVQVAE


# ----------------------------
# Mapping: GM drum note -> 9 classes (same as training)
# ----------------------------
GM_TO_9: Dict[int, int] = {
    # Kick
    35: 0, 36: 0,
    # Snare (+ side/hand clap)
    38: 1, 40: 1, 37: 1, 39: 1,
    # Hi-hat
    42: 2, 44: 2,  # closed/pedal
    46: 3,         # open
    # Toms
    41: 4, 43: 4,  # low
    45: 5, 47: 5,  # mid
    48: 6, 50: 6,  # high
    # Ride
    51: 7, 53: 7, 59: 7,
    # Crash
    49: 8, 52: 8, 55: 8, 57: 8,
}

# class -> representative GM pitch (you can change if you prefer)
CLASS_TO_GM: List[int] = [
    36,  # kick
    38,  # snare
    42,  # closed hh
    46,  # open hh
    41,  # low tom
    45,  # mid tom
    48,  # high tom
    51,  # ride
    49,  # crash
]


@dataclass
class InferCfg:
    T: int = 1024
    hop_sec: float = 0.01      # must match how you formed rolls in training
    hop_frames: int = 512      # sliding step in frames (overlap). default: T//2
    thresh: float = 0.5
    # If True, use max pooling in overlap. If False, use mean pooling.
    overlap_max: bool = False
    # post-process
    min_gap_frames: int = 1    # suppress immediate repeats per drum
    note_len_sec: float = 0.05 # fixed note length for written MIDI notes


def load_drum_notes(pm: pretty_midi.PrettyMIDI) -> List[pretty_midi.Note]:
    notes: List[pretty_midi.Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            notes.extend(inst.notes)
    # fallback (rare): if no inst.is_drum found, try take all notes and map pitches
    if len(notes) == 0:
        for inst in pm.instruments:
            notes.extend(inst.notes)
    return notes


def midi_to_roll(midi_path: Path, hop_sec: float, K: int) -> Tuple[torch.Tensor, float]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = load_drum_notes(pm)
    if len(notes) == 0:
        # empty -> 1 frame
        return torch.zeros(1, K, dtype=torch.float32), 0.0

    end_time = max(n.end for n in notes)
    total_frames = int(np.ceil(end_time / hop_sec))
    total_frames = max(1, total_frames)

    roll = torch.zeros(total_frames, K, dtype=torch.float32)
    for n in notes:
        k = GM_TO_9.get(n.pitch, None)
        if k is None or k >= K:
            continue
        t = int(round(n.start / hop_sec))
        if 0 <= t < total_frames:
            roll[t, k] = 1.0

    return roll, end_time


def chunk_indices(total_frames: int, T: int, hop_frames: int) -> List[Tuple[int, int]]:
    if total_frames <= T:
        return [(0, total_frames)]
    out = []
    s = 0
    while s < total_frames:
        e = min(s + T, total_frames)
        out.append((s, e))
        if e == total_frames:
            break
        s += hop_frames
    return out


@torch.no_grad()
def denoise_roll_fullsong(
    model: DrumDenoiseVQVAE,
    roll_noisy: torch.Tensor,  # (F, K)
    cfg: InferCfg,
    device: str,
) -> torch.Tensor:
    """
    returns probs: (F, K) in [0,1]
    """
    model.eval()

    F, K = roll_noisy.shape
    T = cfg.T
    hop_frames = cfg.hop_frames if cfg.hop_frames > 0 else T // 2

    # output accumulator
    acc = torch.zeros(F, K, dtype=torch.float32)
    wts = torch.zeros(F, 1, dtype=torch.float32)

    # slide windows
    for (s, e) in chunk_indices(F, T, hop_frames):
        x = roll_noisy[s:e]  # (len, K)

        # pad to T
        if (e - s) < T:
            pad = torch.zeros(T - (e - s), K, dtype=torch.float32)
            x = torch.cat([x, pad], dim=0)

        # model expects (B,T,K)
        x = x.unsqueeze(0).to(device)  # (1,T,K)
        out = model(x)

        # choose which output to use
        # If you keep your old forward where y_logits==y_anchor, either is same.
        # If you applied the recommended fix (y_logits from z_seq, anchor from c), use y_logits here.
        logits = out["y_logits"][0]  # (T,K)
        probs = torch.sigmoid(logits).detach().cpu()

        # crop back to original length
        probs = probs[: (e - s)]

        if cfg.overlap_max:
            # max pooling overlap
            acc[s:e] = torch.maximum(acc[s:e], probs)
            wts[s:e] = 1.0
        else:
            # mean pooling overlap
            acc[s:e] += probs
            wts[s:e] += 1.0

    if not cfg.overlap_max:
        acc = acc / torch.clamp(wts, min=1.0)

    return acc


def suppress_min_gap(onsets: torch.Tensor, min_gap: int) -> torch.Tensor:
    """
    onsets: (F,K) 0/1
    suppress repeated hits within min_gap frames per drum class
    """
    if min_gap <= 0:
        return onsets
    F, K = onsets.shape
    out = onsets.clone()
    for k in range(K):
        last = -10**9
        for t in range(F):
            if out[t, k] > 0.5:
                if (t - last) <= min_gap:
                    out[t, k] = 0.0
                else:
                    last = t
    return out


def roll_to_midi(
    onsets01: torch.Tensor,   # (F,K) 0/1
    hop_sec: float,
    note_len_sec: float,
    class_to_gm: List[int],
) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="denoised_drums")

    F, K = onsets01.shape
    for t in range(F):
        for k in range(K):
            if onsets01[t, k] > 0.5:
                start = t * hop_sec
                end = start + note_len_sec
                pitch = class_to_gm[k]
                inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=float(start), end=float(end)))

    pm.instruments.append(inst)
    return pm


def load_model(ckpt_path: Path, cfg: DrumVQCfg, device: str) -> DrumDenoiseVQVAE:
    model = DrumDenoiseVQVAE(cfg).to(device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt (state_dict or {'model':...})")
    ap.add_argument("--midi", type=str, required=True, help="input noisy midi path")
    ap.add_argument("--out_midi", type=str, required=True, help="output denoised midi path")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--hop_sec", type=float, default=0.01)
    ap.add_argument("--hop_frames", type=int, default=512)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--overlap_max", action="store_true")
    ap.add_argument("--min_gap_frames", type=int, default=1)
    ap.add_argument("--note_len_sec", type=float, default=0.05)

    args = ap.parse_args()

    device = args.device
    in_midi = Path(args.midi)
    out_midi = Path(args.out_midi)
    ckpt = Path(args.ckpt)

    # model cfg must match training
    K = 9
    model_cfg = DrumVQCfg(
        n_drums=K,
        latent_dim=256,
        n_codes=512,
        n_layers=3,
        ema_decay=0.99,
        beta_commit=0.05,
        gamma_anchor=0.2,
    )
    model = load_model(ckpt, model_cfg, device)

    infer_cfg = InferCfg(
        T=args.T,
        hop_sec=args.hop_sec,
        hop_frames=args.hop_frames,
        thresh=args.thresh,
        overlap_max=args.overlap_max,
        min_gap_frames=args.min_gap_frames,
        note_len_sec=args.note_len_sec,
    )

    # midi -> noisy roll
    roll_noisy, end_time = midi_to_roll(in_midi, hop_sec=infer_cfg.hop_sec, K=K)
    print(f"[INFO] loaded roll: frames={roll_noisy.shape[0]}, K={roll_noisy.shape[1]}, sec~{end_time:.2f}")

    # denoise -> probs
    probs = denoise_roll_fullsong(model, roll_noisy, infer_cfg, device)

    # threshold -> onsets
    onsets = (probs >= infer_cfg.thresh).float()
    onsets = suppress_min_gap(onsets, infer_cfg.min_gap_frames)

    # write midi
    pm_out = roll_to_midi(onsets, hop_sec=infer_cfg.hop_sec, note_len_sec=infer_cfg.note_len_sec, class_to_gm=CLASS_TO_GM)
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm_out.write(str(out_midi))
    print("[INFO] wrote:", out_midi)


if __name__ == "__main__":
    main()
