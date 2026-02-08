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


GM_TO_9: Dict[int, int] = {
    35: 0, 36: 0,
    38: 1, 40: 1, 37: 1, 39: 1,
    42: 2, 44: 2, 46: 3,
    41: 4, 43: 4,
    45: 5, 47: 5,
    48: 6, 50: 6,
    51: 7, 53: 7, 59: 7,
    49: 8, 52: 8, 55: 8, 57: 8,
}

CLASS_TO_GM: List[int] = [36, 38, 42, 46, 41, 45, 48, 51, 49]


@dataclass
class InferCfg:
    T: int = 1024
    hop_sec: float = 0.01
    hop_frames: int = 512
    thresh: float = 0.5
    overlap_max: bool = False
    min_gap_frames: int = 1
    note_len_sec: float = 0.05


def load_drum_notes(pm: pretty_midi.PrettyMIDI):
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            notes.extend(inst.notes)
    if len(notes) == 0:
        for inst in pm.instruments:
            notes.extend(inst.notes)
    return notes


def midi_to_roll(midi_path: Path, hop_sec: float, K: int) -> Tuple[torch.Tensor, float]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = load_drum_notes(pm)
    if len(notes) == 0:
        return torch.zeros(1, K, dtype=torch.float32), 0.0

    end_time = max(n.end for n in notes)
    total_frames = max(1, int(np.ceil(end_time / hop_sec)))
    roll = torch.zeros(total_frames, K, dtype=torch.float32)
    for n in notes:
        k = GM_TO_9.get(n.pitch, None)
        if k is None or k >= K:
            continue
        t = int(round(n.start / hop_sec))
        if 0 <= t < total_frames:
            roll[t, k] = 1.0
    return roll, end_time


def chunk_indices(total_frames: int, T: int, hop_frames: int):
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
def denoise_roll_fullsong(model: DrumDenoiseVQVAE, roll_noisy: torch.Tensor, cfg: InferCfg, device: str) -> torch.Tensor:
    model.eval()

    F, K = roll_noisy.shape
    T = cfg.T
    hop_frames = cfg.hop_frames if cfg.hop_frames > 0 else T // 2

    acc = torch.zeros(F, K, dtype=torch.float32)
    wts = torch.zeros(F, 1, dtype=torch.float32)

    for (s, e) in chunk_indices(F, T, hop_frames):
        x = roll_noisy[s:e]
        if (e - s) < T:
            pad = torch.zeros(T - (e - s), K, dtype=torch.float32)
            x = torch.cat([x, pad], dim=0)

        x = x.unsqueeze(0).to(device)  # (1,T,K)
        out = model.forward_phase1(x)  # denoiser only
        logits = out["y_logits"][0]    # (T,K)
        probs = torch.sigmoid(logits).cpu()[: (e - s)]

        if cfg.overlap_max:
            acc[s:e] = torch.maximum(acc[s:e], probs)
            wts[s:e] = 1.0
        else:
            acc[s:e] += probs
            wts[s:e] += 1.0

    if not cfg.overlap_max:
        acc = acc / torch.clamp(wts, min=1.0)
    return acc


def suppress_min_gap(onsets: torch.Tensor, min_gap: int) -> torch.Tensor:
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


def roll_to_midi(onsets01: torch.Tensor, hop_sec: float, note_len_sec: float) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="denoised_drums")
    F, K = onsets01.shape
    for t in range(F):
        for k in range(K):
            if onsets01[t, k] > 0.5:
                start = t * hop_sec
                end = start + note_len_sec
                inst.notes.append(pretty_midi.Note(velocity=100, pitch=CLASS_TO_GM[k], start=float(start), end=float(end)))
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
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--midi", type=str, required=True)
    ap.add_argument("--out_midi", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--hop_sec", type=float, default=0.01)
    ap.add_argument("--hop_frames", type=int, default=512)
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--overlap_max", action="store_true")
    ap.add_argument("--min_gap_frames", type=int, default=1)
    ap.add_argument("--note_len_sec", type=float, default=0.05)
    args = ap.parse_args()

    device = args.device
    K = 9

    cfg = DrumVQCfg(n_drums=K, latent_dim=256, n_codes=512, n_layers=3, ema_decay=0.99)
    model = load_model(Path(args.ckpt), cfg, device)

    infer_cfg = InferCfg(
        T=args.T,
        hop_sec=args.hop_sec,
        hop_frames=args.hop_frames,
        thresh=args.thresh,
        overlap_max=args.overlap_max,
        min_gap_frames=args.min_gap_frames,
        note_len_sec=args.note_len_sec,
    )

    roll_noisy, end_time = midi_to_roll(Path(args.midi), hop_sec=infer_cfg.hop_sec, K=K)
    print(f"[INFO] roll frames={roll_noisy.shape[0]} sec~{end_time:.2f}")

    probs = denoise_roll_fullsong(model, roll_noisy, infer_cfg, device)
    breakpoint()
    onsets = (probs >= infer_cfg.thresh).float()
    onsets = suppress_min_gap(onsets, infer_cfg.min_gap_frames)

    pm_out = roll_to_midi(onsets, hop_sec=infer_cfg.hop_sec, note_len_sec=infer_cfg.note_len_sec)
    out_path = Path(args.out_midi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm_out.write(str(out_path))
    print("[OK] wrote:", out_path)


if __name__ == "__main__":
    main()
