from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import DecodeConfig
from .midi_utils import NoteEvent

try:
    import pretty_midi
except Exception:  # pragma: no cover
    pretty_midi = None


def decode_notes(
    onset: torch.Tensor,
    frame: torch.Tensor,
    cfg: DecodeConfig,
    hop_length: int = 512,
    sample_rate: int = 16000,
    velocity: Optional[torch.Tensor] = None,
    midi_min: int = 21,
) -> List[NoteEvent]:
    """Decode framewise probabilities into note events.

    Implements the core O&F inference idea:
      - binarize onset/frame predictions
      - do NOT allow a new note to start unless onset agrees in that frame
        (onset gating), as described in the O&F paper.

    Args:
        onset/frame: (P,T) or (1,P,T) or (B,P,T) - if batch, only first item is used
    """
    if onset.dim() == 3:
        onset = onset[0]
    if frame.dim() == 3:
        frame = frame[0]
    if velocity is not None and velocity.dim() == 3:
        velocity = velocity[0]

    P, T = onset.shape
    dt = hop_length / sample_rate

    onset_bin = (onset >= cfg.tau_on).cpu().numpy()
    frame_bin = (frame >= cfg.tau_fr).cpu().numpy()

    notes: List[NoteEvent] = []

    for p in range(P):
        active = False
        start_idx = 0

        for t in range(T):
            if not active:
                if cfg.onset_gating:
                    cond = onset_bin[p, t] and frame_bin[p, t]
                else:
                    cond = frame_bin[p, t]
                if cond:
                    active = True
                    start_idx = t
            else:
                # if frame goes inactive, close note
                if not frame_bin[p, t]:
                    end_idx = t
                    start = start_idx * dt
                    end = end_idx * dt
                    if end > start:
                        vel = 64
                        if velocity is not None:
                            v = float(torch.clamp(velocity[p, start_idx], 0.0, 1.0).item())
                            vel = int(round(80.0 * v + 10.0))  # O&F mapping
                            vel = int(np.clip(vel, 1, 127))
                        notes.append(NoteEvent(pitch=int(midi_min + p), start=float(start), end=float(end), velocity=int(vel)))
                    active = False

        # flush active note
        if active:
            start = start_idx * dt
            end = T * dt
            if end > start:
                vel = 64
                if velocity is not None:
                    v = float(torch.clamp(velocity[p, start_idx], 0.0, 1.0).item())
                    vel = int(round(80.0 * v + 10.0))
                    vel = int(np.clip(vel, 1, 127))
                notes.append(NoteEvent(pitch=int(midi_min + p), start=float(start), end=float(end), velocity=int(vel)))

    # sort notes by start time
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def notes_to_pretty_midi(notes: Sequence[NoteEvent], program: int = 0) -> "pretty_midi.PrettyMIDI":
    if pretty_midi is None:  # pragma: no cover
        raise ImportError("pretty_midi is required for MIDI writing. Install with: pip install pretty_midi")

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=int(n.velocity),
            pitch=int(n.pitch),
            start=float(n.start),
            end=float(n.end),
        ))
    pm.instruments.append(inst)
    return pm


def save_notes_as_midi(notes: Sequence[NoteEvent], out_path: Union[str, Path], program: int = 0) -> None:
    pm = notes_to_pretty_midi(notes, program=program)
    pm.write(str(out_path))
