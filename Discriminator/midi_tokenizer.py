from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import random

import torch
import pretty_midi


@dataclass
class MidiTokCfg:
    # Token IDs
    pad_id: int = 0
    cls_id: int = 1

    # Time shift tokenization (fixed grid)
    # One time_shift token represents `time_step_sec` seconds.
    time_step_sec: float = 0.01  # 10ms
    max_shift_steps: int = 100   # max token represents 100*10ms = 1.0s gap

    # Vocab layout (computed)
    # time_shift: [2 .. 2+max_shift_steps-1]
    # note_on:    next 128
    # note_off:   next 128
    # [MASK]:     last token (for MLM pre-training)
    # total vocab = 2 + max_shift_steps + 128 + 128 + 1
    @property
    def mask_id(self) -> int:
        return 2 + self.max_shift_steps + 128 + 128

    def vocab_size(self) -> int:
        return 2 + self.max_shift_steps + 128 + 128 + 1

    def time_shift_id(self, steps_1_to_max: int) -> int:
        # steps is 1..max_shift_steps
        assert 1 <= steps_1_to_max <= self.max_shift_steps
        return 2 + (steps_1_to_max - 1)

    def note_on_id(self, pitch: int) -> int:
        assert 0 <= pitch <= 127
        base = 2 + self.max_shift_steps
        return base + pitch

    def note_off_id(self, pitch: int) -> int:
        assert 0 <= pitch <= 127
        base = 2 + self.max_shift_steps + 128
        return base + pitch


@dataclass
class AugCfg:
    # pitch shift in semitones (uniform integer in [min,max])
    pitch_shift_min: int = -5
    pitch_shift_max: int = 5
    pitch_shift_prob: float = 0.8

    # time scaling factor (uniform in [min,max])
    time_scale_min: float = 0.9
    time_scale_max: float = 1.1
    time_scale_prob: float = 0.8


def load_piano_notes(midi_path: str) -> List[pretty_midi.Note]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes: List[pretty_midi.Note] = []
    for inst in pm.instruments:
        # piano only assumption: is_drum should be False
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    # fallback: if empty, still return empty list
    return notes


def apply_augmentation(
    notes: List[pretty_midi.Note],
    *,
    aug: AugCfg,
) -> List[pretty_midi.Note]:
    """
    Apply:
      - pitch shift (parallel transpose)
      - uniform time scaling for start/end (all notes equally)
    """
    if len(notes) == 0:
        return []

    # decide aug params
    do_pitch = random.random() < aug.pitch_shift_prob
    do_time = random.random() < aug.time_scale_prob

    shift = 0
    if do_pitch:
        shift = random.randint(aug.pitch_shift_min, aug.pitch_shift_max)

    scale = 1.0
    if do_time:
        scale = random.uniform(aug.time_scale_min, aug.time_scale_max)

    out: List[pretty_midi.Note] = []
    for n in notes:
        p = int(n.pitch) + shift
        if p < 0 or p > 127:
            # drop out-of-range notes (simplest)
            continue
        s = float(n.start) * scale
        e = float(n.end) * scale
        if e <= s:
            continue
        out.append(pretty_midi.Note(
            velocity=int(n.velocity),
            pitch=p,
            start=s,
            end=e,
        ))
    return out


def midi_notes_to_tokens(
    notes: List[pretty_midi.Note],
    tok: MidiTokCfg,
) -> torch.Tensor:
    """
    Convert piano notes to event tokens:
      time_shift(Δt), note_on(p), note_off(p)
    Notes are converted into events at start/end times.
    """
    if len(notes) == 0:
        return torch.zeros((0,), dtype=torch.long)

    # build events
    # type_order: note_off before note_on at same time (common)
    events: List[Tuple[float, int, int]] = []  # (time, type_order, token_id)
    for n in notes:
        p = int(n.pitch)
        events.append((float(n.start), 1, tok.note_on_id(p)))
        events.append((float(n.end),   0, tok.note_off_id(p)))

    events.sort(key=lambda x: (x[0], x[1]))

    # walk time
    out: List[int] = []
    t_prev = events[0][0]
    for t, _, ev_tok in events:
        dt = max(0.0, t - t_prev)
        steps = int(round(dt / tok.time_step_sec))

        # emit time shifts (can require multiple tokens if gap > max)
        while steps > 0:
            s = min(steps, tok.max_shift_steps)
            out.append(tok.time_shift_id(s))
            steps -= s

        out.append(ev_tok)
        t_prev = t

    return torch.tensor(out, dtype=torch.long)
