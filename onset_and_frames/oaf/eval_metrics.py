from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .midi_utils import NoteEvent


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float


def note_onset_prf(
    ref: Sequence[NoteEvent],
    est: Sequence[NoteEvent],
    onset_tolerance: float = 0.05,
) -> PRF:
    """Compute note-onset precision/recall/F1 with ±onset_tolerance.

    This corresponds to the common AMT note metric where only pitch + onset timing are checked.
    """
    # group by pitch
    ref_by: Dict[int, List[float]] = {}
    est_by: Dict[int, List[float]] = {}
    for n in ref:
        ref_by.setdefault(n.pitch, []).append(float(n.start))
    for n in est:
        est_by.setdefault(n.pitch, []).append(float(n.start))

    tp = 0
    total_ref = 0
    total_est = 0

    for pitch, ref_times in ref_by.items():
        est_times = est_by.get(pitch, [])
        ref_times = sorted(ref_times)
        est_times = sorted(est_times)
        total_ref += len(ref_times)
        total_est += len(est_times)

        used = np.zeros(len(ref_times), dtype=bool)

        for et in est_times:
            # candidates within tolerance
            # find closest unused ref time
            best_i = -1
            best_d = float("inf")
            for i, rt in enumerate(ref_times):
                if used[i]:
                    continue
                d = abs(rt - et)
                if d <= onset_tolerance and d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                used[best_i] = True
                tp += 1

    prec = tp / total_est if total_est > 0 else 0.0
    rec = tp / total_ref if total_ref > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return PRF(precision=prec, recall=rec, f1=f1)


def notes_to_frame_roll(
    notes: Sequence[NoteEvent],
    n_pitches: int,
    n_frames: int,
    dt: float,
    midi_min: int = 21,
) -> np.ndarray:
    """Convert notes to a framewise piano-roll (bool) of shape (P,T)."""
    roll = np.zeros((n_pitches, n_frames), dtype=bool)
    for n in notes:
        p = n.pitch - midi_min
        if p < 0 or p >= n_pitches:
            continue
        s = int(np.floor(n.start / dt))
        e = int(np.ceil(n.end / dt))
        s = max(s, 0)
        e = min(e, n_frames)
        if e > s:
            roll[p, s:e] = True
    return roll


def frame_prf(
    ref_roll: np.ndarray,
    est_roll: np.ndarray,
) -> PRF:
    """Framewise precision/recall/F1 on piano-rolls."""
    ref = ref_roll.astype(bool)
    est = est_roll.astype(bool)
    tp = int(np.logical_and(ref, est).sum())
    fp = int(np.logical_and(~ref, est).sum())
    fn = int(np.logical_and(ref, ~est).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return PRF(precision=prec, recall=rec, f1=f1)
