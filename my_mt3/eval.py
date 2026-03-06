# my_mt3/eval.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pretty_midi
import mido
import mir_eval

from my_mt3.dataset import _load_notes_mido


def midi_to_intervals_pitches(
    midi_path: str,
    *,
    use_drums_only: bool = False,
    program: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MIDI -> (intervals [N,2], pitches [N], velocities [N]).
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except ValueError:
        notes = _load_notes_mido(midi_path)
        intervals, pitches, velocities = [], [], []
        for start, end, pitch in notes:
            if use_drums_only:
                continue
            intervals.append([start, end])
            pitches.append(pitch)
            velocities.append(80)
        if not intervals:
            return (
                np.zeros((0, 2), dtype=float),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=int),
            )
        intervals_arr = np.asarray(intervals, dtype=float)
        pitches_arr = np.asarray(pitches, dtype=int)
        velocities_arr = np.asarray(velocities, dtype=int)
        order = np.argsort(intervals_arr[:, 0])
        return intervals_arr[order], pitches_arr[order], velocities_arr[order]

    intervals, pitches, velocities = [], [], []

    for inst in pm.instruments:
        if use_drums_only and not inst.is_drum:
            continue
        if (program is not None) and (not inst.is_drum) and (inst.program != program):
            continue
        if (program is not None) and inst.is_drum:
            continue

        for n in inst.notes:
            intervals.append([n.start, n.end])
            pitches.append(n.pitch)
            velocities.append(n.velocity)

    if len(intervals) == 0:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    intervals = np.asarray(intervals, dtype=float)
    pitches = np.asarray(pitches, dtype=int)
    velocities = np.asarray(velocities, dtype=int)

    order = np.argsort(intervals[:, 0])
    return intervals[order], pitches[order], velocities[order]


def onset_f_measure_with_pitch(
    ref_int: np.ndarray,
    ref_pitch: np.ndarray,
    est_int: np.ndarray,
    est_pitch: np.ndarray,
    *,
    window: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Onset F-measure considering pitch: match onsets per pitch, then micro-average.
    Returns (F, P, R).
    """
    tp = fp = fn = 0

    ref_on = ref_int[:, 0] if len(ref_int) else np.zeros((0,))
    est_on = est_int[:, 0] if len(est_int) else np.zeros((0,))

    pitches = np.union1d(ref_pitch, est_pitch)

    for p in pitches:
        r_times = ref_on[ref_pitch == p]
        e_times = est_on[est_pitch == p]

        matched = mir_eval.util.match_events(r_times, e_times, window=window)

        if isinstance(matched, tuple) and len(matched) == 2:
            k = len(matched[0])
        else:
            k = len(matched)

        tp += k
        fp += (len(e_times) - k)
        fn += (len(r_times) - k)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec


def evaluate_midi_pair(
    ref_midi: str,
    est_midi: str,
    *,
    onset_tolerance: float = 0.05,
    offset_ratio: float | None = 0.2,
    offset_min_tolerance: float = 0.05,
    velocity_tolerance: float = 0.1,
    use_drums_only: bool = False,
    program: int | None = None,
) -> Dict[str, float]:
    """
    Compute standard mir_eval transcription metrics for a single pair.

    Returns dict with keys:
      onset_f, onset_p, onset_r,
      note_f, note_p, note_r,
      note_vel_f, note_vel_p, note_vel_r
    """
    ref_int, ref_pitch, ref_vel = midi_to_intervals_pitches(
        ref_midi, use_drums_only=use_drums_only, program=program,
    )
    est_int, est_pitch, est_vel = midi_to_intervals_pitches(
        est_midi, use_drums_only=use_drums_only, program=program,
    )

    ref_vel_f = (ref_vel.astype(float) / 127.0) if len(ref_vel) else ref_vel.astype(float)
    est_vel_f = (est_vel.astype(float) / 127.0) if len(est_vel) else est_vel.astype(float)

    out: Dict[str, float] = {}

    # 1) Onset-only — mir_eval.onset.f_measure returns (F, P, R)
    f, p, r = mir_eval.onset.f_measure(
        ref_int[:, 0] if len(ref_int) else np.zeros((0,)),
        est_int[:, 0] if len(est_int) else np.zeros((0,)),
        window=onset_tolerance,
    )
    out["onset_f"] = float(f)
    out["onset_p"] = float(p)
    out["onset_r"] = float(r)

    # 2) Onset + Pitch (offset不問)
    f, p, r = onset_f_measure_with_pitch(
        ref_int, ref_pitch, est_int, est_pitch, window=onset_tolerance,
    )
    out["onset_pitch_f"] = float(f)
    out["onset_pitch_p"] = float(p)
    out["onset_pitch_r"] = float(r)

    # 3) Onset + Offset + Pitch (note-based)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pitch,
        est_int, est_pitch,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
    )
    out["note_f"] = float(f)
    out["note_p"] = float(p)
    out["note_r"] = float(r)

    # 4) Onset + Offset + Pitch + Velocity
    p, r, f, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        ref_int, ref_pitch, ref_vel_f,
        est_int, est_pitch, est_vel_f,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
        velocity_tolerance=velocity_tolerance,
    )
    out["note_vel_f"] = float(f)
    out["note_vel_p"] = float(p)
    out["note_vel_r"] = float(r)

    return out


def extract_notes_in_range(
    pm: pretty_midi.PrettyMIDI,
    t0: float,
    t1: float,
    *,
    program: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract notes whose onset falls in [t0, t1) and shift to local time.

    Returns:
      intervals [N,2], pitches [N], velocities [N]  (local time, i.e. start -= t0)
    """
    intervals, pitches, velocities = [], [], []

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if (program is not None) and (inst.program != program):
            continue
        for n in inst.notes:
            if t0 <= n.start < t1:
                intervals.append([n.start - t0, max(n.end - t0, n.start - t0 + 0.001)])
                pitches.append(n.pitch)
                velocities.append(n.velocity)

    if len(intervals) == 0:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    intervals = np.asarray(intervals, dtype=float)
    pitches = np.asarray(pitches, dtype=int)
    velocities = np.asarray(velocities, dtype=int)

    order = np.argsort(intervals[:, 0])
    return intervals[order], pitches[order], velocities[order]


def evaluate_notes_direct(
    ref_int: np.ndarray,
    ref_pitch: np.ndarray,
    ref_vel: np.ndarray,
    est_int: np.ndarray,
    est_pitch: np.ndarray,
    est_vel: np.ndarray,
    *,
    onset_tolerance: float = 0.05,
    offset_ratio: float | None = 0.2,
    offset_min_tolerance: float = 0.05,
    velocity_tolerance: float = 0.1,
) -> Dict[str, float]:
    """
    Same metrics as evaluate_midi_pair but takes numpy arrays directly
    (no file I/O). Useful for chunk-level evaluation.
    """
    ref_vel_f = (ref_vel.astype(float) / 127.0) if len(ref_vel) else ref_vel.astype(float)
    est_vel_f = (est_vel.astype(float) / 127.0) if len(est_vel) else est_vel.astype(float)

    out: Dict[str, float] = {}

    f, p, r = mir_eval.onset.f_measure(
        ref_int[:, 0] if len(ref_int) else np.zeros((0,)),
        est_int[:, 0] if len(est_int) else np.zeros((0,)),
        window=onset_tolerance,
    )
    out["onset_f"] = float(f)
    out["onset_p"] = float(p)
    out["onset_r"] = float(r)

    f, p, r = onset_f_measure_with_pitch(
        ref_int, ref_pitch, est_int, est_pitch, window=onset_tolerance,
    )
    out["onset_pitch_f"] = float(f)
    out["onset_pitch_p"] = float(p)
    out["onset_pitch_r"] = float(r)

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pitch,
        est_int, est_pitch,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
    )
    out["note_f"] = float(f)
    out["note_p"] = float(p)
    out["note_r"] = float(r)

    p, r, f, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        ref_int, ref_pitch, ref_vel_f,
        est_int, est_pitch, est_vel_f,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
        velocity_tolerance=velocity_tolerance,
    )
    out["note_vel_f"] = float(f)
    out["note_vel_p"] = float(p)
    out["note_vel_r"] = float(r)

    return out


def evaluate_directory(
    pairs: List[Tuple[str, str]],
    *,
    onset_tolerance: float = 0.05,
    offset_ratio: float | None = 0.2,
    offset_min_tolerance: float = 0.05,
    velocity_tolerance: float = 0.1,
    use_drums_only: bool = False,
    program: int | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Evaluate a list of (ref_midi, est_midi) pairs.

    Returns:
      per_file: List of per-file metric dicts (same order as pairs)
      summary:  Dict of averaged metrics
    """
    per_file: List[Dict[str, float]] = []

    for ref_path, est_path in pairs:
        m = evaluate_midi_pair(
            ref_path, est_path,
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
            velocity_tolerance=velocity_tolerance,
            use_drums_only=use_drums_only,
            program=program,
        )
        per_file.append(m)

    if not per_file:
        return per_file, {}

    keys = per_file[0].keys()
    summary = {k: float(np.mean([m[k] for m in per_file])) for k in keys}

    return per_file, summary