from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mido


@dataclass
class NoteEvent:
    pitch: int          # MIDI pitch (0-127)
    start: float        # seconds
    end: float          # seconds
    velocity: int = 64  # 1-127


def _get_sustain_intervals_from_events(
    cc_events: Sequence[Tuple[float, int]],
    threshold: int = 64,
) -> List[Tuple[float, float]]:
    """Return sustain pedal ON intervals as (start,end) seconds."""
    ccs = sorted(cc_events, key=lambda x: x[0])

    intervals: List[Tuple[float, float]] = []
    on_t: Optional[float] = None
    sustain_on = False

    for t_sec, value in ccs:
        is_on = value >= threshold
        if is_on and not sustain_on:
            sustain_on = True
            on_t = t_sec
        elif (not is_on) and sustain_on:
            sustain_on = False
            if on_t is not None and t_sec > on_t:
                intervals.append((on_t, t_sec))
            on_t = None

    # If sustain never turned off, leave it open-ended (caller may clamp by audio length)
    if sustain_on and on_t is not None:
        intervals.append((on_t, float("inf")))

    return intervals


def _extend_notes_with_sustain(
    notes: Sequence[NoteEvent],
    sustain_intervals: Sequence[Tuple[float, float]],
) -> List[NoteEvent]:
    """Extend note end-times if sustain pedal is pressed at note end.

    Based on the description in Onsets and Frames: if sustain is on, extend until sustain off
    or the same note (same pitch) is played again.
    """
    # Prepare next note start per pitch
    notes_sorted = sorted(notes, key=lambda n: (n.start, n.pitch))
    by_pitch: Dict[int, List[NoteEvent]] = {}
    for n in notes_sorted:
        by_pitch.setdefault(n.pitch, []).append(n)

    # Build mapping from note id to next same-pitch start
    next_start: Dict[int, float] = {}
    for pitch, ns in by_pitch.items():
        for i, n in enumerate(ns):
            nxt = ns[i + 1].start if i + 1 < len(ns) else float("inf")
            next_start[id(n)] = nxt

    def find_interval_end(t: float) -> Optional[float]:
        for s, e in sustain_intervals:
            if s <= t <= e:
                return e
        return None

    out: List[NoteEvent] = []
    for n in notes_sorted:
        end = n.end
        sustain_end = find_interval_end(end)
        if sustain_end is not None and sustain_end != float("inf"):
            # extend, but not beyond next note of same pitch
            end = min(sustain_end, next_start.get(id(n), float("inf")))
        elif sustain_end == float("inf"):
            # open-ended sustain: extend to next note of same pitch, or keep as-is if none
            end = min(next_start.get(id(n), float("inf")), end if end != float("inf") else end)

        out.append(NoteEvent(pitch=n.pitch, start=float(n.start), end=float(end), velocity=int(n.velocity)))
    return out


def _load_midi_notes_mido(
    midi_path: Union[str, Path],
) -> List[NoteEvent]:
    """Load MIDI with mido and return note events + sustain-CC events.

    Notes are parsed from merged tracks with tempo handling.
    """
    mid = mido.MidiFile(str(midi_path))
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000  # default 120 BPM
    t_sec = 0.0

    # key: (channel, note) -> stack[(start_sec, velocity)]
    active: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
    notes: List[NoteEvent] = []
    cc64_events: List[Tuple[float, int]] = []

    for msg in mido.merge_tracks(mid.tracks):
        if msg.time:
            t_sec += mido.tick2second(msg.time, ticks_per_beat, tempo)

        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue

        if msg.type == "control_change" and msg.control == 64:
            cc64_events.append((float(t_sec), int(msg.value)))
            continue

        if msg.type == "note_on" and msg.velocity > 0:
            key = (int(getattr(msg, "channel", 0)), int(msg.note))
            active.setdefault(key, []).append((float(t_sec), int(msg.velocity)))
            continue

        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (int(getattr(msg, "channel", 0)), int(msg.note))
            lst = active.get(key)
            if not lst:
                continue
            start_sec, vel = lst.pop()
            end_sec = float(t_sec)
            if end_sec <= start_sec:
                end_sec = start_sec + 1e-4
            notes.append(
                NoteEvent(
                    pitch=int(msg.note),
                    start=float(start_sec),
                    end=float(end_sec),
                    velocity=int(max(1, min(127, vel))),
                )
            )

    return notes, cc64_events


def load_midi_notes(
    midi_path: Union[str, Path],
    apply_sustain: bool = True,
    sustain_cc: int = 64,
    sustain_threshold: int = 64,
) -> List[NoteEvent]:
    """Load a MIDI file and return note events (optionally applying sustain pedal).

    Uses mido parser so files with very large tick counts are handled robustly.
    """
    notes, cc64_events = _load_midi_notes_mido(midi_path)
    if not apply_sustain:
        return sorted(notes, key=lambda n: (n.start, n.pitch))

    if sustain_cc != 64:
        # mido path currently collects CC64 only; keep behavior deterministic.
        cc64_events = []
    sustain_intervals = _get_sustain_intervals_from_events(cc64_events, threshold=sustain_threshold)
    out = _extend_notes_with_sustain(notes, sustain_intervals)
    return sorted(out, key=lambda n: (n.start, n.pitch))
