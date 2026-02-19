import numpy as np
import pretty_midi
import mir_eval

def _midi_to_intervals_pitches_velocities(
    midi_path: str,
    *,
    use_drums_only: bool = False,
    program: int | None = None,
):
    """
    MIDI -> (intervals [N,2], pitches [N], velocities [N])
    - use_drums_only=True なら drum トラックのみ抽出
    - program を指定すると、そのprogramの楽器のみ抽出（drumは除く）
    """
    pm = pretty_midi.PrettyMIDI(midi_path)

    intervals = []
    pitches = []
    velocities = []

    for inst in pm.instruments:
        if use_drums_only and not inst.is_drum:
            continue
        if (program is not None) and (not inst.is_drum) and (inst.program != program):
            continue
        if (program is not None) and inst.is_drum:
            # program指定時は通常、ドラムは混ぜない方が安全
            continue

        for n in inst.notes:
            intervals.append([n.start, n.end])
            pitches.append(n.pitch)
            velocities.append(n.velocity)

    if len(intervals) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)

    intervals = np.asarray(intervals, dtype=float)
    pitches = np.asarray(pitches, dtype=int)
    velocities = np.asarray(velocities, dtype=int)

    # 安定のため start順にソート
    order = np.argsort(intervals[:, 0])
    return intervals[order], pitches[order], velocities[order]


def evaluate_midi_transcription(
    ref_midi: str,
    est_midi: str,
    *,
    onset_tolerance: float = 0.05,   # 50ms
    offset_tolerance: float = 0.05,  # 50ms（慣習で20%等もある）
    velocity_tolerance: float = 0.1, # mir_evalは velocity を [0,1]想定にすることが多い
    use_drums_only: bool = False,
    program: int | None = None,
):
    """
    代表的な mir_eval 指標をまとめて計算して dict で返す。
    """
    ref_int, ref_pitch, ref_vel = _midi_to_intervals_pitches_velocities(
        ref_midi, use_drums_only=use_drums_only, program=program
    )
    est_int, est_pitch, est_vel = _midi_to_intervals_pitches_velocities(
        est_midi, use_drums_only=use_drums_only, program=program
    )

    # velocity を [0,1] に正規化（mir_eval側の想定に合わせる）
    ref_vel_f = (ref_vel.astype(float) / 127.0) if len(ref_vel) else ref_vel.astype(float)
    est_vel_f = (est_vel.astype(float) / 127.0) if len(est_vel) else est_vel.astype(float)

    out = {}

    # 1) Onset-only（transcriptionでも offset無視版が欲しいなら onset モジュールも使える）
    # mir_eval.transcription の offset_tolerance を大きくする方法もあるが、ここでは onset専用を使う
    p, r, f, _ = mir_eval.onset.f_measure(
        ref_int[:, 0] if len(ref_int) else np.zeros((0,)),
        est_int[:, 0] if len(est_int) else np.zeros((0,)),
        window=onset_tolerance,
    )
    out["onset_f"] = float(f)
    out["onset_p"] = float(p)
    out["onset_r"] = float(r)

    # 2) Onset+Offset+Pitch（標準のnote-based）
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pitch,
        est_int, est_pitch,
        onset_tolerance=onset_tolerance,
        offset_tolerance=offset_tolerance,
        offset_ratio=None,   # ratio(例:0.2)を使う場合はここを設定しoffset_toleranceをNoneに
    )
    out["note_f"] = float(f)
    out["note_p"] = float(p)
    out["note_r"] = float(r)

    # 3) Onset+Offset+Pitch+Velocity
    # transcription_velocity は velocity_tolerance を使う（[0,1]スケール）
    p, r, f, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        ref_int, ref_pitch, ref_vel_f,
        est_int, est_pitch, est_vel_f,
        onset_tolerance=onset_tolerance,
        offset_tolerance=offset_tolerance,
        velocity_tolerance=velocity_tolerance,
        offset_ratio=None,
    )
    out["note_vel_f"] = float(f)
    out["note_vel_p"] = float(p)
    out["note_vel_r"] = float(r)

    return out


if __name__ == "__main__":
    metrics = evaluate_midi_transcription(
        ref_midi="ref.mid",
        est_midi="est.mid",
        onset_tolerance=0.05,
        offset_tolerance=0.05,
        use_drums_only=False,   # ドラムだけなら True
        program=None,           # 特定楽器だけなら例: program=0 (Acoustic Grand Piano)
    )
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
