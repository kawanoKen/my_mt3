import argparse
import subprocess
import tempfile
import os
from pathlib import Path

import mido


DRUM_CH = 9  # MIDIのCh10 (0-indexed)


def rewrite_midi_all_drums(src_midi: Path, dst_midi: Path, program: int = 0) -> None:
    import mido

    DRUM_CH = 9
    mid = mido.MidiFile(str(src_midi))
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    for tr in mid.tracks:
        new_tr = mido.MidiTrack()

        # トラック先頭に program_change を置く（Bankは入れない）
        new_tr.append(mido.Message("program_change", channel=DRUM_CH, program=program, time=0))

        for msg in tr:
            msg = msg.copy()

            if msg.is_meta:
                new_tr.append(msg)
                continue

            if hasattr(msg, "channel"):
                msg.channel = DRUM_CH

            # Bank Select (CC0/32) は削除（不要・干渉防止）
            if msg.type == "control_change" and msg.control in (0, 32):
                continue

            # program_change も固定
            if msg.type == "program_change":
                msg.program = program

            new_tr.append(msg)

        out.tracks.append(new_tr)

    out.save(str(dst_midi))


def midi_to_wav(
    midi_path: str | Path,
    wav_path: str | Path,
    soundfont: str | Path,
    sr: int = 44100,
    force_drums: bool = False,
) -> None:
    midi_path = Path(midi_path)
    wav_path = Path(wav_path)
    soundfont = Path(soundfont)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # force_drums のときは MIDI を書き換えた一時ファイルを使う
    tmp_midi_path = None
    if force_drums:
        fd, p = tempfile.mkstemp(suffix=".mid")
        os.close(fd)
        tmp_midi_path = Path(p)
        rewrite_midi_all_drums(midi_path, tmp_midi_path, program=0)
        midi_for_render = tmp_midi_path
    else:
        midi_for_render = midi_path

    try:
        args = [
            "fluidsynth",
            "-ni",
            str(soundfont),
            str(midi_for_render),
            "-F",
            str(wav_path),
            "-r",
            str(sr),
        ]
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    finally:
        if tmp_midi_path and tmp_midi_path.exists():
            tmp_midi_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description="MIDI -> WAV (fluidsynth)")
    default_sf2 = Path(__file__).resolve().parents[1] / "soundfont" / "SGM-V2.01.sf2"

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--midi", help="入力MIDIファイル")
    g.add_argument("--midi_dir", help="入力MIDIディレクトリ（再帰）")

    ap.add_argument("--sf2", default=str(default_sf2), help="SoundFont(.sf2)")
    ap.add_argument("--out", help="出力WAV（--midi時）")
    ap.add_argument("--out_dir", help="出力ディレクトリ（--midi_dir時）")
    ap.add_argument("--sr", type=int, default=44100, help="サンプリングレート")
    ap.add_argument("--drums", action="store_true", help="どんなMIDIでも強制ドラム再生")

    args = ap.parse_args()

    if args.midi:
        midi = Path(args.midi)
        out = Path(args.out) if args.out else midi.with_suffix(".wav")
        midi_to_wav(midi, out, args.sf2, sr=args.sr, force_drums=args.drums)
        return

    root = Path(args.midi_dir)
    out_base = Path(args.out_dir) if args.out_dir else root.with_name(root.name + "_wavs")

    for ext in ["*.mid", "*.midi"]:
        for midi in root.rglob(ext):
            rel = midi.relative_to(root)
            wav = out_base.joinpath(rel).with_suffix(".wav")
            midi_to_wav(midi, wav, args.sf2, sr=args.sr, force_drums=args.drums)


if __name__ == "__main__":
    main()
