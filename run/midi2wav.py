from midi2audio import FluidSynth
from pathlib import Path
import argparse
import os

# https://drive.usercontent.google.com/download?id=1UJ1mrY2l_C_YbKeyywNUymBz7OTVzQLU&export=download&authuser=0
# こちらでsfをダウンロード

def convert_midi_to_wav(midi_file, soundfont_file, output_wav):
    """
    MIDIファイルをWAVに変換する
    
    Args:
        midi_file (str): 入力するMIDIファイルのパス
        soundfont_file (str): 使用するSoundFont(.sf2)のパス
        output_wav (str): 出力するWAVファイルのパス
    """
    # FluidSynthのインスタンスを作成（SoundFontを指定）
    fs = FluidSynth(soundfont_file)
    
    # 変換実行
    print(f"変換中: {midi_file} ...")
    fs.midi_to_audio(midi_file, output_wav)
    print(f"完了: {output_wav}")

def midi_to_wav_with_fs(fs: FluidSynth, midi_path: Path, wav_path: Path) -> None:
    print(f"変換中: {midi_path} ...")
    fs.midi_to_audio(str(midi_path), str(wav_path))
    print(f"完了: {wav_path}")

def find_midi_files(root: Path, recursive: bool):
    patterns = ["*.mid", "*.midi"]
    if recursive:
        for pat in patterns:
            yield from root.rglob(pat)
    else:
        for pat in patterns:
            yield from root.glob(pat)

def main():
    parser = argparse.ArgumentParser(
        description="MIDIをWAVへ変換（単一ファイル or ディレクトリ対応）"
    )
    default_sf2 = (Path(__file__).resolve().parents[1] / "GeneralUser-GS" / "GeneralUser-GS.sf2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--midi", help="変換するMIDIファイルのパス")
    group.add_argument("--midi_dir", help="変換するMIDIファイルが入ったディレクトリ")
    parser.add_argument(
        "--sf2",
        default=str(default_sf2),
        help=f"使用するSoundFont(.sf2)のパス（既定: {default_sf2}）"
    )
    parser.add_argument("--out", help="出力WAVファイルのパス（--midi指定時のみ）")
    parser.add_argument("--out_dir", help="出力先ディレクトリ（--midi_dir指定時）")
    parser.add_argument("--recursive", action="store_true", help="サブディレクトリも再帰的に処理")
    parser.add_argument("--overwrite", action="store_true", help="既存のWAVがあっても上書きする")
    args = parser.parse_args()

    sf2_path = Path(args.sf2)
    if not sf2_path.is_file():
        raise SystemExit(
            f"SoundFontが見つかりません: {sf2_path}\n"
            f"--sf2 で有効な .sf2 を指定してください。"
        )

    if args.midi:
        midi_path = Path(args.midi)
        if not midi_path.is_file():
            raise SystemExit(f"ファイルが見つかりません: {midi_path}")
        out_path = Path(args.out) if args.out else midi_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        convert_midi_to_wav(str(midi_path), args.sf2, str(out_path))
        return

    # ディレクトリ処理
    dir_path = Path(args.midi_dir)
    if not dir_path.is_dir():
        raise SystemExit(f"ディレクトリが見つかりません: {dir_path}")

    out_base = Path(args.out_dir) if args.out_dir else None
    if out_base:
        out_base.mkdir(parents=True, exist_ok=True)

    fs = FluidSynth(args.sf2)
    converted = 0
    for midi_file in find_midi_files(dir_path, args.recursive):
        rel = midi_file.relative_to(dir_path)
        wav_path = (
            out_base.joinpath(rel).with_suffix(".wav")
            if out_base
            else midi_file.with_suffix(".wav")
        )
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        if wav_path.exists() and not args.overwrite:
            print(f"スキップ（既存）: {wav_path}")
            continue
        try:
            midi_to_wav_with_fs(fs, midi_file, wav_path)
            converted += 1
        except Exception as e:
            print(f"失敗: {midi_file} -> {e}")

    print(f"処理完了: {converted} ファイル変換")

if __name__ == "__main__":
    main()