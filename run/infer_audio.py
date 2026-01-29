#!/usr/bin/env python3
# run/infer_audio.py
#
# WAV -> MIDI 推論スクリプト（my_mt3/infer.py を使用）
#
# 使い方例:
#   単一ファイル:
#     python run/infer_audio.py --wav data/wavs/pno_0001.wav --ckpt ckpt_piano.pt
#   ディレクトリ一括:
#     python run/infer_audio.py --wav_dir data/wavs --out_dir outputs/midis
#

# ==== add this at the very top ====
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# ==================================

import argparse
from pathlib import Path
import torch

from my_mt3.model import MT3Mini
from my_mt3.tokenizer import VOCAB
from my_mt3.audio import load_wav_mono, wav_to_logmel
from my_mt3.infer import greedy_decode, to_midi_from_tokens


def build_model(ckpt_path: Path, device: str):
    model = MT3Mini(vocab_size=len(VOCAB.itos))
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def transcribe_wav(model: MT3Mini, wav_path: Path, device: str, out_path: Path):
    wav, sr = load_wav_mono(str(wav_path))
    mel = wav_to_logmel(wav, sr=sr)
    token_ids = greedy_decode(model, mel, device=device)
    pm = to_midi_from_tokens(token_ids, sr=sr, step_ms=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
    print(f"生成: {out_path}")


def find_wav_files(root: Path, recursive: bool):
    patterns = ["*.wav", "*.wave"]
    if recursive:
        for pat in patterns:
            yield from root.rglob(pat)
    else:
        for pat in patterns:
            yield from root.glob(pat)


def main():
    parser = argparse.ArgumentParser(description="WAV を MIDI に推論変換")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", help="入力WAVファイル")
    group.add_argument("--wav_dir", help="入力WAVディレクトリ")

    parser.add_argument("--ckpt", default=str(Path(__file__).resolve().parents[1] / "ckpt_piano.pt"),
                        help="学習済みチェックポイントへのパス（既定: プロジェクト直下の ckpt_piano.pt）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推論デバイス（cuda/cpu）")
    parser.add_argument("--out", help="出力MIDI（--wav時）。省略で入力と同ディレクトリに .mid")
    parser.add_argument("--out_dir", help="出力ディレクトリ（--wav_dir時）。未指定なら <wav_dir>_midi を自動作成")
    parser.add_argument("--recursive", action="store_true", help="ディレクトリ処理時に再帰探索")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"チェックポイントが見つかりません: {ckpt_path}")

    device = args.device
    print(f"device={device}, ckpt={ckpt_path}")
    model = build_model(ckpt_path, device=device)

    if args.wav:
        wav_path = Path(args.wav)
        if not wav_path.is_file():
            raise SystemExit(f"ファイルが見つかりません: {wav_path}")
        out_path = Path(args.out) if args.out else wav_path.with_suffix(".mid")
        transcribe_wav(model, wav_path, device, out_path)
        return

    # ディレクトリ処理
    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_dir():
        raise SystemExit(f"ディレクトリが見つかりません: {wav_dir}")

    if args.out_dir:
        out_base = Path(args.out_dir)
    else:
        out_base = wav_dir.with_name(wav_dir.name + "_midi")
        print(f"出力先が未指定のため自動作成します: {out_base}")
    out_base.mkdir(parents=True, exist_ok=True)

    total, converted = 0, 0
    for wav_file in find_wav_files(wav_dir, args.recursive):
        total += 1
        rel = wav_file.relative_to(wav_dir)
        out_path = out_base.joinpath(rel).with_suffix(".mid")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            transcribe_wav(model, wav_file, device, out_path)
            converted += 1
        except Exception as e:
            print(f"失敗: {wav_file} -> {e}")
    print(f"処理完了: {converted}/{total} 件")


if __name__ == "__main__":
    main()

