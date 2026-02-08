# run/infer_groove_test.py
#
# GrooveMIDI の test split を 1曲まるごと推論して、MIDI を保存するスクリプト。
# - AMTDataset の「chunk列挙」と同じ条件（center=False の log-mel）で推論
# - 各チャンクを独立に decode → MIDI化 → チャンク開始時刻で time shift して結合
#
# 使い方例:
#   python run/infer_groove_test.py --ckpt ckpt_piano.pt --out_dir outputs/groove_test_pred
#

from __future__ import annotations

# ==== add this at the very top ====
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# ==================================

import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import pretty_midi
from tqdm import tqdm

from my_mt3.model import MT3Mini
from my_mt3.tokenizer import VOCAB, INPUT_FRAMES
from my_mt3.dataset import chunk_indices, LogMelCfg, LogMelExtractor   # ← dataset.py にある前提
from my_mt3.audio import load_audio_mono, ensure_wave_cache, DEFAULT_SR
from my_mt3.infer import greedy_decode, to_midi_from_tokens            # ← 既存関数の想定


def collect_pairs_groove(
    root: str | Path = "dataset/groove",
    split: str = "test",
    program_id: int = 0,
) -> list[tuple[str, str, int]]:
    """
    returns: [(audio_path, midi_path, pid), ...]
    """
    root = Path(root)
    df = pd.read_csv(root / "info.csv")

    subset = df[df["split"] == split]
    pairs = []
    for a, m in zip(subset["audio_filename"], subset["midi_filename"]):
        audio = root / str(a)
        midi  = root / str(m)
        if audio.exists() and midi.exists():
            pairs.append((str(audio), str(midi), program_id))
    return pairs


def shift_and_merge_pm(dst: pretty_midi.PrettyMIDI, src: pretty_midi.PrettyMIDI, t0: float):
    """
    src のノートを t0 秒だけシフトして dst に追加
    """
    # 楽器の program / is_drum が一致するものがあればそこへ、なければ新規作成
    for inst in src.instruments:
        # 既存探す
        target = None
        for di in dst.instruments:
            if di.program == inst.program and di.is_drum == inst.is_drum and di.name == inst.name:
                target = di
                break
        if target is None:
            target = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            dst.instruments.append(target)

        for n in inst.notes:
            target.notes.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=float(n.start) + t0,
                    end=float(n.end) + t0,
                )
            )


@torch.no_grad()
def infer_one_song(
    model: MT3Mini,
    audio_path: str,
    *,
    device: str,
    sr: int,
    chunk_sec: float,
    mel_cfg: LogMelCfg,
    max_len: int,
    pid: int,
) -> pretty_midi.PrettyMIDI:
    """
    1曲を chunk に分けて推論し、PrettyMIDI を返す
    """
    y, _ = load_audio_mono(audio_path, sr=sr)
    total_sec = float(len(y)) / float(sr)

    feat = LogMelExtractor(mel_cfg)

    out_pm = pretty_midi.PrettyMIDI()
    for s, e in chunk_indices(total_sec, chunk_sec=chunk_sec, include_last=True):
        ss = int(round(s * sr))
        ee = int(round(e * sr))
        if ee <= ss:
            continue
        y_seg = y[ss:ee].astype(np.float32, copy=False)

        # center=False のため n_fft 未満は pad（フレーム0を避ける）
        if len(y_seg) < mel_cfg.n_fft:
            y_seg = np.pad(y_seg, (0, mel_cfg.n_fft - len(y_seg)), mode="constant")

        mel = feat(y_seg)  # [T, n_mels]
        mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)  # [1, T, F]

        # --- decode ---
        # greedy_decode のシグネチャはプロジェクト依存なので、合わなければここだけ調整してください
        token_ids = greedy_decode(
            model,
            mel_t,
            max_len=max_len,
            device=device,
            program_id=int(pid)
        )

        # --- tokens -> MIDI (chunk内) ---
        # to_midi_from_tokens もプロジェクト依存。pid が必要なら渡してください。
        pm_chunk = to_midi_from_tokens(token_ids, program_id=pid)

        # --- shift by chunk start ---
        shift_and_merge_pm(out_pm, pm_chunk, t0=float(s))

    return out_pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="model state_dict (.pt)")
    ap.add_argument("--root", type=str, default="dataset/groove", help="GrooveMIDI root (contains info.csv)")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    # 単体ファイル推論用
    ap.add_argument("--wav", type=str, help="入力WAVファイル（単体推論）")
    ap.add_argument("--out", type=str, help="出力MIDIパス（--wav 指定時）")
    ap.add_argument("--out_dir", type=str, default="outputs/groove_validation_pred")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--program_id", type=int, default=0)
    ap.add_argument("--use_cache", action="store_true", help="use/create wave cache (.npy)")
    ap.add_argument("--cache_dir", type=str, default="cache/wave_sr16000")
    args = ap.parse_args()

    chunk_sec = INPUT_FRAMES * args.hop / args.sr

    # --- model ---
    model = MT3Mini(vocab_size=len(VOCAB.itos)).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    mel_cfg = LogMelCfg(sr=args.sr, n_fft=args.n_fft, hop=args.hop, n_mels=args.n_mels)

    # --- single file mode ---
    if args.wav:
        wav_path = Path(args.wav)
        if not wav_path.is_file():
            raise SystemExit(f"file not found: {wav_path}")
        out_path = Path(args.out) if args.out else wav_path.with_suffix(".pred.mid")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        a_path = wav_path
        if args.use_cache and not str(wav_path).endswith(".npy"):
            a_path = Path(ensure_wave_cache(str(wav_path), cache_dir=args.cache_dir, sr=args.sr))

        pm_pred = infer_one_song(
            model,
            str(a_path),
            device=args.device,
            sr=args.sr,
            chunk_sec=chunk_sec,
            mel_cfg=mel_cfg,
            max_len=args.max_len,
            pid=args.program_id,
        )
        pm_pred.write(str(out_path))
        print(f"done -> {out_path}")
        return

    # --- directory/dataset mode ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect test pairs
    pairs = collect_pairs_groove(args.root, split=args.split, program_id=args.program_id)
    if len(pairs) == 0:
        raise RuntimeError("No test pairs found. Check root/info.csv paths.")
    print(f"pairs[{args.split}]: {len(pairs)}")

    for audio_path, midi_path, pid in tqdm(pairs, desc=f"infer {args.split}", unit="song"):
        a_path = audio_path
        if args.use_cache and not str(audio_path).endswith(".npy"):
            a_path = ensure_wave_cache(audio_path, cache_dir=args.cache_dir, sr=args.sr)

        # 出力名は audio の stem
        stem = Path(audio_path).stem
        out_mid = out_dir / f"{stem}.pred.mid"

        pm_pred = infer_one_song(
            model,
            a_path,
            device=args.device,
            sr=args.sr,
            chunk_sec=chunk_sec,
            mel_cfg=mel_cfg,
            max_len=args.max_len,
            pid=pid,
        )
        pm_pred.write(str(out_mid))

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
