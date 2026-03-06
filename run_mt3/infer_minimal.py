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
from typing import List, Optional


from my_mt3.model import MT3Mini
from my_mt3.tokenizer import build_vocab, INPUT_FRAMES, Vocab, VOCAB_PIANO 
from my_mt3.dataset import chunk_indices, LogMelCfg, LogMelExtractor   # ← dataset.py にある前提
from my_mt3.audio import load_audio_mono, ensure_wave_cache, DEFAULT_SR
from my_mt3.infer import greedy_decode, to_midi_from_tokens            # ← 汎用（固定長、NOF未使用）


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



def make_start_samples(
    total_samples: int,
    *,
    need_samples: int,
    hop: int,
    mode: str = "test",
    input_frames: int = 256,
    stride_frames: Optional[int] = None,
    include_last: bool = True,
    max_chunks_per_song: Optional[int] = None,  # train相当をやりたいなら使う
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    AMTDataset._make_start_samples と同じ思想で、window開始サンプルssを作る。
    推論では通常 mode="test" 相当（決定論的走査）を使う。
    """
    max_start = max(0, total_samples - need_samples)

    if mode == "train":
        # 推論で train 風にサンプルするケースは稀だが、一応用意
        if max_chunks_per_song is None:
            # 全列挙に近くなるので注意
            return list(range(0, max_start + 1, need_samples))
        K = int(max_chunks_per_song)
        if rng is None:
            rng = np.random.default_rng()
        if max_start == 0:
            starts = [0] * K
        else:
            starts = rng.integers(0, max_start + 1, size=K).tolist()
        starts.sort()
        return starts

    # val/test: 決定論的スライド
    if stride_frames is None:
        stride_frames = input_frames
    stride_samples = int(stride_frames) * hop
    if stride_samples <= 0:
        raise ValueError(f"stride_samples must be > 0, got {stride_samples}")

    starts = list(range(0, max_start + 1, stride_samples))

    if include_last:
        if len(starts) == 0:
            starts = [0]
        else:
            last = starts[-1]
            if last != max_start and (max_start - last) > 0:
                starts.append(max_start)

    return starts


def shift_and_merge_pm(out_pm: pretty_midi.PrettyMIDI, pm_chunk: pretty_midi.PrettyMIDI, *, t0: float) -> None:
    """
    チャンクMIDIを t0 秒シフトして out_pm にマージ。
    - programが同じ instrument が out_pm にあればそこへ追加
    - なければ instrument を新規追加
    """
    # program -> instrument index
    prog2idx = {inst.program: idx for idx, inst in enumerate(out_pm.instruments)}

    for inst in pm_chunk.instruments:
        prog = int(inst.program)
        if prog in prog2idx:
            dst = out_pm.instruments[prog2idx[prog]]
        else:
            dst = pretty_midi.Instrument(program=prog, is_drum=inst.is_drum, name=inst.name)
            out_pm.instruments.append(dst)
            prog2idx[prog] = len(out_pm.instruments) - 1

        for n in inst.notes:
            dst.notes.append(
                pretty_midi.Note(
                    velocity=int(n.velocity),
                    pitch=int(n.pitch),
                    start=float(n.start) + float(t0),
                    end=float(n.end) + float(t0),
                )
            )

    # 任意：ノートを時刻順に整列
    for inst in out_pm.instruments:
        inst.notes.sort(key=lambda n: (n.start, n.pitch))


@torch.no_grad()
def infer_one_song(
    model,
    audio_path: str,
    *,
    device: str = "cuda",
    sr: int = 16000,
    hop: int = 256,
    n_fft: int = 2048,
    n_mels: int = 256,
    input_frames: int = 256,
    step_ms: int = 10,
    program_id: int = 0,
    vocab: Vocab = VOCAB_PIANO,
    max_len: int = 1024,
    # window sweep options
    stride_frames: Optional[int] = None,     # None => input_frames（非オーバーラップ）
    include_last: bool = True,
    # train風サンプルがしたいなら
    mode: str = "test",                      # "test"/"val"/"train"
    max_chunks_per_song: Optional[int] = None,
) -> pretty_midi.PrettyMIDI:
    """
    AMTDataset と同じ窓定義で 1曲推論する。

    - 波形を sr へ resample & mono 化して y:[N] を得る
    - window = need_samples サンプル固定で切り出す（末尾不足は0-pad）
    - center=False の log-mel を計算し、必ず [input_frames, n_mels] に揃える
    - greedy_decode で token列生成
    - tokens->MIDI（窓内0起点）を t0 シフトして曲全体へマージ

    Returns:
      out_pm: pretty_midi.PrettyMIDI（曲全体）
    """
    model.eval()

    # ---- audio ----
    y, _ = load_audio_mono(audio_path, sr=sr)
    total_samples = int(len(y))

    # ---- feature extractor（学習と同一設定）----
    feat = LogMelExtractor(LogMelCfg(sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels))

    # ---- datasetと同じ窓長 ----
    need_samples = (int(input_frames) - 1) * int(hop) + int(n_fft)
    window_sec = need_samples / float(sr)

    # 参考（推論ではtoken側エンコードしないがデバッグ用）
    frame_max_template = int(round(window_sec * 1000.0 / float(step_ms)))
    frame_max_token = max(0, frame_max_template - 1)

    # ---- window starts（dataset同等）----
    starts = make_start_samples(
        total_samples,
        need_samples=need_samples,
        hop=hop,
        mode=mode,
        input_frames=input_frames,
        stride_frames=stride_frames,
        include_last=include_last,
        max_chunks_per_song=max_chunks_per_song,
    )

    out_pm = pretty_midi.PrettyMIDI()

    for ss in starts:
        ee = ss + need_samples
        y_seg = y[ss:ee]

        # 末尾不足は0-padして固定長に（dataset同等）
        if len(y_seg) < need_samples:
            y_seg = np.pad(y_seg, (0, need_samples - len(y_seg)), mode="constant")

        # mel: [input_frames, n_mels] のはず
        mel = feat(y_seg)

        # 保険（理論上一致する）
        if mel.shape[0] != input_frames:
            if mel.shape[0] > input_frames:
                mel = mel[:input_frames]
            else:
                padT = input_frames - mel.shape[0]
                mel = np.pad(mel, ((0, padT), (0, 0)), mode="constant")

        # ---- model input: [1, T, F] ----
        mel_t = torch.from_numpy(mel).to(device=device, dtype=torch.float32).unsqueeze(0)

        # ---- decode ----
        token_ids = greedy_decode(
            model,
            mel_t,
            max_len=max_len,
            device=device,
            program_id=int(program_id),
            vocab=vocab,
        )

        # ---- tokens -> MIDI (window内0起点) ----
        pm_chunk = to_midi_from_tokens(
            token_ids,
            program_id=int(program_id),
            step_ms=int(step_ms),
            vocab=vocab,
        )

        # ---- shift by window start ----
        s_sec = ss / float(sr)
        shift_and_merge_pm(out_pm, pm_chunk, t0=float(s_sec))

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

    # --- model ---
    vocab = build_vocab(input_frames=INPUT_FRAMES, instrument_type="piano", include_note_off=True)
    model = MT3Mini(vocab_size=len(vocab.itos)).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

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
            hop=args.hop,
            n_fft=args.n_fft,
            n_mels=args.n_mels,
            input_frames=INPUT_FRAMES,
            max_len=args.max_len,
            program_id=args.program_id,
            vocab=vocab,
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
            hop=args.hop,
            n_fft=args.n_fft,
            n_mels=args.n_mels,
            input_frames=INPUT_FRAMES,
            max_len=args.max_len,
            program_id=pid,
            vocab=vocab,
        )
        pm_pred.write(str(out_mid))

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
