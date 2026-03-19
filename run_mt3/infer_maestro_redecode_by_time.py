from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pretty_midi
import torch
from tqdm import tqdm

from my_mt3.audio import ensure_wave_cache, load_audio_mono
from my_mt3.dataset import LogMelCfg, LogMelExtractor
from my_mt3.decode_kv import FastDecoderKV
from my_mt3.infer import to_midi_from_tokens_piano
from my_mt3.model import MT3Mini
from my_mt3.tokenizer import INPUT_FRAMES, build_vocab


def collect_pairs_maestro(
    root: str | Path,
    split: str = "validation",
    *,
    program_id: int = 0,
) -> List[Tuple[str, str, int]]:
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    out: List[Tuple[str, str, int]] = []
    for _, row in df[df["split"] == split].iterrows():
        wav = root / str(row["audio_filename"])
        midi = root / str(row["midi_filename"])
        if wav.exists() and midi.exists():
            out.append((str(wav), str(midi), int(program_id)))
    return out


def shift_and_merge_pm(out_pm: pretty_midi.PrettyMIDI, pm_chunk: pretty_midi.PrettyMIDI, *, t0: float) -> None:
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


def _split_generated_until_next_time(
    generated: List[int],
    *,
    time_token_set: set[int],
    eos_id: int,
    pad_id: int,
) -> tuple[List[int], int | None]:
    out: List[int] = []
    for tok in generated:
        t = int(tok)
        if t == eos_id or t == pad_id:
            return out, None
        if t in time_token_set:
            # 次の time token が出たらこの区間は終了し、その token を次セグメント起点に使う
            return out, int(t)
        out.append(t)
    return out, None


@torch.no_grad()
def _decode_full_tokens_with_prefix(
    model: MT3Mini,
    mem_1: torch.Tensor,  # [1,T,D]
    *,
    bos_id: int,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
) -> List[int]:
    fast = FastDecoderKV(model.dec, max_len=max_new_tokens + 4)
    y0 = torch.tensor([[int(bos_id)]], dtype=torch.long, device=mem_1.device)
    y, _pmax, _margin, _logp = fast.greedy_generate_with_probs(
        mem_1,
        y0=y0,
        max_new_tokens=int(max_new_tokens),
        eos_id=int(eos_id),
        pad_id=int(pad_id),
        return_with_prefix=False,
    )
    toks = y[0].tolist()
    out = []
    for t in toks:
        t = int(t)
        if t == pad_id:
            break
        out.append(t)
        if t == eos_id:
            break
    return out


@torch.no_grad()
def _decode_segment_from_time_token(
    model: MT3Mini,
    mem_1: torch.Tensor,  # [1,T,D]
    *,
    bos_id: int,
    time_token_id: int,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    time_token_set: set[int],
) -> tuple[List[int], int | None]:
    # Prefix: [PRG, TIM_k]
    y0 = torch.tensor([[int(bos_id), int(time_token_id)]], dtype=torch.long, device=mem_1.device)
    fast = FastDecoderKV(model.dec, max_len=max_new_tokens + y0.size(1) + 4)
    y, _pmax, _margin, _logp = fast.greedy_generate_with_probs(
        mem_1,
        y0=y0,
        max_new_tokens=int(max_new_tokens),
        eos_id=int(eos_id),
        pad_id=int(pad_id),
        return_with_prefix=False,  # generated part only
    )
    gen = y[0].tolist()
    body, next_time_token = _split_generated_until_next_time(
        gen,
        time_token_set=time_token_set,
        eos_id=int(eos_id),
        pad_id=int(pad_id),
    )
    # to_midi_from_tokens_piano は PRG/TIM を含む列を期待
    return [int(bos_id), int(time_token_id)] + body + [int(eos_id)], next_time_token


@torch.no_grad()
def infer_one_song_redecode_by_time(
    model: MT3Mini,
    audio_path: str,
    *,
    device: str,
    sr: int,
    hop: int,
    n_fft: int,
    n_mels: int,
    input_frames: int,
    step_ms: int,
    program_id: int,
    max_len: int,
    stride_frames: Optional[int],
    batch_size: int,
) -> tuple[pretty_midi.PrettyMIDI, List[dict]]:
    model.eval()
    vocab = build_vocab(input_frames=input_frames, instrument_type="piano", include_note_off=True)
    eos_id = int(vocab.eos)
    pad_id = int(vocab.pad)
    bos_id = int(vocab.instrument_type[f"PRG_{int(program_id)}"])
    time_token_set = set(int(x) for x in vocab.time.values())

    y, _ = load_audio_mono(audio_path, sr=sr)
    total_samples = int(len(y))
    feat = LogMelExtractor(LogMelCfg(sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels))
    need_samples = (int(input_frames) - 1) * int(hop) + int(n_fft)
    if stride_frames is None:
        stride_frames = input_frames
    stride_samples = int(stride_frames) * int(hop)
    starts = list(range(0, max(0, total_samples - need_samples) + 1, max(1, stride_samples)))
    if len(starts) == 0:
        starts = [0]
    last_start = max(0, total_samples - need_samples)
    if starts[-1] != last_start:
        starts.append(last_start)

    mel_list: List[np.ndarray] = []
    for ss in starts:
        ee = ss + need_samples
        y_seg = y[ss:ee]
        if len(y_seg) < need_samples:
            y_seg = np.pad(y_seg, (0, need_samples - len(y_seg)), mode="constant")
        mel = feat(y_seg)
        if mel.shape[0] != input_frames:
            if mel.shape[0] > input_frames:
                mel = mel[:input_frames]
            else:
                mel = np.pad(mel, ((0, input_frames - mel.shape[0]), (0, 0)), mode="constant")
        mel_list.append(mel.astype(np.float32, copy=False))

    out_pm = pretty_midi.PrettyMIDI()
    rows: List[dict] = []

    for b0 in range(0, len(starts), max(1, int(batch_size))):
        b1 = min(len(starts), b0 + int(batch_size))
        mels_bt = torch.from_numpy(np.stack(mel_list[b0:b1], axis=0)).to(device=device, dtype=torch.float32)
        mem_bt = model.enc(mels_bt)

        for local_i in range(b1 - b0):
            chunk_idx = b0 + local_i
            ss = starts[chunk_idx]
            s_sec = ss / float(sr)
            mem_1 = mem_bt[local_i:local_i + 1]

            # 厳密版:
            #   TIM_0 からdecodeし、次のTIMが出たら停止。
            #   次セグメントはその出現TIMから再decode（逐次連鎖）。
            current_time_id = int(vocab.time.get(0, min(vocab.time.values())))
            max_segments = 256
            for seg_i in range(max_segments):
                seg_tokens, next_time_id = _decode_segment_from_time_token(
                    model, mem_1,
                    bos_id=bos_id,
                    time_token_id=int(current_time_id),
                    max_new_tokens=max_len,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    time_token_set=time_token_set,
                )
                res = to_midi_from_tokens_piano(
                    seg_tokens,
                    program_id=int(program_id),
                    step_ms=int(step_ms),
                    vocab=vocab,
                )
                shift_and_merge_pm(out_pm, res.pm, t0=float(s_sec))
                rows.append(
                    {
                        "chunk_idx": int(chunk_idx),
                        "segment_idx": int(seg_i),
                        "chunk_start_sec": float(s_sec),
                        "time_token_id": int(current_time_id),
                        "time_token_str": str(vocab.itos[int(current_time_id)]),
                        "next_time_token_id": (int(next_time_id) if next_time_id is not None else None),
                        "next_time_token_str": (str(vocab.itos[int(next_time_id)]) if next_time_id is not None else None),
                        "segment_token_len": int(len(seg_tokens)),
                    }
                )
                if next_time_id is None:
                    break
                # 同じtimeが連続して無限ループ化するのを防ぐ
                if int(next_time_id) == int(current_time_id):
                    break
                current_time_id = int(next_time_id)

    for inst in out_pm.instruments:
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
    return out_pm, rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "MAESTRO inference variant: for each TIM_* found in a chunk's full decode, "
            "restart decoding from [PRG, TIM_*] and decode until next TIM_* appears."
        )
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--root", type=str, default=None, help="MAESTRO root (directory mode)")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--wav", type=str, default=None, help="single-file mode input wav")
    ap.add_argument("--out", type=str, default=None, help="single-file mode output midi")
    ap.add_argument("--out_dir", type=str, default="outputs/maestro_validation_pred_timewise")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--program_id", type=int, default=0)
    ap.add_argument("--use_cache", action="store_true")
    ap.add_argument("--cache_dir", type=str, default="cache/wave_sr16000")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--stride_frames", type=int, default=None)
    ap.add_argument("--max_songs", type=int, default=0)
    args = ap.parse_args()

    model = MT3Mini(vocab_size=len(build_vocab(input_frames=INPUT_FRAMES, instrument_type="piano", include_note_off=True).itos)).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    if args.wav:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            raise SystemExit(f"file not found: {wav_path}")
        out_mid = Path(args.out) if args.out else wav_path.with_suffix(".timewise.pred.mid")
        out_mid.parent.mkdir(parents=True, exist_ok=True)
        a_path = str(wav_path)
        if args.use_cache and not a_path.endswith(".npy"):
            a_path = ensure_wave_cache(a_path, cache_dir=args.cache_dir, sr=args.sr)
        pm, rows = infer_one_song_redecode_by_time(
            model,
            a_path,
            device=args.device,
            sr=args.sr,
            hop=args.hop,
            n_fft=args.n_fft,
            n_mels=args.n_mels,
            input_frames=INPUT_FRAMES,
            step_ms=10,
            program_id=args.program_id,
            max_len=args.max_len,
            stride_frames=args.stride_frames,
            batch_size=args.batch_size,
        )
        pm.write(str(out_mid))
        seg_csv = out_mid.with_suffix(".segments.csv")
        pd.DataFrame(rows).to_csv(seg_csv, index=False)
        print(f"done -> {out_mid}")
        print(f"segments -> {seg_csv}")
        return

    if args.root is None:
        raise SystemExit("Either --wav or --root is required.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = collect_pairs_maestro(args.root, split=args.split, program_id=args.program_id)
    if args.max_songs > 0:
        pairs = pairs[:args.max_songs]

    for audio_path, _midi_path, pid in tqdm(pairs, desc=f"infer {args.split}", unit="song"):
        stem = Path(audio_path).stem
        out_mid = out_dir / f"{stem}.pred.mid"
        a_path = audio_path
        if args.use_cache and not str(audio_path).endswith(".npy"):
            a_path = ensure_wave_cache(audio_path, cache_dir=args.cache_dir, sr=args.sr)
        pm, rows = infer_one_song_redecode_by_time(
            model,
            a_path,
            device=args.device,
            sr=args.sr,
            hop=args.hop,
            n_fft=args.n_fft,
            n_mels=args.n_mels,
            input_frames=INPUT_FRAMES,
            step_ms=10,
            program_id=pid,
            max_len=args.max_len,
            stride_frames=args.stride_frames,
            batch_size=args.batch_size,
        )
        pm.write(str(out_mid))
        pd.DataFrame(rows).to_csv(out_dir / f"{stem}.segments.csv", index=False)

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
