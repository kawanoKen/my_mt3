from __future__ import annotations

# ==== add this at the very top ====
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# ==================================

import argparse
from pathlib import Path
import os
from typing import List, Tuple, Optional
import numpy as np
import pretty_midi
import concurrent.futures as futures

import torch
from tqdm import tqdm
import pandas as pd

from my_mt3.model import MT3Mini
from my_mt3.tokenizer import VOCAB, INPUT_FRAMES, build_vocab
from my_mt3.audio import ensure_wave_cache, load_audio_mono
from my_mt3.dataset import LogMelCfg, LogMelExtractor
from my_mt3.infer import to_midi_from_tokens_piano, ChunkDecodeResult, greedy_decode_batch, greedy_decode_batch_with_logprobs
from my_mt3.eval import evaluate_midi_pair


def collect_pairs_maps_csv(
    maps_csv: str | Path,
    split: str = "validation",
    *,
    program_id: int = 0,
) -> List[Tuple[str, ...]]:
    """MAPS_*_scenario.csv から指定 split の (audio_path, midi_path, program_id) を収集する。"""
    df = pd.read_csv(maps_csv)
    out: List[Tuple[str, ...]] = []
    for _, row in df[df["split"] == split].iterrows():
        out.append((str(row["audio_path"]), str(row["midi_path"]), int(program_id)))
    return out


def collect_pairs_maestro(
    root: str | Path,
    split: str = "validation",
    *,
    program_id: Optional[int] = 0,   # None なら (audio, midi) の2要素にする
    require_exists: bool = True,
) -> List[Tuple[str, ...]]:
    """
    MAESTRO v3.0.0 の CSV (maestro-v3.0.0.csv) を読み、
    指定 split の (audio_path, midi_path, program_id) のタプルを収集する。
    """
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # 期待カラムの存在確認
    for col in ("split", "audio_filename", "midi_filename"):
        if col not in df.columns:
            raise ValueError(f"CSV column '{col}' not found in {csv_path}")

    out: List[Tuple[str, ...]] = []
    subset = df[df["split"] == split]
    for audio_rel, midi_rel in zip(subset["audio_filename"], subset["midi_filename"]):
        audio_path = root / str(audio_rel)
        midi_path  = root / str(midi_rel)
        if require_exists and (not audio_path.exists() or not midi_path.exists()):
            continue
        if program_id is None:
            out.append((str(audio_path), str(midi_path)))
        else:
            out.append((str(audio_path), str(midi_path), int(program_id)))
    return out


def shift_and_merge_pm(out_pm: pretty_midi.PrettyMIDI, pm_chunk: pretty_midi.PrettyMIDI, *, t0: float) -> None:
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
    # 並べ替えは最後に一括で実施（性能最適化）


@torch.no_grad()
def infer_one_song_piano(
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
    max_len: int = 1024,
    stride_frames: Optional[int] = None,
    batch_size: int = 16,
    return_confidence: bool = False,
):
    """
    Returns:
      return_confidence=False: PrettyMIDI
      return_confidence=True:  (PrettyMIDI, List[dict])
        dict keys: chunk_idx, t0, t1, n_tokens, log_pyx, log_pyx_norm
    """
    model.eval()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    vocab = build_vocab(input_frames=input_frames, instrument_type="piano", include_note_off=True)
    y, _ = load_audio_mono(audio_path, sr=sr)
    total_samples = int(len(y))
    feat = LogMelExtractor(LogMelCfg(sr=sr, n_fft=n_fft, hop=hop, n_mels=n_mels))
    need_samples = (int(input_frames) - 1) * int(hop) + int(n_fft)
    window_sec = need_samples / float(sr)
    if stride_frames is None:
        stride_frames = input_frames
    stride_samples = int(stride_frames) * int(hop)
    starts = list(range(0, max(0, total_samples - need_samples) + 1, max(1, stride_samples)))
    if len(starts) == 0:
        starts = [0]
    last_start = max(0, total_samples - need_samples)
    if starts[-1] != last_start:
        starts.append(last_start)
    out_pm = pretty_midi.PrettyMIDI()
    conf_rows: List[dict] = []
    # carry-over: pitch -> absolute onset (sec).  前チャンクから持ち越し中のノート
    carry: dict[int, float] = {}
    _vel = 80
    _default_dur_sec = 0.05

    def _add_carry_note(pitch: int, onset_sec: float, offset_sec: float) -> None:
        if offset_sec <= onset_sec:
            offset_sec = onset_sec + _default_dur_sec
        if not out_pm.instruments:
            out_pm.instruments.append(
                pretty_midi.Instrument(program=int(program_id))
            )
        out_pm.instruments[0].notes.append(
            pretty_midi.Note(velocity=_vel, pitch=pitch,
                             start=onset_sec, end=offset_sec)
        )

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
                padT = input_frames - mel.shape[0]
                mel = np.pad(mel, ((0, padT), (0, 0)), mode="constant")
        mel_list.append(mel.astype(np.float32, copy=False))

    for b0 in range(0, len(starts), max(1, int(batch_size))):
        b1 = min(len(starts), b0 + int(batch_size))
        mels_bt = torch.from_numpy(np.stack(mel_list[b0:b1], axis=0)).to(device=device, dtype=torch.float32)

        use_autocast = torch.cuda.is_available()

        if return_confidence:
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    token_batch, lp_batch = greedy_decode_batch_with_logprobs(
                        model, mels_bt, max_len=max_len, device=device,
                        program_id=int(program_id), vocab=vocab,
                    )
            else:
                token_batch, lp_batch = greedy_decode_batch_with_logprobs(
                    model, mels_bt, max_len=max_len, device=device,
                    program_id=int(program_id), vocab=vocab,
                )
        else:
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    token_batch = greedy_decode_batch(
                        model, mels_bt, max_len=max_len, device=device,
                        program_id=int(program_id), vocab=vocab,
                    )
            else:
                token_batch = greedy_decode_batch(
                    model, mels_bt, max_len=max_len, device=device,
                    program_id=int(program_id), vocab=vocab,
                )
            lp_batch = None

        for local_i, token_ids in enumerate(token_batch):
            chunk_idx = b0 + local_i
            ss = starts[chunk_idx]
            res: ChunkDecodeResult = to_midi_from_tokens_piano(
                token_ids,
                program_id=int(program_id),
                step_ms=int(step_ms),
                vocab=vocab,
            )
            s_sec = ss / float(sr)

            # --- carry-over 解決 ---
            for p in list(carry.keys()):
                if p in res.tie_pitches:
                    if p in res.tie_offsets_ms:
                        # tie 宣言あり & offset 確定 → ノート完成
                        _add_carry_note(p, carry.pop(p),
                                        s_sec + res.tie_offsets_ms[p] / 1000.0)
                    # else: tie 宣言あり & offset 未確定 → まだ鳴り続ける (carry 維持)
                else:
                    # tie 宣言なし → チャンク境界で強制終了
                    _add_carry_note(p, carry.pop(p), s_sec)

            # --- チャンク内の通常ノートをマージ ---
            shift_and_merge_pm(out_pm, res.pm, t0=float(s_sec))

            # --- open notes → carry に追加 ---
            for p, on_ms in res.open_onsets_ms.items():
                if p not in carry:
                    carry[p] = s_sec + on_ms / 1000.0

            if return_confidence and lp_batch is not None:
                lps = lp_batch[local_i]
                n_tok = len(lps)
                log_pyx = sum(lps) if n_tok > 0 else 0.0
                conf_rows.append({
                    "chunk_idx": chunk_idx,
                    "t0": s_sec,
                    "t1": s_sec + window_sec,
                    "n_tokens": n_tok,
                    "log_pyx": log_pyx,
                    "log_pyx_norm": (log_pyx / n_tok) if n_tok > 0 else 0.0,
                })

    # --- 最終チャンク後に残った carry を閉じる ---
    if carry:
        last_sec = starts[-1] / float(sr) + window_sec
        for p, onset_sec in carry.items():
            _add_carry_note(p, onset_sec, last_sec)
        carry.clear()

    for inst in out_pm.instruments:
        inst.notes.sort(key=lambda n: (n.start, n.pitch))

    if return_confidence:
        return out_pm, conf_rows
    return out_pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="model state_dict (.pt)")
    ap.add_argument(
        "--root",
        type=str,
        default=None,
        help="MAESTRO v3.0.0 root (contains maestro-v3.0.0.csv and year dirs)",
    )
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--maps_csv", type=str, default=None,
                     help="MAPS_*_scenario.csv (overrides --root)")
    ap.add_argument("--maps_split", type=str, default="validation",
                     choices=["train", "validation"],
                     help="split to use from --maps_csv (default: validation)")
    # 単体ファイル推論用
    ap.add_argument("--wav", type=str, help="入力WAVファイル（単体推論）")
    ap.add_argument("--out", type=str, help="出力MIDIパス（--wav 指定時）")
    ap.add_argument("--out_dir", type=str, default="outputs/maestro_validation_pred")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--program_id", type=int, default=0)
    ap.add_argument("--use_cache", action="store_true", help="use/create wave cache (.npy)")
    ap.add_argument("--cache_dir", type=str, default="cache/wave_sr16000")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--stride_frames", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=1, help="manual sharding: total number of shards")
    ap.add_argument("--shard_id", type=int, default=0, help="manual sharding: this shard id (0..num_shards-1)")
    ap.add_argument("--prefetch_cache_workers", type=int, default=0, help="parallelize wave cache creation")
    ap.add_argument("--cpu_threads", type=int, default=4, help="limit PyTorch CPU threads (0=leave default)")
    ap.add_argument("--cpu_interop_threads", type=int, default=1, help="limit PyTorch interop threads (0=auto)")
    # evaluation
    ap.add_argument("--eval", action="store_true", help="evaluate against ground-truth MIDI after inference")
    ap.add_argument("--onset_tolerance", type=float, default=0.05, help="onset tolerance in seconds")
    ap.add_argument("--offset_ratio", type=float, default=0.2, help="offset ratio (None to use fixed tolerance)")
    ap.add_argument("--offset_min_tolerance", type=float, default=0.05, help="minimum offset tolerance in seconds")
    ap.add_argument("--max_songs", type=int, default=0, help="limit number of songs to process (0=all)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip inference if output .pred.mid already exists (resume-friendly)")
    ap.add_argument("--save_confidence", action="store_true",
                     help="save per-chunk confidence (log_pyx) to chunk_confidence.csv")
    args = ap.parse_args()

    # ---- limit CPU thread usage to avoid CPU hog ----
    if args.cpu_threads and args.cpu_threads > 0:
        try:
            torch.set_num_threads(int(args.cpu_threads))
        except Exception:
            pass
        interop = int(args.cpu_interop_threads) if args.cpu_interop_threads > 0 else max(1, args.cpu_threads // 2)
        try:
            torch.set_num_interop_threads(int(interop))
        except Exception:
            pass
        # Common BLAS/OpenMP envs (best-effort)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.cpu_threads))

    # --- model ---
    model = MT3Mini(vocab_size=len(VOCAB.itos)).to(args.device)
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

        pm_pred = infer_one_song_piano(
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
            batch_size=args.batch_size,
            stride_frames=args.stride_frames,
        )
        pm_pred.write(str(out_path))
        print(f"done -> {out_path}")
        return

    # --- directory/dataset mode ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect pairs
    if args.maps_csv is not None:
        pairs = collect_pairs_maps_csv(args.maps_csv, split=args.maps_split, program_id=args.program_id)
        print(f"[MAPS] {args.maps_csv}  split={args.maps_split}  pairs={len(pairs)}")
    else:
        if args.root is None:
            raise SystemExit("Either --root or --maps_csv must be specified.")
        pairs = collect_pairs_maestro(args.root, split=args.split, program_id=args.program_id)
    if len(pairs) == 0:
        raise RuntimeError("No pairs found. Check root/csv paths.")
    if args.num_shards <= 0:
        raise SystemExit("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard_id must be in [0,{args.num_shards-1}]")
    pairs = [pairs[i] for i in range(args.shard_id, len(pairs), args.num_shards)]
    if args.max_songs > 0:
        pairs = pairs[:args.max_songs]
    split_label = args.maps_split if args.maps_csv else args.split
    print(f"pairs[{split_label}] shard {args.shard_id}/{args.num_shards}: {len(pairs)}")

    # optional: precreate wave cache in parallel (CPU-bound IO)
    if args.use_cache and args.prefetch_cache_workers > 0:
        def _cache_one(tup):
            audio_path, _, _ = tup
            if str(audio_path).endswith(".npy"):
                return audio_path
            return ensure_wave_cache(audio_path, cache_dir=args.cache_dir, sr=args.sr)
        with futures.ThreadPoolExecutor(max_workers=int(args.prefetch_cache_workers)) as ex:
            list(tqdm(ex.map(_cache_one, pairs), total=len(pairs), desc="prefetch cache", unit="file"))

    eval_rows = []
    conf_all_rows = []
    n_skipped = 0

    for audio_path, midi_path, pid in tqdm(pairs, desc=f"infer {split_label}", unit="song"):
        a_path = audio_path
        if args.use_cache and not str(audio_path).endswith(".npy"):
            a_path = ensure_wave_cache(audio_path, cache_dir=args.cache_dir, sr=args.sr)

        stem = Path(audio_path).stem
        out_mid = out_dir / f"{stem}.pred.mid"

        if args.skip_existing and out_mid.exists():
            n_skipped += 1
            if args.eval:
                try:
                    m = evaluate_midi_pair(
                        midi_path, str(out_mid),
                        onset_tolerance=args.onset_tolerance,
                        offset_ratio=args.offset_ratio,
                        offset_min_tolerance=args.offset_min_tolerance,
                        program=pid,
                    )
                    m["stem"] = stem
                    eval_rows.append(m)
                except Exception as e:
                    print(f"[eval skip] {stem}: {e}")
            continue

        result = infer_one_song_piano(
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
            batch_size=args.batch_size,
            stride_frames=args.stride_frames,
            return_confidence=args.save_confidence,
        )
        if args.save_confidence:
            pm_pred, chunk_confs = result
            for row in chunk_confs:
                row["stem"] = stem
            conf_all_rows.extend(chunk_confs)
        else:
            pm_pred = result

        pm_pred.write(str(out_mid))

        if args.eval:
            try:
                m = evaluate_midi_pair(
                    midi_path, str(out_mid),
                    onset_tolerance=args.onset_tolerance,
                    offset_ratio=args.offset_ratio,
                    offset_min_tolerance=args.offset_min_tolerance,
                    program=pid,
                )
                m["stem"] = stem
                eval_rows.append(m)
            except Exception as e:
                print(f"[eval skip] {stem}: {e}")

    if args.skip_existing:
        print(f"skipped existing files: {n_skipped}")
    print(f"done -> {out_dir}")

    if args.save_confidence and conf_all_rows:
        df_conf = pd.DataFrame(conf_all_rows)
        conf_csv = out_dir / "chunk_confidence.csv"
        df_conf.to_csv(conf_csv, index=False)
        print(f"confidence CSV -> {conf_csv}  ({len(df_conf)} chunks)")

    if args.eval and eval_rows:
        df_eval = pd.DataFrame(eval_rows)
        csv_path = out_dir / "eval_metrics.csv"
        df_eval.to_csv(csv_path, index=False)
        print(f"eval CSV -> {csv_path}")

        metric_cols = [c for c in df_eval.columns if c != "stem"]
        summary = df_eval[metric_cols].mean()
        print("\n=== Evaluation Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}")
        print()


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 uv run run/infer_maestro.py --ckpt checkpoints_maestro/run_20260216_155146/model_ep1800.pt --root maestro-v3.0.0 --num_shards 2 --shard_id 1
# CUDA_VISIBLE_DEVICES=1 uv run run/infer_maestro.py --ckpt checkpoints_maestro/run_20260216_155146/model_ep1800.pt --root maestro-v3.0.0 --num_shards 2 --shard_id 2