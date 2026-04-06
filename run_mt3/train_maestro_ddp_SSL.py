# run/train_maestro_ddp_SSL.py
#
# Semi-supervised training on MAESTRO:
#   - A small fraction of MAESTRO train is used as labeled data
#   - The rest is treated as unlabeled (pseudo-label with confidence filtering)
#   - Validation split is kept intact for evaluation
#
# Usage:
#   python -m torch.distributed.run --nproc_per_node=4 run/train_maestro_ddp_SSL.py \
#     --root dataset/maestro-v3.0.0 \
#     --label_frac 0.05 \
#     --epochs 200 --bs 8

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os
import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import warnings
from tqdm import tqdm

from my_mt3.tokenizer import build_vocab
from my_mt3.audio import ensure_wave_cache
import concurrent.futures as futures
import csv
import json

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


def collect_pairs_maestro_ssl(
    root: str | Path,
    *,
    label_frac: float = 0.05,
    seed: int = 42,
    program_id: int = 0,
    require_exists: bool = True,
) -> Tuple[Dict[str, List[Tuple[str, ...]]], List[str], List[str]]:
    """
    MAESTRO train split を labeled / unlabeled に分割。
    validation/test はそのまま返す。

    Returns:
      pairs_labeled: {"train": [...], "validation": [...]}
      unlabeled_wavs: List[str]  (audio paths only, no MIDI)
      unlabeled_midis: List[str] (corresponding MIDI paths, for oracle filter)
    """
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("split", "audio_filename", "midi_filename"):
        if col not in df.columns:
            raise ValueError(f"CSV column '{col}' not found in {csv_path}")

    all_train: List[Tuple[str, str, int]] = []
    val_pairs: List[Tuple[str, str, int]] = []

    for _, row in df.iterrows():
        audio_path = root / str(row["audio_filename"])
        midi_path = root / str(row["midi_filename"])
        if require_exists and (not audio_path.exists() or not midi_path.exists()):
            continue
        entry = (str(audio_path), str(midi_path), int(program_id))
        if row["split"] == "train":
            all_train.append(entry)
        elif row["split"] == "validation":
            val_pairs.append(entry)

    rng = random.Random(seed)
    rng.shuffle(all_train)

    n_labeled = max(1, int(len(all_train) * label_frac))
    labeled = all_train[:n_labeled]
    unlabeled = all_train[n_labeled:]

    unlabeled_wavs = [wav for wav, _midi, _pid in unlabeled]
    unlabeled_midis = [midi for _wav, midi, _pid in unlabeled]

    pairs_labeled = {
        "train": labeled,
        "validation": val_pairs,
    }

    return pairs_labeled, unlabeled_wavs, unlabeled_midis


def load_split_from_csv(
    split_csv: str | Path,
    root: str | Path,
    *,
    program_id: int = 0,
) -> Tuple[Dict[str, List[Tuple[str, ...]]], List[str], List[str]]:
    """
    保存済み ssl_split.csv から labeled/unlabeled 分割を復元。
    validation は MAESTRO CSV から別途取得。
    """
    root = Path(root)
    split_df = pd.read_csv(split_csv)

    labeled: List[Tuple[str, str, int]] = []
    unlabeled_wavs: List[str] = []
    unlabeled_midis: List[str] = []

    for _, row in split_df.iterrows():
        if row["type"] == "labeled":
            labeled.append((str(row["audio_path"]), str(row["midi_path"]), int(program_id)))
        elif row["type"] == "unlabeled":
            unlabeled_wavs.append(str(row["audio_path"]))
            unlabeled_midis.append(str(row.get("midi_path", "")))

    maestro_csv = root / "maestro-v3.0.0.csv"
    val_pairs: List[Tuple[str, str, int]] = []
    if maestro_csv.exists():
        mdf = pd.read_csv(maestro_csv)
        for _, row in mdf[mdf["split"] == "validation"].iterrows():
            audio_path = root / str(row["audio_filename"])
            midi_path = root / str(row["midi_filename"])
            if audio_path.exists() and midi_path.exists():
                val_pairs.append((str(audio_path), str(midi_path), int(program_id)))

    pairs_labeled = {"train": labeled, "validation": val_pairs}
    return pairs_labeled, unlabeled_wavs, unlabeled_midis


def load_maps_csv(
    maps_csv: str | Path,
    *,
    program_id: int = 0,
) -> Tuple[Dict[str, List[Tuple[str, ...]]], List[str], List[str]]:
    """MAPS_*_scenario.csv (split/audio_path/midi_path/...) を読み込む。

    Returns:
      pairs_labeled: {"train": [...], "validation": [...]}
      unlabeled_wavs: [] (MAPS は全て有ラベルなので空)
      unlabeled_midis: []
    """
    df = pd.read_csv(maps_csv)
    train_pairs: List[Tuple[str, str, int]] = []
    val_pairs: List[Tuple[str, str, int]] = []
    for _, row in df.iterrows():
        entry = (str(row["audio_path"]), str(row["midi_path"]), int(program_id))
        if row["split"] == "train":
            train_pairs.append(entry)
        elif row["split"] == "validation":
            val_pairs.append(entry)
    return {"train": train_pairs, "validation": val_pairs}, [], []


def load_maestro_unlabeled(
    root: str | Path,
    *,
    require_exists: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    MAESTRO train split を unlabeled 用に読み込む。
    Returns:
      unlabeled_wavs: List[str]
      unlabeled_midis: List[str]  # oracle_filter 用
    """
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("split", "audio_filename", "midi_filename"):
        if col not in df.columns:
            raise ValueError(f"CSV column '{col}' not found in {csv_path}")

    unlabeled_wavs: List[str] = []
    unlabeled_midis: List[str] = []
    for _, row in df[df["split"] == "train"].iterrows():
        audio_path = root / str(row["audio_filename"])
        midi_path = root / str(row["midi_filename"])
        if require_exists and (not audio_path.exists() or not midi_path.exists()):
            continue
        unlabeled_wavs.append(str(audio_path))
        unlabeled_midis.append(str(midi_path))

    return unlabeled_wavs, unlabeled_midis


def _extract_epoch_num(path: Path, prefix: str) -> int:
    m = re.search(rf"{re.escape(prefix)}(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def resolve_resume_checkpoint(resume_path: str) -> tuple[str, bool]:
    """
    Resolve resume checkpoint path.
    Returns:
      (ckpt_path, is_full_training_state)
    """
    p = Path(resume_path)
    if p.is_file():
        is_full = p.name.startswith("train_state_ep")
        return str(p), is_full

    if p.is_dir():
        full_states = sorted(
            p.glob("train_state_ep*.pt"),
            key=lambda x: _extract_epoch_num(x, "train_state_ep"),
        )
        if full_states:
            return str(full_states[-1]), True

        model_only = sorted(
            p.glob("model_ep*.pt"),
            key=lambda x: _extract_epoch_num(x, "model_ep"),
        )
        if model_only:
            return str(model_only[-1]), False

    raise FileNotFoundError(
        f"no resume checkpoint found under: {resume_path} "
        "(expected train_state_ep*.pt or model_ep*.pt)"
    )


def _collect_cli_option_keys(argv: list[str]) -> set[str]:
    keys: set[str] = set()
    for tok in argv:
        if not tok.startswith("--"):
            continue
        key = tok[2:].split("=", 1)[0].strip()
        if key:
            keys.add(key.replace("-", "_"))
    return keys


def apply_meta_defaults(args, *, cli_keys: set[str], resume_dir: Path) -> None:
    meta_path = resume_dir / "meta.json"
    if not meta_path.exists():
        print(f"[resume] meta.json not found: {meta_path}")
        return
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[resume] failed to load meta.json: {e}")
        return

    meta_args = meta.get("args", {})
    if not isinstance(meta_args, dict):
        print(f"[resume] invalid meta args format: {meta_path}")
        return

    skipped = {"resume_ckpt", "save_dir"}
    restored = []
    for k, v in meta_args.items():
        if k in skipped:
            continue
        if k in cli_keys:
            continue
        if hasattr(args, k):
            setattr(args, k, v)
            restored.append(k)

    if "save_dir" not in cli_keys and getattr(args, "save_dir", None) is None:
        args.save_dir = str(resume_dir)
        restored.append("save_dir")

    if restored:
        print(f"[resume] restored args from meta.json ({len(restored)}): {', '.join(sorted(restored))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Semi-supervised MAESTRO training (partial labels)")
    ap.add_argument("--root", type=str, default="dataset/maestro-v3.0.0",
                     help="MAESTRO v3.0.0 root directory")
    ap.add_argument("--label_frac", type=float, default=0.1,
                     help="fraction of MAESTRO train to use as labeled (0.0-1.0)")
    ap.add_argument("--label_seed", type=int, default=42,
                     help="random seed for labeled/unlabeled split")
    ap.add_argument("--split_csv", type=str, default=None,
                     help="path to a saved ssl_split.csv to reproduce an exact split "
                          "(overrides --label_frac and --label_seed)")
    ap.add_argument("--maps_csv", type=str, default=None,
                     help="path to a MAPS_*_scenario.csv (overrides --root/--split_csv; "
                          "supervised-only, no pseudo-label)")
    ap.add_argument("--maps_labeled_maestro_unlabeled", action="store_true",
                     help="use MAPS (maps_csv) as labeled data and MAESTRO train (root) as unlabeled data")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr_t", type=float, default=2e-4, help="Transformer lr")
    ap.add_argument("--input_frames", type=int, default=256,
                     help="segment length in frames (hop=256, sr=16k: 121 ~ 2.048s)")
    ap.add_argument("--lr_warmup_epochs", type=int, default=100,
                     help="number of warmup epochs for LR scheduler")
    ap.add_argument("--lr_min_ratio", type=float, default=0.1,
                     help="minimum LR ratio for cosine decay scheduler")

    # Pseudo-label (SSL)
    ap.add_argument("--pseudo_start_epoch", type=int, default=1000,
                     help="start pseudo-label training from this epoch")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--unsup_weight", type=float, default=0.6,
                     help="weight for unsupervised pseudo-label loss")
    ap.add_argument("--pseudo_max_len", type=int, default=1024)
    ap.add_argument("--pseudo_threshold", type=float, default=-1.1,
                     help="chunk-level mean log-prob threshold for pseudo-label (used when pseudo_topn=0)")
    ap.add_argument("--pseudo_topn", type=int, default=0,
                     help="select top-N most confident chunks per batch as pseudo-labels "
                          "(0=use threshold mode instead)")
    ap.add_argument("--pseudo_note_target_only", action="store_true",
                     help="compute unsupervised loss only on pseudo note tokens filtered by note confidence")
    ap.add_argument("--pseudo_note_threshold", type=float, default=-0.6,
                     help="minimum note-level confidence to keep pseudo note tokens")
    ap.add_argument(
        "--pseudo_note_prob_threshold",
        type=float,
        default=None,
        help="note confidence threshold for output-probability score (default: fallback to --pseudo_note_threshold)",
    )
    ap.add_argument(
        "--pseudo_note_mask_threshold",
        type=float,
        default=None,
        help="note confidence threshold for mask-delta score (default: fallback to --pseudo_note_threshold)",
    )
    ap.add_argument(
        "--pseudo_note_conf_mode",
        type=str,
        default="single",
        choices=["single", "prob", "mask", "prob_and_mask", "prob_or_mask"],
        help="how to combine output-probability and mask-delta confidences",
    )
    ap.add_argument(
        "--pseudo_note_score_metric",
        type=str,
        default="logprob_mean",
        choices=["logprob_mean", "abs_mask_delta", "log_abs_mask_delta"],
        help="legacy single-score metric used when --pseudo_note_conf_mode=single",
    )
    ap.add_argument(
        "--pseudo_note_mask_score_metric",
        type=str,
        default="abs_mask_delta",
        choices=["abs_mask_delta", "log_abs_mask_delta"],
        help="mask-delta metric used in mask/prob_and_mask/prob_or_mask modes",
    )
    ap.add_argument(
        "--pseudo_note_mask_width_ratio",
        type=float,
        default=0.2,
        help="mask band width ratio for mask-delta note score",
    )
    ap.add_argument(
        "--pseudo_note_mask_fill",
        type=str,
        default="zero",
        choices=["zero", "mean"],
        help="mask fill strategy for mask-delta note score",
    )
    ap.add_argument("--pseudo_note_onset_only", action="store_true",
                     help="when pseudo_note_target_only is enabled, keep only note-on tokens")
    ap.add_argument("--pseudo_note_without_chunk", action="store_true",
                     help="when pseudo_note_target_only is enabled, ignore chunk filter and use token-only mask")
    ap.add_argument("--pseudo_repair_order", action="store_true",
                     help="repair pseudo token order before chunk filter: same-time pitch low->high, on->off, dedup same token")
    ap.add_argument("--pseudo_double_chunk_middle_only", action="store_true",
                     help="for pseudo labels, decode two consecutive real chunks and keep only middle window "
                          "(2nd half of chunk-A + 1st half of chunk-B)")
    ap.add_argument("--pseudo_ignore_second_zero_onset", action="store_true",
                     help="when pseudo_double_chunk_middle_only is enabled, ignore NOTE_ON at local 0s in chunk-B")
    ap.add_argument("--pseudo_debug_n", type=int, default=0,
                     help="save N pseudo-label debug samples (txt + piano roll) for kept chunks")
    ap.add_argument("--pseudo_debug_dir", type=str, default=None,
                     help="output directory for pseudo debug artifacts (default: <save_dir>/pseudo_debug)")
    ap.add_argument("--pseudo_debug_start_epoch", type=int, default=0,
                     help="(deprecated) ignored; pseudo debug always starts at pseudo_start_epoch")

    # Pretrained
    ap.add_argument("--pretrained_ckpt", type=str, default=None,
                     help="path to a pretrained MT3 checkpoint (.pt) to initialise model weights")
    ap.add_argument("--resume_ckpt", type=str, default=None,
                     help="checkpoint path or run directory to resume training "
                          "(prefers train_state_ep*.pt; falls back to model_ep*.pt)")

    # Oracle filter (実験用: 正解 MIDI で疑似ラベルをフィルタ)
    ap.add_argument("--oracle_filter", action="store_true",
                     help="use ground-truth MIDI to filter pseudo-labels (oracle experiment)")
    ap.add_argument("--oracle_metric", type=str, default="note_f",
                     help="evaluation metric for oracle filter (default: note_f)")
    ap.add_argument("--oracle_threshold", type=float, default=0.5,
                     help="minimum metric value to keep a pseudo-label chunk")
    ap.add_argument("--oracle_note_target_only", action="store_true",
                     help="after oracle chunk filtering, compute unsupervised loss only on "
                          "pseudo note tokens matched to GT notes (same pitch, close onset/offset)")
    ap.add_argument("--oracle_note_without_chunk", action="store_true",
                     help="when oracle_note_target_only is enabled, ignore oracle chunk filter and use token-only mask")
    ap.add_argument("--timewise_onset_tf_weight", type=float, default=0.0,
                     help="aux loss weight: teacher forcing on [TIM_t -> NOTE_ON@t] sequences")
    ap.add_argument("--timewise_onset_tf_max_groups", type=int, default=0,
                     help="max time-token groups per sample for auxiliary onset TF (0=all)")
    ap.add_argument("--timewise_onset_tf_min_onsets", type=int, default=1,
                     help="minimum NOTE_ON count in a time group to include in auxiliary onset TF")
    ap.add_argument("--pseudo_unsup_cross_attn_only", action="store_true",
                     help="restrict pseudo-label unsupervised gradient update to decoder cross-attention only")

    # Saving
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=100,
                     help="run validation (incl. mir_eval metrics) every N epochs")
    ap.add_argument("--save-dir", type=str, default=None)

    # Cache
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache-root", type=str, default="cache/wave_sr16000")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--prefetch_cache_workers", type=int, default=0)
    ap.add_argument("--no_augment", action="store_true",
                     help="disable spectrogram augmentation for SSL pseudo-label student step")
    args = ap.parse_args()
    cli_keys = _collect_cli_option_keys(sys.argv[1:])

    resume_ckpt = None
    resume_dir = None
    if args.resume_ckpt is not None:
        resume_ckpt, is_full_state = resolve_resume_checkpoint(args.resume_ckpt)
        resume_dir = Path(resume_ckpt).parent
        print(f"[resume] checkpoint: {resume_ckpt}")
        if not is_full_state:
            print("[resume] WARNING: model-only checkpoint selected; optimizer/scheduler states are not available")
        apply_meta_defaults(args, cli_keys=cli_keys, resume_dir=resume_dir)

    if int(getattr(args, "pseudo_debug_start_epoch", 0)) > 0:
        print("[pseudo-debug] --pseudo_debug_start_epoch is ignored; start is fixed at --pseudo_start_epoch")

    # Output directory
    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.maps_csv is not None:
            scenario = Path(args.maps_csv).stem
            save_dir = os.path.join("checkpoints_MAPS", f"run_{ts}_{scenario}")
        else:
            frac_str = f"{args.label_frac:.0%}".replace("%", "pct")
            save_dir = os.path.join("checkpoints_maestro_SSL", f"run_{ts}_frac{frac_str}")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    if args.maps_csv is not None:
        print(f"[MAPS] Loading scenario CSV: {args.maps_csv}")
        pairs_labeled, unlabeled_wavs, unlabeled_midis = load_maps_csv(args.maps_csv, program_id=0)
        if args.maps_labeled_maestro_unlabeled:
            print(f"[MAPS+MAESTRO] Loading MAESTRO unlabeled from: {args.root}")
            unlabeled_wavs, unlabeled_midis = load_maestro_unlabeled(args.root)
    elif args.split_csv is not None:
        print(f"[SSL] Loading split from CSV: {args.split_csv}")
        pairs_labeled, unlabeled_wavs, unlabeled_midis = load_split_from_csv(
            args.split_csv, args.root, program_id=0,
        )
    else:
        pairs_labeled, unlabeled_wavs, unlabeled_midis = collect_pairs_maestro_ssl(
            args.root,
            label_frac=args.label_frac,
            seed=args.label_seed,
            program_id=0,
        )

    n_train_labeled = len(pairs_labeled["train"])
    n_unlabeled = len(unlabeled_wavs)
    n_val = len(pairs_labeled["validation"])
    total_train = n_train_labeled + n_unlabeled

    if args.maps_csv is not None:
        if args.maps_labeled_maestro_unlabeled:
            print(f"[MAPS+MAESTRO] labeled_train={n_train_labeled} | unlabeled_maestro={n_unlabeled} | val={n_val}")
        else:
            print(f"[MAPS] train={n_train_labeled} | val={n_val}")
    else:
        total_str = max(total_train, 1)
        print(f"[SSL] label_frac={args.label_frac:.1%} | "
              f"labeled={n_train_labeled} ({n_train_labeled/total_str:.1%}) | "
              f"unlabeled={n_unlabeled} ({n_unlabeled/total_str:.1%}) | "
              f"val={n_val}")

    # Vocab
    vocab = build_vocab(input_frames=args.input_frames, instrument_type="piano", include_note_off=True)

    # ===== Save meta.json =====
    meta = {
        "script": "train_maestro_ddp_SSL.py",
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "data": {
            "labeled_train": n_train_labeled,
            "unlabeled_train": n_unlabeled,
            "total_train": total_train,
            "label_frac_actual": round(n_train_labeled / total_train, 4) if total_train else 0,
            "validation": n_val,
        },
    }
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Meta saved -> {meta_path}")

    # Save split info (always, so the exact split is reproducible from this run)
    split_out = os.path.join(save_dir, "ssl_split.csv")
    with open(split_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "audio_path", "midi_path"])
        for wav, midi, _pid in pairs_labeled["train"]:
            w.writerow(["labeled", wav, midi])
        for wav, midi in zip(unlabeled_wavs, unlabeled_midis):
            w.writerow(["unlabeled", wav, midi])
    print(f"Split info saved -> {split_out}")

    # Cache directory: MAPS uses its own subdir to avoid collision with MAESTRO cache
    if args.maps_csv is not None:
        cache_subdir = "MAPS_MAESTRO" if args.maps_labeled_maestro_unlabeled else "MAPS"
    else:
        cache_subdir = "maestro-v3.0.0"
    pairs_real = {"train": unlabeled_wavs}
    cache_dir_maestro = str(Path(args.cache_root) / cache_subdir)

    if not args.no_cache:
        all_wavs = (
            [w for (w, _m, _pid) in pairs_labeled["train"] if not str(w).endswith(".npy")]
            + [w for (w, _m, _pid) in pairs_labeled["validation"] if not str(w).endswith(".npy")]
            + [w for w in unlabeled_wavs if not str(w).endswith(".npy")]
        )

        def _prefetch_set(wavs, cache_dir, title):
            if not wavs:
                return []
            if args.prefetch_cache_workers > 0:
                print(f"[cache:{title}] prefetch: {len(wavs)} files -> {cache_dir}")
                def _cache_one(w):
                    try:
                        return ensure_wave_cache(w, cache_dir=cache_dir, sr=args.sr)
                    except Exception as e:
                        return f"ERR:{w}:{e}"
                with futures.ThreadPoolExecutor(max_workers=int(args.prefetch_cache_workers)) as ex:
                    cached = list(tqdm(ex.map(_cache_one, wavs), total=len(wavs),
                                       desc=f"prefetch {title}", unit="wav"))
            else:
                cached = [ensure_wave_cache(w, cache_dir=cache_dir, sr=args.sr) for w in wavs]
            return [c for c in cached if isinstance(c, str) and not c.startswith("ERR:")]

        cached_all = _prefetch_set(all_wavs, cache_dir_maestro, "maestro")
        w2c = dict(zip(all_wavs, cached_all))

        pairs_labeled["train"] = [(w2c.get(w, w), m, pid)
                                   for (w, m, pid) in pairs_labeled["train"]]
        pairs_labeled["validation"] = [(w2c.get(w, w), m, pid)
                                        for (w, m, pid) in pairs_labeled["validation"]]
        pairs_real = {"train": [w2c.get(w, w) for w in unlabeled_wavs]}

    print(f"Checkpoints -> {save_dir}")

    # Use the confusion-based training loop (no discriminator, pseudo-label only)
    from my_mt3.train_DA_confusion import train_loop_distributed_DA_confusion

    model = train_loop_distributed_DA_confusion(
        pairs_labeled,
        vocab=vocab,
        use_dc=False,
        pairs_real=pairs_real,
        lambda_adv=0.0,
        lr_t=args.lr_t,
        lr_c=0.0,
        chunk_frames=None,
        disc_hidden=256,
        use_pseudo=True,
        pseudo_start_epoch=args.pseudo_start_epoch,
        ema_decay=args.ema_decay,
        unsup_weight=args.unsup_weight,
        pseudo_max_len=args.pseudo_max_len,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_topn=args.pseudo_topn,
        pretrained_ckpt=args.pretrained_ckpt,
        resume_ckpt=resume_ckpt,
        oracle_filter=args.oracle_filter,
        oracle_metric=args.oracle_metric,
        oracle_threshold=args.oracle_threshold,
        oracle_midi_paths=unlabeled_midis if (args.oracle_filter or args.pseudo_debug_n > 0) else None,
        oracle_note_target_only=args.oracle_note_target_only,
        oracle_note_without_chunk=args.oracle_note_without_chunk,
        pseudo_note_target_only=args.pseudo_note_target_only,
        pseudo_note_onset_only=args.pseudo_note_onset_only,
        pseudo_note_threshold=args.pseudo_note_threshold,
        pseudo_note_prob_threshold=args.pseudo_note_prob_threshold,
        pseudo_note_mask_threshold=args.pseudo_note_mask_threshold,
        pseudo_note_conf_mode=args.pseudo_note_conf_mode,
        pseudo_note_score_metric=args.pseudo_note_score_metric,
        pseudo_note_mask_score_metric=args.pseudo_note_mask_score_metric,
        pseudo_note_mask_width_ratio=args.pseudo_note_mask_width_ratio,
        pseudo_note_mask_fill=args.pseudo_note_mask_fill,
        pseudo_note_without_chunk=args.pseudo_note_without_chunk,
        pseudo_repair_order=args.pseudo_repair_order,
        pseudo_double_chunk_middle_only=args.pseudo_double_chunk_middle_only,
        pseudo_ignore_second_zero_onset=args.pseudo_ignore_second_zero_onset,
        pseudo_debug_n=args.pseudo_debug_n,
        pseudo_debug_dir=args.pseudo_debug_dir,
        pseudo_debug_start_epoch=args.pseudo_debug_start_epoch,
        timewise_onset_tf_weight=args.timewise_onset_tf_weight,
        timewise_onset_tf_max_groups=args.timewise_onset_tf_max_groups,
        timewise_onset_tf_min_onsets=args.timewise_onset_tf_min_onsets,
        pseudo_unsup_cross_attn_only=args.pseudo_unsup_cross_attn_only,
        use_augment=not args.no_augment,
        epochs=args.epochs,
        bs=args.bs,
        input_frames=args.input_frames,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_min_ratio=args.lr_min_ratio,
        val_every=args.val_every,
        save_every=args.save_every,
        save_dir=save_dir,
        use_cache=False,
        cache_dir=args.cache_root,
        sr=args.sr,
        num_workers=2,
    )

    # Plot losses
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs, total, sup, unsup, val_l, val_acc = [], [], [], [], [], []
        csv_path = Path(save_dir) / "da_losses.csv"
        if csv_path.exists():
            with open(csv_path, "r") as f:
                r = csv.DictReader(f)
                for row in r:
                    xs.append(int(row["epoch"]))
                    total.append(float(row["train_total"]))
                    sup.append(float(row["train_sup"]))
                    unsup.append(float(row["train_unsup"]))
                    val_l.append(float(row["val_loss"]))
                    val_acc.append(float(row.get("val_token_acc", "0") or 0.0))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(xs, total, label="total")
            ax.plot(xs, sup, label="supervised")
            ax.plot(xs, unsup, label="pseudo-label (unsup)")
            if any(val_l):
                ax.plot(xs, val_l, label="val_loss", linestyle="--")
            if any(val_acc):
                ax.plot(xs, val_acc, label="val_token_acc", linestyle="-.")
            ax.axvline(x=args.pseudo_start_epoch, color="gray", linestyle=":",
                       label=f"pseudo start (ep {args.pseudo_start_epoch})")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_title(f"SSL Training  (label_frac={args.label_frac:.1%})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = Path(save_dir) / "ssl_losses.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"Loss plot -> {fig_path}")
    except Exception as e:
        print(f"(warn) failed to plot: {e}")

    print(f"Training finished -> {save_dir}")

# python -m torch.distributed.run --nproc_per_node=4 run/train_maestro_ddp_SSL.py \
#   --root dataset/maestro-v3.0.0 --label_frac 0.05 --epochs 200 --bs 8
