from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from my_mt3.analysis_attribution import analyze_chunk
from my_mt3.dataset import AMTDataset
from my_mt3.model import MT3Mini
from my_mt3.tokenizer import build_vocab


def _build_pair_tensors(token_ids: list[int], device: torch.device):
    if len(token_ids) < 2:
        raise RuntimeError("token_ids too short for teacher forcing.")
    y_in = torch.tensor(token_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y_tg = torch.tensor(token_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    return y_in, y_tg


def _pick_chunk_index(n: int, chunk_idx: int, chunk_select: str) -> int:
    if n <= 0:
        raise RuntimeError("No chunks extracted from input wav/midi.")
    if chunk_idx >= 0:
        if chunk_idx >= n:
            raise SystemExit(f"--chunk_idx must be < {n}, got {chunk_idx}")
        return chunk_idx
    if chunk_select == "last":
        return n - 1
    if chunk_select == "middle":
        return n // 2
    return 0


def _parse_float_list(v: str) -> list[float]:
    out: list[float] = []
    for x in v.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _parse_mode_list(v: str) -> list[str]:
    out = [x.strip() for x in v.split(",") if x.strip()]
    for m in out:
        if m not in ("pair_and_offset", "offset_only"):
            raise SystemExit(f"Unsupported prefix mode: {m}")
    return out


def _plot_results(df: pd.DataFrame, out_path: Path) -> None:
    valid = df[df["skipped"] == 0].copy()
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=False)
    kinds = [("non", "onset"), ("nof", "offset")]

    for col, (kind_key, kind_name) in enumerate(kinds):
        kind_df = valid[valid["target_kind"] == kind_key].copy()

        prefix = kind_df[kind_df["experiment_type"] == "prefix_drop"].copy()
        if not prefix.empty:
            for mode in sorted(prefix["prefix_mode"].dropna().unique()):
                sub = prefix[prefix["prefix_mode"] == mode]
                g = sub.groupby("drop_ratio", as_index=False).agg(
                    delta_nll=("delta_nll", "mean"),
                    delta_nll_std=("delta_nll", "std"),
                )
                g = g.sort_values("drop_ratio")
                x = g["drop_ratio"].to_numpy()
                y = g["delta_nll"].to_numpy()
                ystd = g["delta_nll_std"].fillna(0.0).to_numpy()
                axes[0, col].plot(x, y, marker="o", label=f"{mode}: Δnll")
                axes[0, col].fill_between(x, y - ystd, y + ystd, alpha=0.2)
        axes[0, col].set_title(f"Prefix token drop ({kind_name})")
        axes[0, col].set_xlabel("drop ratio")
        axes[0, col].set_ylabel("degradation")
        axes[0, col].grid(alpha=0.3)
        if not prefix.empty:
            axes[0, col].legend(loc="best")

        mask = kind_df[kind_df["experiment_type"] == "source_mask"].copy()
        if not mask.empty:
            for fill in sorted(mask["mask_fill"].dropna().unique()):
                sub = mask[mask["mask_fill"] == fill]
                g = sub.groupby("mask_width_ratio", as_index=False).agg(
                    delta_nll=("delta_nll", "mean"),
                    delta_nll_std=("delta_nll", "std"),
                )
                g = g.sort_values("mask_width_ratio")
                x = g["mask_width_ratio"].to_numpy()
                y = g["delta_nll"].to_numpy()
                ystd = g["delta_nll_std"].fillna(0.0).to_numpy()
                axes[1, col].plot(x, y, marker="o", label=f"fill={fill}: Δnll")
                axes[1, col].fill_between(x, y - ystd, y + ystd, alpha=0.2)
        axes[1, col].set_title(f"Input mask around target time ({kind_name})")
        axes[1, col].set_xlabel("mask width ratio")
        axes[1, col].set_ylabel("degradation")
        axes[1, col].grid(alpha=0.3)
        if not mask.empty:
            axes[1, col].legend(loc="best")

        noise = kind_df[kind_df["experiment_type"] == "source_noise"].copy()
        if not noise.empty:
            for width in sorted(noise["noise_width_ratio"].dropna().unique()):
                sub = noise[noise["noise_width_ratio"] == width]
                g = sub.groupby("noise_sigma", as_index=False).agg(
                    delta_nll=("delta_nll", "mean"),
                    delta_nll_std=("delta_nll", "std"),
                )
                g = g.sort_values("noise_sigma")
                x = g["noise_sigma"].to_numpy()
                y = g["delta_nll"].to_numpy()
                ystd = g["delta_nll_std"].fillna(0.0).to_numpy()
                axes[2, col].plot(x, y, marker="o", label=f"w={width}: Δnll")
                axes[2, col].fill_between(x, y - ystd, y + ystd, alpha=0.2)
        axes[2, col].set_title(f"Input noise around target time ({kind_name})")
        axes[2, col].set_xlabel("noise sigma")
        axes[2, col].set_ylabel("degradation")
        axes[2, col].grid(alpha=0.3)
        if not noise.empty:
            axes[2, col].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summary_from_df(df: pd.DataFrame) -> dict:
    valid = df[df["skipped"] == 0].copy()
    out: dict = {
        "num_rows_total": int(len(df)),
        "num_rows_valid": int(len(valid)),
        "num_rows_skipped": int(len(df) - len(valid)),
    }
    if "chunk_idx" in df.columns:
        out["num_chunks"] = int(df["chunk_idx"].nunique())

    if valid.empty:
        out["message"] = "No valid experiment rows."
        out["prefix_by_condition"] = {}
        out["source_mask_by_condition"] = {}
        out["source_noise_by_condition"] = {}
        return out

    def _stats(sub: pd.DataFrame) -> dict:
        vals = sub["delta_nll"].to_numpy(dtype=float)
        return {
            "count": int(len(vals)),
            "mean_delta_nll": float(np.mean(vals)),
            "std_delta_nll": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "var_delta_nll": float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "p50_delta_nll": float(np.percentile(vals, 50)),
            "p90_delta_nll": float(np.percentile(vals, 90)),
        }

    prefix = valid[valid["experiment_type"] == "prefix_drop"]
    mask = valid[valid["experiment_type"] == "source_mask"]
    noise = valid[valid["experiment_type"] == "source_noise"]
    out["prefix_overall"] = _stats(prefix) if not prefix.empty else {}
    out["source_mask_overall"] = _stats(mask) if not mask.empty else {}
    out["source_noise_overall"] = _stats(noise) if not noise.empty else {}

    by_kind: dict = {}
    for kind in sorted(valid["target_kind"].dropna().unique()):
        ksub = valid[valid["target_kind"] == kind]
        by_kind[str(kind)] = _stats(ksub) if not ksub.empty else {}
    out["overall_by_target_kind"] = by_kind

    p_dict: dict = {}
    if not prefix.empty:
        for (mode, ratio, kind), g in prefix.groupby(["prefix_mode", "drop_ratio", "target_kind"]):
            p_dict[f"{mode}|{ratio}|{kind}"] = _stats(g)
    out["prefix_by_condition"] = p_dict

    m_dict: dict = {}
    if not mask.empty:
        for (fill, width, kind), g in mask.groupby(["mask_fill", "mask_width_ratio", "target_kind"]):
            m_dict[f"fill={fill}|width={width}|{kind}"] = _stats(g)
    out["source_mask_by_condition"] = m_dict

    n_dict: dict = {}
    if not noise.empty:
        for (sigma, width, kind), g in noise.groupby(["noise_sigma", "noise_width_ratio", "target_kind"]):
            n_dict[f"sigma={sigma}|width={width}|{kind}"] = _stats(g)
    out["source_noise_by_condition"] = n_dict
    return out


def main():
    ap = argparse.ArgumentParser(description="Counterfactual attribution by prefix-drop and source-noise")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--midi", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--program_id", type=int, default=0)

    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--n_mels", type=int, default=256)
    ap.add_argument("--input_frames", type=int, default=256)
    ap.add_argument("--step_ms", type=int, default=10)

    ap.add_argument("--chunk_idx", type=int, default=-1, help=">=0: explicit index, -1: use --chunk_select")
    ap.add_argument("--chunk_select", type=str, default="first", choices=["first", "middle", "last"])
    ap.add_argument("--all_chunks", action="store_true", help="analyze all chunks in the song")

    ap.add_argument("--prefix_drop_ratios", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--prefix_drop_modes", type=str, default="pair_and_offset,offset_only")
    ap.add_argument("--mask_width_ratios", type=str, default="0.1,0.2,0.3")
    ap.add_argument("--mask_fill", type=str, default="zero", choices=["zero", "mean"])
    ap.add_argument("--noise_sigmas", type=str, default="1.0,5.0, 10, 30")
    ap.add_argument("--noise_width_ratios", type=str, default="0.1,0.2,0.3,0.5")
    ap.add_argument("--noise_repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix_drop_ratios = _parse_float_list(args.prefix_drop_ratios)
    prefix_drop_modes = _parse_mode_list(args.prefix_drop_modes)
    mask_width_ratios = _parse_float_list(args.mask_width_ratios)
    noise_sigmas = _parse_float_list(args.noise_sigmas)
    noise_width_ratios = _parse_float_list(args.noise_width_ratios)
    if not prefix_drop_ratios:
        raise SystemExit("--prefix_drop_ratios is empty.")
    if not prefix_drop_modes:
        raise SystemExit("--prefix_drop_modes is empty.")
    if not mask_width_ratios:
        raise SystemExit("--mask_width_ratios is empty.")
    if not noise_sigmas:
        raise SystemExit("--noise_sigmas is empty.")
    if not noise_width_ratios:
        raise SystemExit("--noise_width_ratios is empty.")

    device = torch.device(args.device)
    vocab = build_vocab(
        input_frames=args.input_frames,
        sr=args.sr,
        hop=args.hop,
        n_fft=args.n_fft,
        time_step_ms=args.step_ms,
        instrument_type="piano",
        include_note_off=True,
    )

    ds = AMTDataset(
        [(args.wav, args.midi, int(args.program_id))],
        mode="validation",
        sr=args.sr,
        hop=args.hop,
        step_ms=args.step_ms,
        input_frames=args.input_frames,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        vocab=vocab,
    )
    chunks = ds[0]

    model = MT3Mini(vocab_size=len(vocab.itos), n_mels=args.n_mels).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    chunk_indices = list(range(len(chunks))) if args.all_chunks else [_pick_chunk_index(len(chunks), args.chunk_idx, args.chunk_select)]
    all_rows = []
    all_targets = []
    chunk_meta = []
    for cidx in chunk_indices:
        mel_np, token_ids, (s_sec, e_sec) = chunks[cidx]
        mel = torch.tensor(mel_np, dtype=torch.float32, device=device).unsqueeze(0)
        y_in, y_tg = _build_pair_tensors([int(x) for x in token_ids], device)
        res = analyze_chunk(
            model,
            mel=mel,
            y_in=y_in,
            y_tg=y_tg,
            vocab=vocab,
            pad_id=int(vocab.pad),
            step_ms=int(args.step_ms),
            sr=int(args.sr),
            hop=int(args.hop),
            prefix_drop_ratios=prefix_drop_ratios,
            prefix_drop_modes=prefix_drop_modes,
            mask_width_ratios=mask_width_ratios,
            mask_fill=args.mask_fill,
            noise_sigmas=noise_sigmas,
            noise_width_ratios=noise_width_ratios,
            noise_repeats=int(args.noise_repeats),
            seed=int(args.seed) + int(cidx),
        )
        for row in res.rows:
            row["chunk_idx"] = int(cidx)
            row["chunk_start_sec"] = float(s_sec)
            row["chunk_end_sec"] = float(e_sec)
            all_rows.append(row)
        for t in res.targets:
            all_targets.append(
                {
                    "chunk_idx": int(cidx),
                    "chunk_start_sec": float(s_sec),
                    "chunk_end_sec": float(e_sec),
                    "target_token_idx": t.idx,
                    "target_token_id": t.token_id,
                    "target_token": t.token_str,
                    "target_kind": t.kind,
                    "target_pitch": t.pitch,
                    "target_time_idx": t.time_idx,
                    "target_time_ms": t.time_ms,
                    "target_frame": t.time_frame,
                }
            )
        chunk_meta.append({"chunk_idx": int(cidx), "chunk_start_sec": float(s_sec), "chunk_end_sec": float(e_sec)})

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "token_attribution.csv"
    df.to_csv(csv_path, index=False)

    targets_df = pd.DataFrame(all_targets)
    targets_path = out_dir / "target_tokens.csv"
    targets_df.to_csv(targets_path, index=False)

    chunks_path = out_dir / "chunk_windows.csv"
    pd.DataFrame(chunk_meta).to_csv(chunks_path, index=False)

    summary = _summary_from_df(df)
    summary.update(
        {
            "wav": args.wav,
            "midi": args.midi,
            "all_chunks": bool(args.all_chunks),
            "chunk_indices": chunk_indices,
            "prefix_drop_ratios": prefix_drop_ratios,
            "prefix_drop_modes": prefix_drop_modes,
            "mask_width_ratios": mask_width_ratios,
            "mask_fill": args.mask_fill,
            "noise_sigmas": noise_sigmas,
            "noise_width_ratios": noise_width_ratios,
            "noise_repeats": int(args.noise_repeats),
        }
    )
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fig_path = out_dir / "attribution_plot.png"
    _plot_results(df, fig_path)

    print(f"token CSV   -> {csv_path}")
    print(f"targets CSV -> {targets_path}")
    print(f"chunks CSV  -> {chunks_path}")
    print(f"summary JSON-> {summary_path}")
    print(f"plot PNG    -> {fig_path}")


if __name__ == "__main__":
    main()
