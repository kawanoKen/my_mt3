from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from my_mt3.dataset import AMTDataset
from my_mt3.model import MT3Mini
from my_mt3.tokenizer import Vocab, build_vocab
from my_mt3.train import make_collate


def collect_maestro_pairs(
    root: str | Path,
    *,
    split: str = "validation",
    max_songs: int = 8,
    program_id: int = 0,
) -> List[Tuple[str, str, int]]:
    root = Path(root)
    csv_path = root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    rows = df[df["split"] == split]
    pairs: List[Tuple[str, str, int]] = []
    for _, row in rows.iterrows():
        wav = root / str(row["audio_filename"])
        mid = root / str(row["midi_filename"])
        if not wav.exists() or not mid.exists():
            continue
        pairs.append((str(wav), str(mid), int(program_id)))
        if max_songs > 0 and len(pairs) >= max_songs:
            break
    return pairs


def load_model(ckpt_path: str | Path, *, vocab_size: int, device: torch.device) -> MT3Mini:
    model = MT3Mini(vocab_size=vocab_size).to(device)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def delete_indices_keep_length(seq: torch.Tensor, remove_idx: List[int], pad_id: int) -> torch.Tensor:
    # seq: [S]
    s = int(seq.numel())
    if not remove_idx:
        return seq.clone()
    remove_set = set(int(i) for i in remove_idx if 0 <= int(i) < s)
    kept = [int(seq[i].item()) for i in range(s) if i not in remove_set]
    kept = kept[:s]
    if len(kept) < s:
        kept.extend([int(pad_id)] * (s - len(kept)))
    return torch.tensor(kept, dtype=seq.dtype, device=seq.device)


def insert_tokens_keep_length(
    seq: torch.Tensor,
    *,
    insert_before_positions: List[int],
    insert_token_ids: List[int],
    pad_id: int,
) -> torch.Tensor:
    # seq: [S], insert token(s) before designated positions; keep output length S
    s = int(seq.numel())
    if not insert_before_positions or not insert_token_ids:
        return seq.clone()

    arr = [int(x.item()) for x in seq]
    pairs = sorted(
        [(int(p), int(t)) for p, t in zip(insert_before_positions, insert_token_ids)],
        key=lambda x: x[0],
    )
    offset = 0
    for p, tid in pairs:
        p = max(0, min(s, p))
        arr.insert(p + offset, int(tid))
        offset += 1

    arr = arr[:s]
    if len(arr) < s:
        arr.extend([int(pad_id)] * (s - len(arr)))
    return torch.tensor(arr, dtype=seq.dtype, device=seq.device)


def _plot_results(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    ops = [("delete", axes[0, 0], axes[1, 0]), ("add", axes[0, 1], axes[1, 1])]

    for op, ax_delta, ax_abs in ops:
        sub = df[df["operation"] == op].copy()
        if sub.empty:
            ax_delta.set_title(f"{op}: no data")
            ax_abs.set_title(f"{op}: no data")
            ax_delta.grid(alpha=0.3)
            ax_abs.grid(alpha=0.3)
            continue

        g = sub.groupby("ratio", as_index=False).agg(
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            abs_mean=("abs_delta", "mean"),
            abs_std=("abs_delta", "std"),
            base_mean=("base_p_target", "mean"),
            ablated_mean=("ablated_p_target", "mean"),
        ).sort_values("ratio")

        x = g["ratio"].to_numpy(dtype=float)
        d = g["delta_mean"].to_numpy(dtype=float)
        ds = g["delta_std"].fillna(0.0).to_numpy(dtype=float)
        a = g["abs_mean"].to_numpy(dtype=float)
        astd = g["abs_std"].fillna(0.0).to_numpy(dtype=float)
        bmean = g["base_mean"].to_numpy(dtype=float)
        amean = g["ablated_mean"].to_numpy(dtype=float)

        ax_delta.plot(x, d, marker="o", label="ΔP(target onset token) mean")
        ax_delta.fill_between(x, d - ds, d + ds, alpha=0.2, label="±1 std")
        ax_delta.plot(x, bmean, marker="x", linestyle="--", label="base P mean")
        ax_delta.plot(x, amean, marker="s", linestyle="--", label="ablated P mean")
        ax_delta.set_title(f"{op}: signed change")
        ax_delta.set_xlabel("perturb ratio")
        ax_delta.set_ylabel("probability")
        ax_delta.grid(alpha=0.3)
        ax_delta.legend(loc="best")

        ax_abs.plot(x, a, marker="o", color="tab:red", label="|ΔP(target onset token)| mean")
        ax_abs.fill_between(x, a - astd, a + astd, alpha=0.2, color="tab:red", label="±1 std")
        ax_abs.set_title(f"{op}: absolute change")
        ax_abs.set_xlabel("perturb ratio")
        ax_abs.set_ylabel("probability")
        ax_abs.grid(alpha=0.3)
        ax_abs.legend(loc="best")

    fig.suptitle("Onset Context Ablation (single-song oriented view)", fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


@torch.no_grad()
def run_ablation(
    model: MT3Mini,
    dl: DataLoader,
    *,
    vocab: Vocab,
    device: torch.device,
    ratios: List[float],
    trials_per_ratio: int,
    positions_per_sample: int,
    include_time_tokens_in_deletion: bool,
    operation: str,
    add_source: str,
    context_source: str,
    max_batches: int,
    seed: int,
) -> dict:
    if context_source != "gt":
        raise ValueError(f"Unsupported context_source: {context_source} (currently only 'gt')")

    note_on_ids_list = list(vocab.note_on.values())
    note_on_ids = torch.tensor(note_on_ids_list, dtype=torch.long, device=device)
    note_on_set = set(int(x) for x in note_on_ids_list)
    time_ids = set(int(tid) for tid in vocab.time.values())
    pad_id = int(vocab.pad)

    rng = np.random.default_rng(int(seed))
    records = []
    batch_count = 0

    for mels, y_in, y_tg in dl:
        mels = mels.to(device, non_blocking=True)
        y_in = y_in.to(device, non_blocking=True)
        y_tg = y_tg.to(device, non_blocking=True)

        # context_source='gt': y_in/y_tg are built from GT MIDI token sequence by AMTDataset/make_collate.
        logits = model(mels, y_in)
        probs = torch.softmax(logits, dim=-1)  # [B, S, V]

        bsz, seq_len = y_in.shape
        for b in range(bsz):
            # 対象位置: y_tg が NOTE_ON の位置（= その正解onset token）。
            onset_pos = [i for i in range(seq_len) if int(y_tg[b, i].item()) in note_on_set]
            if not onset_pos:
                continue
            if len(onset_pos) > positions_per_sample:
                onset_pos = sorted(rng.choice(onset_pos, size=int(positions_per_sample), replace=False).tolist())

            for s in onset_pos:
                target_token_id = int(y_tg[b, s].item())

                # 「対応するtime token」= y_in の中で s 以下にある最新 TIM_*
                time_pos_candidates = [j for j in range(int(s) + 1) if int(y_in[b, j].item()) in time_ids]
                if not time_pos_candidates:
                    continue
                time_pos = int(time_pos_candidates[-1])

                base_p = float(probs[b, s, target_token_id].item())
                # 削除/追加対象は対応TIMより前のみ
                prefix_idxs = list(range(0, time_pos))
                if len(prefix_idxs) <= 1:
                    continue

                # 先頭トークンは温存（PRG等）
                prefix_idxs = [i for i in prefix_idxs if i != 0]
                if not include_time_tokens_in_deletion:
                    prefix_idxs = [i for i in prefix_idxs if int(y_in[b, i].item()) not in time_ids]
                if not prefix_idxs:
                    continue

                for r in ratios:
                    k = int(round(len(prefix_idxs) * float(r)))
                    if k <= 0:
                        k = 1
                    k = min(k, len(prefix_idxs))
                    for t in range(trials_per_ratio):
                        ops = [operation] if operation in {"delete", "add"} else ["delete", "add"]
                        for op in ops:
                            if op == "delete":
                                remove_idx = rng.choice(prefix_idxs, size=k, replace=False).tolist()
                                y_in_abl = delete_indices_keep_length(y_in[b], remove_idx, pad_id=pad_id)
                                new_s = int(s) - sum(1 for x in remove_idx if int(x) < int(s))
                            else:
                                # add: GT prefix 内にトークン挿入
                                insert_pos = rng.choice(prefix_idxs, size=k, replace=False).tolist()
                                if add_source == "prefix":
                                    src_tokens = [int(y_in[b, i].item()) for i in prefix_idxs]
                                    insert_tok = rng.choice(src_tokens, size=k, replace=True).tolist()
                                else:
                                    # random vocab token (exclude PAD)
                                    vmax = int(len(vocab.itos))
                                    insert_tok = rng.integers(low=1, high=max(vmax, 2), size=k).tolist()
                                y_in_abl = insert_tokens_keep_length(
                                    y_in[b],
                                    insert_before_positions=insert_pos,
                                    insert_token_ids=insert_tok,
                                    pad_id=pad_id,
                                )
                                new_s = int(s) + sum(1 for x in insert_pos if int(x) < int(s))

                            new_s = max(0, min(int(seq_len - 1), new_s))
                            logits_abl = model(mels[b:b+1], y_in_abl.unsqueeze(0))
                            probs_abl = torch.softmax(logits_abl, dim=-1)
                            ablated_p = float(probs_abl[0, new_s, target_token_id].item())

                            records.append(
                                {
                                    "operation": str(op),
                                    "ratio": float(r),
                                    "trial": int(t),
                                    "target_token_id": int(target_token_id),
                                    "target_pos": int(s),
                                    "time_pos": int(time_pos),
                                    "base_p_target": base_p,
                                    "ablated_p_target": ablated_p,
                                    "delta": float(ablated_p - base_p),
                                    "abs_delta": float(abs(ablated_p - base_p)),
                                }
                            )

        batch_count += 1
        if max_batches > 0 and batch_count >= int(max_batches):
            break

    if not records:
        raise RuntimeError("No ablation records were collected.")

    # まとめ
    summary = {}
    for op in sorted(set(str(x["operation"]) for x in records)):
        for r in ratios:
            rs = [
                x for x in records
                if str(x["operation"]) == op and abs(float(x["ratio"]) - float(r)) < 1e-12
            ]
            if not rs:
                continue
            base = np.array([x["base_p_target"] for x in rs], dtype=np.float64)
            abl = np.array([x["ablated_p_target"] for x in rs], dtype=np.float64)
            delta = np.array([x["delta"] for x in rs], dtype=np.float64)
            abs_delta = np.array([x["abs_delta"] for x in rs], dtype=np.float64)
            summary[f"{op}:{r}"] = {
                "operation": str(op),
                "ratio": float(r),
                "n": int(len(rs)),
                "base_mean": float(base.mean()),
                "ablated_mean": float(abl.mean()),
                "delta_mean": float(delta.mean()),
                "abs_delta_mean": float(abs_delta.mean()),
                "delta_p50": float(np.quantile(delta, 0.5)),
                "delta_p90": float(np.quantile(delta, 0.9)),
                "increase_rate": float((delta > 0).mean()),
                "decrease_rate": float((delta < 0).mean()),
            }

    return {
        "records": records,
        "summary": summary,
        "n_records": int(len(records)),
        "batch_count": int(batch_count),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze how P(target onset token) changes when perturbing tokens "
            "before the corresponding TIM_* token."
        )
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--root", type=str, default="dataset/maestro-v3.0.0")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--input_frames", type=int, default=256)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_songs", type=int, default=1, help="0=all")
    ap.add_argument("--max_batches", type=int, default=20, help="0=all")
    ap.add_argument("--positions_per_sample", type=int, default=8)
    ap.add_argument("--ratios", type=str, default="0.1,0.3,0.5,0.7")
    ap.add_argument("--trials_per_ratio", type=int, default=3)
    ap.add_argument("--include_time_tokens_in_deletion", action="store_true")
    ap.add_argument(
        "--operation",
        type=str,
        default="both",
        choices=["delete", "add", "both"],
        help="context perturbation type",
    )
    ap.add_argument(
        "--add_source",
        type=str,
        default="prefix",
        choices=["prefix", "random_vocab"],
        help="token source for add-operation insertion",
    )
    ap.add_argument(
        "--context_source",
        type=str,
        default="gt",
        choices=["gt"],
        help="which token sequence is perturbed (currently GT MIDI tokens only)",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_json", type=str, default="outputs/onset_context_ablation.json")
    ap.add_argument("--out_png", type=str, default="outputs/onset_context_ablation_plot.png")
    ap.add_argument("--out_csv", type=str, default="outputs/onset_context_ablation_records.csv")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ratios = [float(x.strip()) for x in str(args.ratios).split(",") if x.strip()]
    if not ratios:
        raise SystemExit("No valid ratios")

    device = torch.device(args.device)
    vocab = build_vocab(input_frames=int(args.input_frames), instrument_type="piano", include_note_off=True)

    pairs = collect_maestro_pairs(
        args.root,
        split=args.split,
        max_songs=int(args.max_songs),
        program_id=0,
    )
    if not pairs:
        raise SystemExit("No pairs found.")

    ds = AMTDataset(
        pairs,
        mode="validation",
        sr=16000,
        input_frames=int(args.input_frames),
        vocab=vocab,
    )
    dl = DataLoader(
        ds,
        batch_size=int(args.bs),
        shuffle=False,
        collate_fn=make_collate(vocab),
        num_workers=0,
        pin_memory=False,
    )

    model = load_model(args.ckpt, vocab_size=len(vocab.itos), device=device)
    res = run_ablation(
        model,
        dl,
        vocab=vocab,
        device=device,
        ratios=ratios,
        trials_per_ratio=int(args.trials_per_ratio),
        positions_per_sample=int(args.positions_per_sample),
        include_time_tokens_in_deletion=bool(args.include_time_tokens_in_deletion),
        operation=str(args.operation),
        add_source=str(args.add_source),
        context_source=str(args.context_source),
        max_batches=int(args.max_batches),
        seed=int(args.seed),
    )

    out = {
        "ckpt": str(args.ckpt),
        "split": str(args.split),
        "max_songs": int(args.max_songs),
        "max_batches": int(args.max_batches),
        "positions_per_sample": int(args.positions_per_sample),
        "ratios": ratios,
        "trials_per_ratio": int(args.trials_per_ratio),
        "operation": str(args.operation),
        "add_source": str(args.add_source),
        "context_source": str(args.context_source),
        "include_time_tokens_in_deletion": bool(args.include_time_tokens_in_deletion),
        "n_records": int(res["n_records"]),
        "batch_count": int(res["batch_count"]),
        "summary": res["summary"],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存しやすいように、全recordをCSV化＋可視化
    df = pd.DataFrame(res["records"])
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    png_path = Path(args.out_png)
    _plot_results(df, png_path)

    print("=== Onset Context Ablation ===")
    print(f"ckpt: {args.ckpt}")
    print(
        f"split: {args.split}  context_source: {args.context_source}  "
        f"records: {res['n_records']}  batches: {res['batch_count']}"
    )
    for key in sorted(res["summary"].keys()):
        s = res["summary"][key]
        print(
            f"[{s['operation']} ratio={s['ratio']:.2f}] n={s['n']} "
            f"base={s['base_mean']:.6f} -> ablated={s['ablated_mean']:.6f}  "
            f"delta_mean={s['delta_mean']:.6f}  abs_delta_mean={s['abs_delta_mean']:.6f}"
        )
    print(f"saved json: {out_path}")
    print(f"saved csv : {csv_path}")
    print(f"saved png : {png_path}")


if __name__ == "__main__":
    main()
