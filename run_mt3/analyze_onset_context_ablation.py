from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

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
    max_batches: int,
    seed: int,
) -> dict:
    note_on_ids = torch.tensor(list(vocab.note_on.values()), dtype=torch.long, device=device)
    time_ids = set(int(tid) for tid in vocab.time.values())
    pad_id = int(vocab.pad)

    rng = np.random.default_rng(int(seed))
    records = []
    batch_count = 0

    for mels, y_in, y_tg in dl:
        mels = mels.to(device, non_blocking=True)
        y_in = y_in.to(device, non_blocking=True)
        y_tg = y_tg.to(device, non_blocking=True)

        logits = model(mels, y_in)
        p_on = torch.softmax(logits, dim=-1).index_select(2, note_on_ids).sum(dim=2)  # [B, S]

        bsz, seq_len = y_in.shape
        for b in range(bsz):
            # 対象位置: TIM_* token の位置（その位置で次トークンとしてnote_onが出る確率を見る）
            pos = [
                i for i in range(seq_len)
                if int(y_in[b, i].item()) in time_ids and int(y_tg[b, i].item()) != pad_id
            ]
            if not pos:
                continue
            if len(pos) > positions_per_sample:
                pos = sorted(rng.choice(pos, size=int(positions_per_sample), replace=False).tolist())

            for s in pos:
                base_p = float(p_on[b, s].item())
                prefix_idxs = list(range(0, int(s)))
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
                                # add: prefix内にトークン挿入
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
                            p_on_abl = torch.softmax(logits_abl, dim=-1).index_select(2, note_on_ids).sum(dim=2)
                            ablated_p = float(p_on_abl[0, new_s].item())

                            records.append(
                                {
                                    "operation": str(op),
                                    "ratio": float(r),
                                    "trial": int(t),
                                    "base_p_on": base_p,
                                    "ablated_p_on": ablated_p,
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
            base = np.array([x["base_p_on"] for x in rs], dtype=np.float64)
            abl = np.array([x["ablated_p_on"] for x in rs], dtype=np.float64)
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
            "Analyze how P(note_on) changes when deleting arbitrary tokens "
            "before a TIM_* token."
        )
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--root", type=str, default="dataset/maestro-v3.0.0")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--input_frames", type=int, default=256)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_songs", type=int, default=8, help="0=all")
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
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_json", type=str, default="outputs/onset_context_ablation.json")
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
        "include_time_tokens_in_deletion": bool(args.include_time_tokens_in_deletion),
        "n_records": int(res["n_records"]),
        "batch_count": int(res["batch_count"]),
        "summary": res["summary"],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Onset Context Ablation ===")
    print(f"ckpt: {args.ckpt}")
    print(f"split: {args.split}  records: {res['n_records']}  batches: {res['batch_count']}")
    for key in sorted(res["summary"].keys()):
        s = res["summary"][key]
        print(
            f"[{s['operation']} ratio={s['ratio']:.2f}] n={s['n']} "
            f"base={s['base_mean']:.6f} -> ablated={s['ablated_mean']:.6f}  "
            f"delta_mean={s['delta_mean']:.6f}  abs_delta_mean={s['abs_delta_mean']:.6f}"
        )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
