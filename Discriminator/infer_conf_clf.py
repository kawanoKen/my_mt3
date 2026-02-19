from __future__ import annotations

import argparse
from pathlib import Path
import csv
import torch

from conf_clf_model import ConfClfCfg, TransformerConfidenceClf
from midi_tokenizer import MidiTokCfg, AugCfg, load_piano_notes, apply_augmentation, midi_notes_to_tokens
from conf_data import pad_and_add_cls


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise ValueError("ckpt must contain 'cfg' used for training.")

    # rebuild tokenizer cfg (must match training)
    tok_cfg = MidiTokCfg(
        time_step_sec=float(cfg_dict.get("time_step_sec", 0.01)),
        max_shift_steps=int(cfg_dict.get("max_shift_steps", 100)),
    )
    vocab_size = int(cfg_dict.get("vocab_size", tok_cfg.vocab_size()))

    mcfg = ConfClfCfg(
        vocab_size=vocab_size,
        max_len=int(cfg_dict.get("max_len", 512)),
        d_model=int(cfg_dict.get("d_model", 256)),
        n_layers=int(cfg_dict.get("n_layers", 6)),
        n_heads=int(cfg_dict.get("n_heads", 8)),
        d_ff=int(cfg_dict.get("d_ff", 1024)),
        dropout=float(cfg_dict.get("dropout", 0.1)),
        pad_id=tok_cfg.pad_id,
        cls_id=tok_cfg.cls_id,
    )
    model = TransformerConfidenceClf(mcfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, tok_cfg


def chunk_windows(seq: torch.Tensor, win_len: int, hop: int):
    N = seq.numel()
    if N <= win_len:
        return [(0, N)]
    out = []
    s = 0
    while s < N:
        e = min(N, s + win_len)
        out.append((s, e))
        if e == N:
            break
        s += hop
    return out


@torch.no_grad()
def score_tokens(model: TransformerConfidenceClf, tok_cfg: MidiTokCfg, seq: torch.Tensor, *, device: str, hop: int):
    max_len = model.cfg.max_len
    win_len = max_len - 1
    spans = chunk_windows(seq, win_len, hop=hop)

    scores = []
    for wi, (s, e) in enumerate(spans):
        w = seq[s:e]
        tokens, attn = pad_and_add_cls(w, max_len=max_len, pad_id=tok_cfg.pad_id, cls_id=tok_cfg.cls_id)
        tokens = tokens.unsqueeze(0).to(device)
        attn = attn.unsqueeze(0).to(device)
        conf = model.score(tokens, attn_mask=attn)[0].item()
        scores.append((wi, s, e, conf))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--midi", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--hop", type=int, default=256)

    # optional: apply same augmentation at inference for analysis (usually off)
    ap.add_argument("--apply_aug", action="store_true")
    ap.add_argument("--pitch_shift_min", type=int, default=0)
    ap.add_argument("--pitch_shift_max", type=int, default=0)
    ap.add_argument("--time_scale_min", type=float, default=1.0)
    ap.add_argument("--time_scale_max", type=float, default=1.0)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    model, tok_cfg = load_model(Path(args.ckpt), device=device)

    notes = load_piano_notes(args.midi)
    if args.apply_aug:
        aug = AugCfg(
            pitch_shift_min=args.pitch_shift_min,
            pitch_shift_max=args.pitch_shift_max,
            pitch_shift_prob=1.0,
            time_scale_min=args.time_scale_min,
            time_scale_max=args.time_scale_max,
            time_scale_prob=1.0,
        )
        notes = apply_augmentation(notes, aug=aug)

    seq = midi_notes_to_tokens(notes, tok_cfg)
    scores = score_tokens(model, tok_cfg, seq, device=device, hop=args.hop)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_idx", "token_start", "token_end", "confidence"])
        for wi, s, e, conf in scores:
            w.writerow([wi, s, e, f"{conf:.6f}"])

    print("[OK] wrote:", out_csv)


if __name__ == "__main__":
    main()
