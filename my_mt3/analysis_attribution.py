from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TokenContext:
    idx: int
    token_id: int
    token_str: str
    kind: Literal["tim", "non", "nof", "other"]
    pitch: int | None
    time_idx: int | None


@dataclass
class TargetToken:
    idx: int
    token_id: int
    token_str: str
    kind: Literal["non", "nof"]
    pitch: int
    time_idx: int | None
    time_ms: float | None
    time_frame: int | None


@dataclass
class AttributionResult:
    token_ids: List[int]
    token_strs: List[str]
    token_contexts: List[TokenContext]
    targets: List[TargetToken]
    nll_base: np.ndarray
    logp_base: np.ndarray
    rows: List[Dict[str, Any]]


def _token_strings(vocab, token_ids: List[int]) -> List[str]:
    out: List[str] = []
    for tid in token_ids:
        if 0 <= int(tid) < len(vocab.itos):
            out.append(str(vocab.itos[int(tid)]))
        else:
            out.append(f"UNK_{int(tid)}")
    return out


def _nll_and_logp_for_targets(logits: torch.Tensor, y_tg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logp = F.log_softmax(logits[0], dim=-1)  # [S,V]
    tgt = y_tg[0].long()
    tgt_logp = logp.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)  # [S]
    nll = -tgt_logp
    return nll, tgt_logp


def _parse_token_contexts(token_ids: List[int], token_strs: List[str]) -> List[TokenContext]:
    contexts: List[TokenContext] = []
    cur_time: int | None = None
    for i, (tid, tok) in enumerate(zip(token_ids, token_strs)):
        kind: Literal["tim", "non", "nof", "other"] = "other"
        pitch: int | None = None
        if tok.startswith("TIM_"):
            kind = "tim"
            cur_time = int(tok.split("_")[1])
        elif tok.startswith("NON_"):
            kind = "non"
            pitch = int(tok.split("_")[1])
        elif tok.startswith("NOF_"):
            kind = "nof"
            pitch = int(tok.split("_")[1])
        contexts.append(
            TokenContext(
                idx=i,
                token_id=int(tid),
                token_str=tok,
                kind=kind,
                pitch=pitch,
                time_idx=cur_time,
            )
        )
    return contexts


def _frame_from_time_idx(time_idx: int, *, step_ms: int, sr: int, hop: int, total_frames: int) -> int:
    t_sec = (float(time_idx) * float(step_ms)) / 1000.0
    f = int(round(t_sec * float(sr) / float(hop)))
    return int(min(max(f, 0), max(total_frames - 1, 0)))


def _select_targets(
    contexts: List[TokenContext],
    *,
    step_ms: int,
    sr: int,
    hop: int,
    total_frames: int,
) -> List[TargetToken]:
    on_pitches = {c.pitch for c in contexts if c.kind == "non" and c.pitch is not None}
    off_pitches = {c.pitch for c in contexts if c.kind == "nof" and c.pitch is not None}
    valid_pitches = on_pitches.intersection(off_pitches)
    targets: List[TargetToken] = []
    for c in contexts:
        if c.kind not in ("non", "nof") or c.pitch not in valid_pitches:
            continue
        t_ms = None if c.time_idx is None else float(c.time_idx * int(step_ms))
        t_fr = (
            None
            if c.time_idx is None
            else _frame_from_time_idx(c.time_idx, step_ms=step_ms, sr=sr, hop=hop, total_frames=total_frames)
        )
        targets.append(
            TargetToken(
                idx=c.idx,
                token_id=c.token_id,
                token_str=c.token_str,
                kind=c.kind,
                pitch=int(c.pitch),
                time_idx=c.time_idx,
                time_ms=t_ms,
                time_frame=t_fr,
            )
        )
    return targets


def _prefix_drop_candidates(
    contexts: List[TokenContext],
    target: TargetToken,
    *,
    mode: Literal["pair_and_offset", "offset_only"],
) -> List[int]:
    out: List[int] = []
    for c in contexts[: target.idx]:
        if c.pitch != target.pitch:
            continue
        if mode == "pair_and_offset" and c.kind in ("non", "nof"):
            out.append(c.idx)
        elif mode == "offset_only" and c.kind == "nof":
            out.append(c.idx)
    return out


def _drop_empty_time_tokens(
    contexts: List[TokenContext],
    *,
    prefix_end: int,
    dropped: set[int],
) -> set[int]:
    out = set(dropped)
    tim_idxs = [c.idx for c in contexts[:prefix_end] if c.kind == "tim"]
    for i, t_idx in enumerate(tim_idxs):
        seg_start = t_idx + 1
        seg_end = tim_idxs[i + 1] if i + 1 < len(tim_idxs) else prefix_end
        seg_events = [j for j in range(seg_start, seg_end) if contexts[j].kind in ("non", "nof")]
        if seg_events and all(j in out for j in seg_events):
            out.add(t_idx)
    return out


def _build_prefix_drops(
    contexts: List[TokenContext],
    target: TargetToken,
    *,
    mode: Literal["pair_and_offset", "offset_only"],
    ratio: float,
    rng: np.random.Generator,
) -> List[int]:
    candidates = _prefix_drop_candidates(contexts, target, mode=mode)
    if not candidates:
        return []
    k = max(1, int(round(float(ratio) * len(candidates))))
    k = min(k, len(candidates))
    picked = set(int(x) for x in rng.choice(np.array(candidates, dtype=np.int64), size=k, replace=False))
    picked = _drop_empty_time_tokens(contexts, prefix_end=target.idx, dropped=picked)
    return sorted(picked)


def _apply_prefix_drop_to_yin(y_in: torch.Tensor, drop_tg_idxs: List[int], pad_id: int) -> torch.Tensor:
    y_mod = y_in.clone()
    mapped = [i + 1 for i in drop_tg_idxs if (i + 1) < y_mod.size(1)]
    if mapped:
        y_mod[0, torch.tensor(mapped, device=y_mod.device)] = int(pad_id)
    return y_mod


def apply_source_noise(
    mel: torch.Tensor,
    *,
    center_frame: int,
    width_ratio: float,
    sigma: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    x = mel.clone()
    T = x.size(1)
    if T <= 0 or width_ratio <= 0 or sigma <= 0:
        return x
    width = max(1, int(round(float(width_ratio) * T)))
    c = int(min(max(center_frame, 0), T - 1))
    s = max(0, c - width // 2)
    e = min(T, s + width)
    seed = int(rng.integers(0, 2**31 - 1))
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    noise = torch.randn(x[:, s:e, :].shape, device=x.device, generator=g) * float(sigma)
    x[:, s:e, :] = x[:, s:e, :] + noise
    return x


def apply_source_mask_band(
    mel: torch.Tensor,
    *,
    center_frame: int,
    width_ratio: float,
    fill: Literal["zero", "mean"] = "zero",
) -> torch.Tensor:
    x = mel.clone()
    T = x.size(1)
    if T <= 0 or width_ratio <= 0:
        return x
    width = max(1, int(round(float(width_ratio) * T)))
    c = int(min(max(center_frame, 0), T - 1))
    s = max(0, c - width // 2)
    e = min(T, s + width)
    if fill == "mean":
        x[:, s:e, :] = float(x.mean().item())
    else:
        x[:, s:e, :] = 0.0
    return x


@torch.no_grad()
def analyze_chunk(
    model,
    *,
    mel: torch.Tensor,            # [1,T,F]
    y_in: torch.Tensor,           # [1,S]
    y_tg: torch.Tensor,           # [1,S]
    vocab,
    pad_id: int,
    step_ms: int,
    sr: int,
    hop: int,
    prefix_drop_ratios: List[float],
    prefix_drop_modes: List[str],
    noise_sigmas: List[float],
    noise_width_ratios: List[float],
    mask_width_ratios: List[float] | None = None,
    mask_fill: str = "zero",
    noise_repeats: int = 3,
    seed: int = 1234,
) -> AttributionResult:
    model.eval()
    mem_base = model.enc(mel)
    logits_base = model.dec(y_in, mem_base)
    nll_base, logp_base = _nll_and_logp_for_targets(logits_base, y_tg)

    token_ids = [int(x) for x in y_tg[0].detach().cpu().tolist()]
    token_strs = _token_strings(vocab, token_ids)
    contexts = _parse_token_contexts(token_ids, token_strs)
    targets = _select_targets(
        contexts,
        step_ms=int(step_ms),
        sr=int(sr),
        hop=int(hop),
        total_frames=int(mel.size(1)),
    )

    prefix_modes: List[Literal["pair_and_offset", "offset_only"]] = []
    for mode in prefix_drop_modes:
        if mode not in ("pair_and_offset", "offset_only"):
            raise ValueError(f"Unsupported prefix mode: {mode}")
        prefix_modes.append(mode)
    mfill: Literal["zero", "mean"] = "mean" if str(mask_fill) == "mean" else "zero"
    mwidths = [float(x) for x in (mask_width_ratios or []) if float(x) > 0.0]

    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, Any]] = []

    for tgt in targets:
        base_logp = float(logp_base[tgt.idx].item())
        base_nll = float(nll_base[tgt.idx].item())

        for mode in prefix_modes:
            for ratio in prefix_drop_ratios:
                drop_idxs = _build_prefix_drops(contexts, tgt, mode=mode, ratio=float(ratio), rng=rng)
                if not drop_idxs:
                    rows.append(
                        {
                            "experiment_type": "prefix_drop",
                            "prefix_mode": mode,
                            "drop_ratio": float(ratio),
                            "noise_sigma": None,
                            "noise_width_ratio": None,
                            "noise_repeats": None,
                            "target_token_idx": int(tgt.idx),
                            "target_token_id": int(tgt.token_id),
                            "target_token": tgt.token_str,
                            "target_kind": tgt.kind,
                            "target_pitch": int(tgt.pitch),
                            "target_time_idx": tgt.time_idx,
                            "target_time_ms": tgt.time_ms,
                            "target_frame": tgt.time_frame,
                            "logp_base": base_logp,
                            "nll_base": base_nll,
                            "logp_perturbed": None,
                            "nll_perturbed": None,
                            "delta_nll": None,
                            "delta_nll_std": None,
                            "skipped": 1,
                            "skip_reason": "no_drop_candidates",
                        }
                    )
                    continue
                y_mod = _apply_prefix_drop_to_yin(y_in, drop_idxs, pad_id=int(pad_id))
                logits_mod = model.dec(y_mod, mem_base)
                nll_mod, logp_mod = _nll_and_logp_for_targets(logits_mod, y_tg)
                mod_logp = float(logp_mod[tgt.idx].item())
                mod_nll = float(nll_mod[tgt.idx].item())
                rows.append(
                    {
                        "experiment_type": "prefix_drop",
                        "prefix_mode": mode,
                        "drop_ratio": float(ratio),
                        "noise_sigma": None,
                        "noise_width_ratio": None,
                        "noise_repeats": None,
                        "target_token_idx": int(tgt.idx),
                        "target_token_id": int(tgt.token_id),
                        "target_token": tgt.token_str,
                        "target_kind": tgt.kind,
                        "target_pitch": int(tgt.pitch),
                        "target_time_idx": tgt.time_idx,
                        "target_time_ms": tgt.time_ms,
                        "target_frame": tgt.time_frame,
                        "logp_base": base_logp,
                        "nll_base": base_nll,
                        "logp_perturbed": mod_logp,
                        "nll_perturbed": mod_nll,
                        "delta_nll": float(mod_nll - base_nll),
                        "delta_nll_std": 0.0,
                        "skipped": 0,
                        "skip_reason": "",
                    }
                )

        for sigma in noise_sigmas:
            for width_ratio in noise_width_ratios:
                if tgt.time_frame is None:
                    rows.append(
                        {
                            "experiment_type": "source_noise",
                            "prefix_mode": None,
                            "drop_ratio": None,
                            "noise_sigma": float(sigma),
                            "noise_width_ratio": float(width_ratio),
                            "noise_repeats": int(noise_repeats),
                            "target_token_idx": int(tgt.idx),
                            "target_token_id": int(tgt.token_id),
                            "target_token": tgt.token_str,
                            "target_kind": tgt.kind,
                            "target_pitch": int(tgt.pitch),
                            "target_time_idx": tgt.time_idx,
                            "target_time_ms": tgt.time_ms,
                            "target_frame": tgt.time_frame,
                            "logp_base": base_logp,
                            "nll_base": base_nll,
                            "logp_perturbed": None,
                            "nll_perturbed": None,
                            "delta_nll": None,
                            "delta_nll_std": None,
                            "skipped": 1,
                            "skip_reason": "target_has_no_time",
                        }
                    )
                    continue
                lp_vals: List[float] = []
                nl_vals: List[float] = []
                for _ in range(max(1, int(noise_repeats))):
                    mel_mod = apply_source_noise(
                        mel,
                        center_frame=int(tgt.time_frame),
                        width_ratio=float(width_ratio),
                        sigma=float(sigma),
                        rng=rng,
                    )
                    logits_mod = model.dec(y_in, model.enc(mel_mod))
                    nll_mod, logp_mod = _nll_and_logp_for_targets(logits_mod, y_tg)
                    lp_vals.append(float(logp_mod[tgt.idx].item()))
                    nl_vals.append(float(nll_mod[tgt.idx].item()))
                mod_logp = float(np.mean(lp_vals))
                mod_nll = float(np.mean(nl_vals))
                rows.append(
                    {
                        "experiment_type": "source_noise",
                        "prefix_mode": None,
                        "drop_ratio": None,
                        "noise_sigma": float(sigma),
                        "noise_width_ratio": float(width_ratio),
                        "noise_repeats": int(noise_repeats),
                        "target_token_idx": int(tgt.idx),
                        "target_token_id": int(tgt.token_id),
                        "target_token": tgt.token_str,
                        "target_kind": tgt.kind,
                        "target_pitch": int(tgt.pitch),
                        "target_time_idx": tgt.time_idx,
                        "target_time_ms": tgt.time_ms,
                        "target_frame": tgt.time_frame,
                        "logp_base": base_logp,
                        "nll_base": base_nll,
                        "logp_perturbed": mod_logp,
                        "nll_perturbed": mod_nll,
                        "delta_nll": float(mod_nll - base_nll),
                        "delta_nll_std": float(np.std(nl_vals)) if len(nl_vals) > 1 else 0.0,
                        "skipped": 0,
                        "skip_reason": "",
                    }
                )

        for width_ratio in mwidths:
            if tgt.time_frame is None:
                rows.append(
                    {
                        "experiment_type": "source_mask",
                        "prefix_mode": None,
                        "drop_ratio": None,
                        "noise_sigma": None,
                        "noise_width_ratio": None,
                        "mask_width_ratio": float(width_ratio),
                        "mask_fill": mfill,
                        "noise_repeats": None,
                        "target_token_idx": int(tgt.idx),
                        "target_token_id": int(tgt.token_id),
                        "target_token": tgt.token_str,
                        "target_kind": tgt.kind,
                        "target_pitch": int(tgt.pitch),
                        "target_time_idx": tgt.time_idx,
                        "target_time_ms": tgt.time_ms,
                        "target_frame": tgt.time_frame,
                        "logp_base": base_logp,
                        "nll_base": base_nll,
                        "logp_perturbed": None,
                        "nll_perturbed": None,
                        "delta_nll": None,
                        "delta_nll_std": None,
                        "skipped": 1,
                        "skip_reason": "target_has_no_time",
                    }
                )
                continue
            mel_mod = apply_source_mask_band(
                mel,
                center_frame=int(tgt.time_frame),
                width_ratio=float(width_ratio),
                fill=mfill,
            )
            logits_mod = model.dec(y_in, model.enc(mel_mod))
            nll_mod, logp_mod = _nll_and_logp_for_targets(logits_mod, y_tg)
            mod_logp = float(logp_mod[tgt.idx].item())
            mod_nll = float(nll_mod[tgt.idx].item())
            rows.append(
                {
                    "experiment_type": "source_mask",
                    "prefix_mode": None,
                    "drop_ratio": None,
                    "noise_sigma": None,
                    "noise_width_ratio": None,
                    "mask_width_ratio": float(width_ratio),
                    "mask_fill": mfill,
                    "noise_repeats": None,
                    "target_token_idx": int(tgt.idx),
                    "target_token_id": int(tgt.token_id),
                    "target_token": tgt.token_str,
                    "target_kind": tgt.kind,
                    "target_pitch": int(tgt.pitch),
                    "target_time_idx": tgt.time_idx,
                    "target_time_ms": tgt.time_ms,
                    "target_frame": tgt.time_frame,
                    "logp_base": base_logp,
                    "nll_base": base_nll,
                    "logp_perturbed": mod_logp,
                    "nll_perturbed": mod_nll,
                    "delta_nll": float(mod_nll - base_nll),
                    "delta_nll_std": 0.0,
                    "skipped": 0,
                    "skip_reason": "",
                }
            )

    return AttributionResult(
        token_ids=token_ids,
        token_strs=token_strs,
        token_contexts=contexts,
        targets=targets,
        nll_base=nll_base.detach().cpu().numpy(),
        logp_base=logp_base.detach().cpu().numpy(),
        rows=rows,
    )


def summarize_result(res: AttributionResult) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["num_tokens"] = int(len(res.token_ids))
    out["num_targets"] = int(len(res.targets))
    out["num_rows_total"] = int(len(res.rows))

    valid_rows = [r for r in res.rows if int(r.get("skipped", 0)) == 0]
    out["num_rows_valid"] = int(len(valid_rows))
    out["num_rows_skipped"] = int(len(res.rows) - len(valid_rows))
    if not valid_rows:
        out["message"] = "No valid experiment rows."
        return out

    def _stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        d_nll = np.array([float(r["delta_nll"]) for r in rows], dtype=np.float64)
        return {
            "mean_delta_nll": float(np.mean(d_nll)),
            "p50_delta_nll": float(np.percentile(d_nll, 50)),
            "p90_delta_nll": float(np.percentile(d_nll, 90)),
        }

    prefix_rows = [r for r in valid_rows if r["experiment_type"] == "prefix_drop"]
    noise_rows = [r for r in valid_rows if r["experiment_type"] == "source_noise"]
    if prefix_rows:
        out["prefix_overall"] = _stats(prefix_rows)
    if noise_rows:
        out["source_noise_overall"] = _stats(noise_rows)

    prefix_by_cond: Dict[str, Any] = {}
    for r in prefix_rows:
        key = f"{r['prefix_mode']}|{r['drop_ratio']}"
        prefix_by_cond.setdefault(key, []).append(r)
    out["prefix_by_condition"] = {k: _stats(v) for k, v in prefix_by_cond.items()}

    noise_by_cond: Dict[str, Any] = {}
    for r in noise_rows:
        key = f"sigma={r['noise_sigma']}|width={r['noise_width_ratio']}"
        noise_by_cond.setdefault(key, []).append(r)
    out["source_noise_by_condition"] = {k: _stats(v) for k, v in noise_by_cond.items()}
    return out
