# my_mt3/train_DA.py


import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_mt3.model import MT3Mini, EMATeacher
from my_mt3.decode_kv import FastDecoderKV, pseudo_label_with_kvcache
from my_mt3.tokenizer import Vocab, INPUT_FRAMES
from my_mt3.dataset import AMTDataset
from my_mt3.dataset_unlabeled import AMTRealDataset
from my_mt3.discriminator import Discriminator
from my_mt3.audio import ensure_wave_cache, DEFAULT_SR
import os
import itertools

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Function
from my_mt3.train import _maybe_cache_pairs_map, make_collate

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class NoteSpan:
    # token indices in the *target sequence* (y_tg side) that correspond to this note
    tok_ids: List[int]   # e.g., [k_on, k_off, ...]  (indices in y sequence excluding BOS or aligned as you decide)

def decode_notes_to_spans(token_ids: List[int], vocab) -> List[NoteSpan]:
    """
    token_ids: 生成されたトークン列。
      - teacherのgreedy decodeが BOS を先頭に付けるなら [BOS, ...]
      - 付けないなら [...]. どちらでも動くようにしてある。
    vocab: あなたの Vocab（program/time/note_on/note_off/eos/bos を持つ）

    返り値:
      List[NoteSpan] where NoteSpan.tok_ids are indices in y_tg (= token_ids[(start+1):])
    """

    # ---- BOSの有無を吸収 ----
    start = 1 if (hasattr(vocab, "bos") and len(token_ids) > 0 and token_ids[0] == vocab.bos) else 0

    # y_tg は token_ids[(start+1):] に対応するので、
    # token_ids[i] の y_tg index は i - (start+1)
    def to_tg_index(i: int) -> int:
        return i - (start + 1)

    # ---- reverse maps: id -> pitch ----
    # vocab.note_on: Dict[pitch -> id]
    id2on: Dict[int, int] = {tid: p for p, tid in vocab.note_on.items()}
    has_off = getattr(vocab, "note_off", None) is not None
    id2off: Dict[int, int] = {}
    if has_off:
        id2off = {tid: p for p, tid in vocab.note_off.items()}

    eos_id = getattr(vocab, "eos", None)

    spans: List[NoteSpan] = []

    if not has_off:
        # ---- NoteOff無し: NOTE_ON 1個 = 1 note ----
        for i in range(start, len(token_ids)):
            tid = token_ids[i]
            if eos_id is not None and tid == eos_id:
                break
            if tid in id2on:
                tg = to_tg_index(i)
                if tg >= 0:
                    spans.append(NoteSpan(tok_ids=[tg]))
        return spans

    # ---- NoteOffあり: on/off をペアリングし、[on_idx, off_idx] を span とする ----
    open_on: Dict[int, int] = {}  # pitch -> tg_index_of_on
    for i in range(start, len(token_ids)):
        tid = token_ids[i]
        if eos_id is not None and tid == eos_id:
            break

        if tid in id2on:
            p = id2on[tid]
            tg = to_tg_index(i)
            if tg >= 0:
                open_on[p] = tg

        elif tid in id2off:
            p = id2off[tid]
            tg = to_tg_index(i)
            if tg >= 0 and p in open_on:
                spans.append(NoteSpan(tok_ids=[open_on[p], tg]))
                del open_on[p]

    # 未クローズのノートは NOTE_ON 単体として扱う（安全側）
    for p, on_tg in open_on.items():
        spans.append(NoteSpan(tok_ids=[on_tg]))

    return spans



def build_note_confidences(note_spans: List[NoteSpan], pmax_1d: torch.Tensor, margin_1d: torch.Tensor):
    """
    pmax_1d: [S-1]  (BOSの次のステップからのpmax)
    margin_1d: [S-1]
    return:
      conf1, conf2: np arrays [N_notes]
    """
    conf1 = []
    conf2 = []
    for ns in note_spans:
        idx = [i for i in ns.tok_ids if 0 <= i < pmax_1d.numel()]
        if len(idx) == 0:
            conf1.append(0.0); conf2.append(0.0)
            continue
        c1 = float(pmax_1d[idx].mean().item())
        c2 = float(margin_1d[idx].mean().item())
        conf1.append(c1); conf2.append(c2)
    return np.array(conf1), np.array(conf2)

def make_pseudo_token_mask_from_notes(
    note_spans: List[NoteSpan],
    conf1: np.ndarray,
    conf2: np.ndarray,
    seq_len_no_bos: int,
    *,
    top_frac: float = 0.2,
    bot_frac: float = 0.2,
    w1: float = 0.5,
    w2: float = 0.5,
):
    """
    returns mask: torch.bool [seq_len_no_bos], True=compute CE
    """
    n = len(note_spans)
    if n == 0:
        return torch.zeros((seq_len_no_bos,), dtype=torch.bool)

    score = w1 * conf1 + w2 * conf2
    order = np.argsort(score)  # ascending

    k_top = max(1, int(round(n * top_frac)))
    k_bot = max(1, int(round(n * bot_frac)))

    idx_bot = set(order[:k_bot].tolist())
    idx_top = set(order[-k_top:].tolist())
    keep_notes = idx_top | idx_bot

    mask = torch.zeros((seq_len_no_bos,), dtype=torch.bool)
    for i, ns in enumerate(note_spans):
        if i not in keep_notes:
            continue
        for t in ns.tok_ids:
            if 0 <= t < seq_len_no_bos:
                mask[t] = True
    return mask


def apply_mask_to_targets(y_tg: torch.Tensor, token_mask: torch.Tensor, ignore_index: int):
    """
    y_tg: [B, S-1]
    token_mask: [B, S-1] bool, True=keep
    """
    y = y_tg.clone()
    y[~token_mask] = ignore_index
    return y


def make_collate_real():
    def _collate(batch):
        # batch: List[Tensor[T,F]]
        return nn.utils.rnn.pad_sequence(batch, batch_first=True)  # [B,T,F]（Tは固定のはずだが保険）
    return _collate

@torch.no_grad()
def eval_loop_ddp(model_ddp, dl, crit, device):
    """
    全rankで val を回し、loss を global average する
    """
    model_ddp.eval()
    total_loss = torch.zeros(1, device=device)
    n_batches = torch.zeros(1, device=device)

    for mels, y_in, y_tg in dl:
        mels, y_in, y_tg = mels.to(device, non_blocking=True), y_in.to(device, non_blocking=True), y_tg.to(device, non_blocking=True)
        logits = model_ddp(mels, y_in)
        loss = crit(logits.reshape(-1, logits.size(-1)), y_tg.reshape(-1))
        total_loss += loss.detach()
        n_batches += 1

    # allreduce
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)

    return (total_loss / torch.clamp(n_batches, min=1)).item()


def train_loop_distributed_DA_confusion(
    pairs,
    *,
    vocab: Vocab,
    # ---- Domain Adaptation (DC専用) ----
    use_dc: bool = True,
    pairs_real: dict | None = None ,            # {"train": [wav_path, ...]} 必須
    lambda_adv: float = 0.01,         # 論文既定
    lr_t: float = 2e-4,               # Transformer(E,D)
    lr_c: float = 1e-4,               # Discriminator
    chunk_frames: int | None = None,  # 0.1s相当。未指定なら hop, sr から自動決定
    disc_hidden: int = 256,
    # ---- SSL (pseudo) ----
    use_pseudo: bool = True,
    pseudo_start_epoch: int = 3,
    ema_decay: float = 0.999,
    unsup_weight: float = 1.0,
    pseudo_max_len: int = 1024,
    top_frac: float = 0.2,
    bot_frac: float = 0.2,
    # ---- 既存 ----
    epochs=5,
    bs=8,
    save_every=10,
    save_dir="checkpoints",
    use_cache: bool = True,
    cache_dir: str = "cache/wave_sr16000",
    sr: int = DEFAULT_SR,
    num_workers: int = 2,
):
    # ---- init DDP ----
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        if use_cache and cache_dir:
            cache_dir_abs = os.path.abspath(cache_dir)
            os.makedirs(cache_dir_abs, exist_ok=True)
            cache_dir = cache_dir_abs
    dist.barrier()

    # ---- broadcast cache_dir (existing behavior) ----
    if use_cache and cache_dir:
        cache_dir_bytes = os.path.abspath(cache_dir).encode("utf-8") if rank == 0 else b""
        length_tensor = torch.tensor([len(cache_dir_bytes)], dtype=torch.int32, device=device)
        dist.broadcast(length_tensor, src=0)
        buf = torch.empty((int(length_tensor.item()),), dtype=torch.uint8, device=device)
        if rank == 0:
            buf.copy_(torch.tensor(list(cache_dir_bytes), dtype=torch.uint8, device=device))
        dist.broadcast(buf, src=0)
        cache_dir = bytes(buf.cpu().numpy().tolist()).decode("utf-8")

    # ---- cache synth pairs ----
    pairs = _maybe_cache_pairs_map(pairs, sr=sr, cache_dir=(cache_dir if use_cache else None))

    # ---- auto chunk_frames for 0.1s ----
    # 1 frame sec = hop/sr (AMTDataset hop=256, sr=16000 => 16ms)
    # 0.1 / 0.016 = 6.25 => 6 or 7
    if chunk_frames is None:
        # AMTDatasetのhopは引数で受け取っていないので、あなたのデフォ hop=256 を想定
        # hopを外から変えている場合は chunk_frames を明示指定してください
        hop = 256
        frames_per_sec = sr / float(hop)
        chunk_frames = max(1, int(round(0.1 * frames_per_sec)))  # ≈6

    # ---- datasets (synth) ----
    train_ds = AMTDataset(pairs["train"], mode="train", sr=sr, vocab=vocab)
    val_ds   = AMTDataset(pairs["validation"], mode="validation", sr=sr, vocab=vocab)

    # ---- samplers/loaders (synth) ----
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=make_collate(vocab),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=bs,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=make_collate(vocab),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # ---- real loader（DCまたはpseudoで使用） ----
    real_dl = None
    if (use_dc or use_pseudo):
        if pairs_real is None or "train" not in pairs_real:
            raise ValueError("pairs_real={'train':[wav,...]} が必須です（DC/pseudoで使用）")
        real_wavs = pairs_real["train"]
        real_ds = AMTRealDataset(real_wavs, sr=sr, hop=256, input_frames=INPUT_FRAMES, n_fft=2048, n_mels=256,)
        real_sampler = DistributedSampler(real_ds, shuffle=True, drop_last=True)
        real_dl = DataLoader(
            real_ds,
            batch_size=bs,
            sampler=real_sampler,
            shuffle=False,
            collate_fn=make_collate_real(),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(num_workers > 0),
        )

    # ---- models ----
    model = MT3Mini(vocab_size=len(vocab.itos)).to(device)
    model_ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    crit_ce = nn.CrossEntropyLoss(ignore_index=vocab.pad)
    bce = nn.BCEWithLogitsLoss()

    # Optimizers / Discriminator
    disc_ddp = None
    if use_dc:
        disc = Discriminator(d=384, hidden=disc_hidden).to(device)
        disc_ddp = DDP(disc, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        opt_c = optim.AdamW(disc_ddp.parameters(), lr=lr_c)
    opt_t = optim.AdamW(model_ddp.parameters(), lr=lr_t)

    if rank == 0:
        msg = f"[DDP-DA] world={world} | synth_train_songs={len(train_ds)} | val_songs={len(val_ds)}"
        if real_dl is not None:
            msg += f" | real_wavs={len(real_ds)}"
        if use_dc:
            msg += f" | DC(lambda_adv={lambda_adv}, chunk_frames={chunk_frames})"
        print(msg)
        # DA損失ログCSVを初期化
        try:
            import csv as _csv
            with open(os.path.join(save_dir, "da_losses.csv"), "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["epoch", "train_total", "train_sup", "train_adv", "train_unsup", "train_disc", "val_loss"])
        except Exception:
            pass
    ema = EMATeacher(model_ddp.module, decay=ema_decay)
    # ---- loop ----
    for ep in range(epochs):
        train_sampler.set_epoch(ep)
        if (use_dc or use_pseudo):
            real_sampler.set_epoch(ep)

        model_ddp.train()
        if use_dc and disc_ddp is not None:
            disc_ddp.train()

        running_total = 0.0
        running_sup = 0.0
        running_adv = 0.0
        running_unsup = 0.0
        running_disc = 0.0
        n_batches = 0
        real_iter = itertools.cycle(real_dl) if real_dl is not None else None

        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}", unit="batch", disable=(rank != 0))
        if use_pseudo:
            fast_dec = FastDecoderKV(ema.teacher.dec, max_len=pseudo_max_len).to(device).eval()

        for mels_s, y_in_s, y_tg_s in pbar:
            mels_r = next(real_iter) if real_iter is not None else None
            if mels_r is not None:
                mels_r = mels_r.to(device, non_blocking=True)

            mels_s = mels_s.to(device, non_blocking=True)
            y_in_s = y_in_s.to(device, non_blocking=True)
            y_tg_s = y_tg_s.to(device, non_blocking=True)

            # ===== encoder outputs =====
            mem_s = model_ddp.module.enc(mels_s)
            mem_r = model_ddp.module.enc(mels_r) if mels_r is not None else None

            # ========= (A) Discriminator step (Eq.1) =========
            loss_disc = torch.zeros((), device=device)
            if use_dc and disc_ddp is not None and mem_r is not None:
                logit_s = disc_ddp(mem_s.detach(), chunk_frames=chunk_frames)
                logit_r = disc_ddp(mem_r.detach(), chunk_frames=chunk_frames)
                loss_disc = bce(logit_s, torch.zeros_like(logit_s)) + bce(logit_r, torch.ones_like(logit_r))
                opt_c.zero_grad(set_to_none=True)
                loss_disc.backward()
                opt_c.step()

            # ========= (B) Student step: supervised + adv + pseudo =========
            if use_dc and disc_ddp is not None:
                for p in disc_ddp.parameters():
                    p.requires_grad_(False)

            # (1) supervised CE on synth
            logits_s = model_ddp.module.dec(y_in_s, mem_s)
            loss_sup = crit_ce(logits_s.reshape(-1, logits_s.size(-1)), y_tg_s.reshape(-1))

            # (2) adversarial confusion on synth+real
            loss_adv = torch.zeros((), device=device)
            if use_dc and disc_ddp is not None and mem_r is not None:
                logit_s2 = disc_ddp(mem_s, chunk_frames=chunk_frames)
                logit_r2 = disc_ddp(mem_r, chunk_frames=chunk_frames)
                half_s = 0.5 * torch.ones_like(logit_s2)
                half_r = 0.5 * torch.ones_like(logit_r2)
                loss_adv = bce(logit_s2, half_s) + bce(logit_r2, half_r)

            # (3) pseudo-label loss on real (start from pseudo_start_epoch)
            loss_unsup = torch.zeros((), device=device)

            if use_pseudo and real_dl is not None and (ep + 1) >= pseudo_start_epoch:
                # teacher generate pseudo seq + token confidences
                out, pmax, margin = pseudo_label_with_kvcache(
                    teacher=ema.teacher,
                    fast_dec=fast_dec,
                    mel=mels_r,
                    program_id=0,
                    vocab=vocab,
                    max_new_tokens=pseudo_max_len,     # ここが “生成長”
                    return_with_prefix=False,          # outはPRG無し
                )

                B = out.size(0)
                prg_id = int(vocab.program["PRG_0"])
                prg = torch.full((B, 1), prg_id, dtype=torch.long, device=out.device)
                y_in_p = torch.cat([prg, out[:, :-1]], dim=1)
                y_tg_p = out

                # ---- NOTE: token->note mapping is project-specific ----
                # You MUST provide:
                #   note_spans = decode_notes_to_spans(y_pseudo[b].tolist(), vocab)
                # that returns List[NoteSpan] where tok_ids are indices in y_tg_p[b]
                token_masks = []
                for b in range(out.size(0)):
                    # pmax/margin are for steps excluding BOS: they align with y_tg_p positions
                    pmax_b = pmax[b]      # [S-1]
                    margin_b = margin[b]  # [S-1]

                    note_spans = decode_notes_to_spans(out[b].tolist(), vocab)

                    conf1, conf2 = build_note_confidences(note_spans, pmax_b, margin_b)
                    mask_b = make_pseudo_token_mask_from_notes(
                        note_spans, conf1, conf2, seq_len_no_bos=y_tg_p.size(1),
                        top_frac=top_frac, bot_frac=bot_frac
                    )
                    token_masks.append(mask_b)

                token_mask = torch.stack(token_masks, dim=0).to(device)  # [B,S-1] bool

                y_tg_masked = apply_mask_to_targets(y_tg_p, token_mask, ignore_index=vocab.pad)

                # student predicts on real with teacher forcing using full pseudo prefix
                logits_r = model_ddp.module.dec(y_in_p.to(device), mem_r)
                loss_unsup = crit_ce(logits_r.reshape(-1, logits_r.size(-1)), y_tg_masked.to(device).reshape(-1))

            loss_total = loss_sup + lambda_adv * loss_adv + unsup_weight * loss_unsup

            opt_t.zero_grad(set_to_none=True)
            loss_total.backward()
            opt_t.step()

            if use_dc and disc_ddp is not None:
                for p in disc_ddp.parameters():
                    p.requires_grad_(True)

            # EMA update（pseudoを使うときに有効）
            if use_pseudo and (ep + 1) >= pseudo_start_epoch:
                ema.update(model_ddp.module)

            # accumulate
            running_total += float(loss_total.item())
            running_sup += float(loss_sup.item())
            running_adv += float(loss_adv.item())
            running_unsup += float(loss_unsup.item())
            running_disc += float(loss_disc.item())
            n_batches += 1
            if rank == 0:
                pbar.set_postfix(
                    loss=f"{loss_total.item():.3f}",
                    sup=f"{loss_sup.item():.3f}",
                    adv=f"{loss_adv.item():.3f}",
                    unsup=f"{float(loss_unsup.item()):.3f}",
                    disc=f"{loss_disc.item():.3f}",
                )


        # ---- val (optional) ----
        val_loss = 0.0
        if (ep + 1) % save_every == 0 or (ep + 1) == epochs:
            val_loss = eval_loop_ddp(model_ddp, val_dl, crit_ce, device)

        if rank == 0:
            denom = max(1, n_batches)
            avg_total = running_total / denom
            avg_sup = running_sup / denom
            avg_adv = running_adv / denom
            avg_unsup = running_unsup / denom
            avg_disc = running_disc / denom
            print(f"[epoch {ep+1}] train_loss={avg_total:.3f} | val_loss={val_loss:.3f}")
            # CSVへ追記
            try:
                import csv as _csv
                with open(os.path.join(save_dir, "da_losses.csv"), "a", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow([ep+1, f"{avg_total:.6f}", f"{avg_sup:.6f}", f"{avg_adv:.6f}", f"{avg_unsup:.6f}", f"{avg_disc:.6f}", f"{val_loss:.6f}"])
            except Exception:
                pass

            if (ep + 1) % save_every == 0 or (ep + 1) == epochs:
                save_path = os.path.join(save_dir, f"model_ep{ep+1}.pt")
                torch.save(model_ddp.module.state_dict(), save_path)
                print(f"✅ saved -> {save_path}")

                if use_dc and disc_ddp is not None:
                    disc_path = os.path.join(save_dir, f"disc_ep{ep+1}.pt")
                    torch.save(disc_ddp.module.state_dict(), disc_path)
                    print(f"✅ saved -> {disc_path}")

        dist.barrier()

    dist.destroy_process_group()
    return model_ddp
