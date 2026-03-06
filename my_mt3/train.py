import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MT3Mini
from .tokenizer import Vocab, INPUT_FRAMES
from .dataset import AMTDataset
from .dataset_unlabeled import AMTRealDataset
from .discriminator import Discriminator
from .audio import ensure_wave_cache, DEFAULT_SR
import os
import itertools

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup():
    # torchrunが設定する環境変数から自動で初期化
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))


def _maybe_cache_pairs_list(pairs_list, *, sr: int, cache_dir: str | None):
    """
    pairs_list: [(wav_or_npy, midi, pid), ...]
    cache_dir が None なら何もしない
    """
    if cache_dir is None:
        return pairs_list

    out = []
    for wav_path, midi_path, pid in pairs_list:
        wav_path = str(wav_path)
        if wav_path.endswith(".npy"):
            out.append((wav_path, midi_path, pid))
        else:
            cache_path = ensure_wave_cache(wav_path, cache_dir=cache_dir, sr=sr)
            out.append((cache_path, midi_path, pid))
    return out


def _maybe_cache_pairs_map(pairs_map, *, sr: int, cache_dir: str | None):
    """
    pairs_map: {"train":[...], "validation":[...], "test":[...]}
    """
    if cache_dir is None:
        return pairs_map

    out = {}
    for split, pairs_list in pairs_map.items():
        out[split] = _maybe_cache_pairs_list(pairs_list, sr=sr, cache_dir=cache_dir)
    return out


def make_collate(vocab: Vocab):
    def _collate(batch):
        items=[]
        for chunks in batch:
            for mel, ids, _ in chunks:
                items.append((torch.tensor(mel, dtype=torch.float32),
                              torch.tensor(ids, dtype=torch.long)))
        if not items:
            raise RuntimeError("No chunks produced. Check dataset/segmentation.")

        maxL = max(len(ids) for _,ids in items)
        ys_in = torch.full((len(items), maxL), vocab.pad, dtype=torch.long)
        ys_tg = torch.full((len(items), maxL), vocab.pad, dtype=torch.long)
        mels  = []
        for i,(mel, ids) in enumerate(items):
            mels.append(mel)
            if len(ids) >= 2:
                ys_in[i,:len(ids)-1] = ids[:-1]
                ys_tg[i,:len(ids)-1] = ids[1:]
        mels = nn.utils.rnn.pad_sequence(mels, batch_first=True)
        return mels, ys_in, ys_tg
    return _collate


def collate(batch, vocab: Vocab):
    # Backward compatibility fallback using default vocab
    return make_collate(vocab)(batch)


@torch.no_grad()
def eval_loop(model, dl, crit, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for mels, y_in, y_tg in dl:
        mels, y_in, y_tg = mels.to(device), y_in.to(device), y_tg.to(device)
        logits = model(mels, y_in)
        loss = crit(logits.reshape(-1, logits.size(-1)), y_tg.reshape(-1))
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


def train_loop(
    pairs,  # ✅ {"train": [...], "validation": [...]} を想定
    epochs=5,
    bs=8,
    lr=2e-4,
    device="cuda",
    *,
    save_every=10,
    save_dir="checkpoints",
    use_cache: bool = True,
    cache_dir: str = "cache/wave_sr16000",
    sr: int = DEFAULT_SR,
    num_workers: int = 2,
    vocab: Vocab,
):
    # 1) cache（splitごとに適用）
    pairs = _maybe_cache_pairs_map(pairs, sr=sr, cache_dir=(cache_dir if use_cache else None))

    # 2) dataset / dataloader（valは決定論・全曲）
    train_ds = AMTDataset(pairs["train"], mode="train", sr=sr, vocab=vocab)
    val_ds   = AMTDataset(pairs["validation"], mode="validation", sr=sr, vocab=vocab)

    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=make_collate(vocab), num_workers=num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        collate_fn=make_collate(vocab), num_workers=num_workers, pin_memory=True
    )

    # 3) model
    model = MT3Mini(vocab_size=len(vocab.itos)).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=vocab.pad)

    print(f"train songs: {len(train_ds)} | val songs: {len(val_ds)}")

    # 4) loop
    for ep in range(epochs):
        current_epoch = ep + 1
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}", unit="batch")
        for mels, y_in, y_tg in pbar:
            mels, y_in, y_tg = mels.to(device), y_in.to(device), y_tg.to(device)

            logits = model(mels, y_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), y_tg.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += float(loss.item())
            pbar.set_postfix(train_loss=f"{loss.item():.3f}")

        train_loss = running_loss / max(1, len(train_dl))
        val_loss = eval_loop(model, val_dl, crit, device)

        print(f"[epoch {current_epoch}] train_loss={train_loss:.3f} | val_loss={val_loss:.3f}")
        if current_epoch % save_every == 0 or current_epoch == epochs:
            save_path = os.path.join(save_dir, f"model_ep{current_epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path}")
        

    return model

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


def train_loop_distributed(
    pairs,
    epochs=5,
    bs=8,
    lr=2e-4,
    *,
    save_every=10,
    save_dir="checkpoints",
    use_cache: bool = True,
    cache_dir: str = "cache/wave_sr16000",
    sr: int = DEFAULT_SR,
    num_workers: int = 2,
    vocab: Vocab,
):
    # ---- init DDP ----
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)

    rank = dist.get_rank()
    world = dist.get_world_size()

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        # cache_dir を絶対パスに正規化し、事前作成（全rankで使うため）
        if use_cache and cache_dir:
            cache_dir_abs = os.path.abspath(cache_dir)
            os.makedirs(cache_dir_abs, exist_ok=True)
            cache_dir = cache_dir_abs
    # すべてのrankがディレクトリ作成完了を待つ
    dist.barrier()

    # 1) cache（splitごとに適用）
    # rank0 で絶対化した cache_dir をブロードキャストして全rankで一致させる
    if use_cache and cache_dir:
        # 文字列長を共有 → バイト列共有（簡易実装）
        cache_dir_bytes = os.path.abspath(cache_dir).encode("utf-8") if rank == 0 else b""
        # 長さ共有
        import torch as _torch
        length_tensor = _torch.tensor([len(cache_dir_bytes)], dtype=_torch.int32, device=device)
        dist.broadcast(length_tensor, src=0)
        buf = _torch.empty((int(length_tensor.item()),), dtype=_torch.uint8, device=device)
        if rank == 0:
            buf.copy_(_torch.tensor(list(cache_dir_bytes), dtype=_torch.uint8, device=device))
        dist.broadcast(buf, src=0)
        cache_dir = bytes(buf.cpu().numpy().tolist()).decode("utf-8")
    pairs = _maybe_cache_pairs_map(pairs, sr=sr, cache_dir=(cache_dir if use_cache else None))

    # 2) dataset
    train_ds = AMTDataset(pairs["train"], mode="train", sr=sr, vocab=vocab)
    val_ds   = AMTDataset(pairs["validation"], mode="validation", sr=sr, vocab=vocab)

    # 3) sampler/loader（★drop_last=True が重要）
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
        drop_last=True,   # ★重要
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

    # 4) model
    model = MT3Mini(vocab_size=len(vocab.itos)).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=vocab.pad)

    if rank == 0:
        print(f"[DDP] world={world} | train songs={len(train_ds)} | val songs={len(val_ds)}")
        import csv as _csv
        with open(os.path.join(save_dir, "losses.csv"), "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss"])

    # 5) loop
    for ep in range(epochs):
        train_sampler.set_epoch(ep)
        model.train()
        running = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}", unit="batch", disable=(rank != 0))
        for mels, y_in, y_tg in pbar:
            mels = mels.to(device, non_blocking=True)
            y_in = y_in.to(device, non_blocking=True)
            y_tg = y_tg.to(device, non_blocking=True)

            logits = model(mels, y_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), y_tg.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            if rank == 0:
                pbar.set_postfix(train_loss=f"{loss.item():.3f}")

        # ---- val ----
        val_loss = 0.0
        if (ep + 1) % 10 == 0 or (ep + 1) == epochs:
            val_loss = eval_loop_ddp(model, val_dl, crit, device)

        if rank == 0:
            avg_train = running / max(1, len(train_dl))
            print(f"[epoch {ep+1}] train_loss={avg_train:.3f} | val_loss={val_loss:.3f}")

            import csv as _csv
            with open(os.path.join(save_dir, "losses.csv"), "a", newline="") as f:
                w = _csv.writer(f)
                w.writerow([ep+1, f"{avg_train:.6f}", f"{val_loss:.6f}"])

            if (ep + 1) % save_every == 0 or (ep + 1) == epochs:
                save_path = os.path.join(save_dir, f"model_ep{ep+1}.pt")
                torch.save(model.module.state_dict(), save_path)
                print(f"saved -> {save_path}")

        dist.barrier()

    dist.destroy_process_group()
    return model