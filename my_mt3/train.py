import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MT3Mini
from .tokenizer import VOCAB
from .dataset import AMTDataset
from .audio import ensure_wave_cache, DEFAULT_SR


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


def collate(batch):
    items=[]
    for chunks in batch:
        for mel, ids, _ in chunks:
            items.append((torch.tensor(mel, dtype=torch.float32),
                          torch.tensor(ids, dtype=torch.long)))
    if not items:
        raise RuntimeError("No chunks produced. Check dataset/segmentation.")

    maxL = max(len(ids) for _,ids in items)
    ys_in = torch.full((len(items), maxL), VOCAB.pad, dtype=torch.long)
    ys_tg = torch.full((len(items), maxL), VOCAB.pad, dtype=torch.long)
    mels  = []
    for i,(mel, ids) in enumerate(items):
        mels.append(mel)
        if len(ids) >= 2:
            ys_in[i,:len(ids)-1] = ids[:-1]
            ys_tg[i,:len(ids)-1] = ids[1:]
    mels = nn.utils.rnn.pad_sequence(mels, batch_first=True)
    return mels, ys_in, ys_tg


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
    use_cache: bool = True,
    cache_dir: str = "cache/wave_sr16000",
    sr: int = DEFAULT_SR,
    num_workers: int = 2,
):
    # 1) cache（splitごとに適用）
    pairs = _maybe_cache_pairs_map(pairs, sr=sr, cache_dir=(cache_dir if use_cache else None))

    # 2) dataset / dataloader（valは決定論・全曲）
    train_ds = AMTDataset(pairs["train"], mode="train", sr=sr)
    val_ds   = AMTDataset(pairs["validation"], mode="validation", sr=sr)

    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate, num_workers=num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        collate_fn=collate, num_workers=num_workers, pin_memory=True
    )

    # 3) model
    model = MT3Mini(vocab_size=len(VOCAB.itos)).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=VOCAB.pad)

    print(f"train songs: {len(train_ds)} | val songs: {len(val_ds)}")

    # 4) loop
    for ep in range(epochs):
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

        print(f"[epoch {ep+1}] train_loss={train_loss:.3f} | val_loss={val_loss:.3f}")

    return model