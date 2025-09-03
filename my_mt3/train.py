import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm    
from .model import MT3Mini
from .tokenizer import VOCAB
from .dataset import AMTDataset

def collate(batch):
    items=[]
    for chunks in batch:
        for mel, ids, _ in chunks:
            items.append((torch.tensor(mel, dtype=torch.float32),
                          torch.tensor(ids, dtype=torch.long)))
    if not items:
        raise RuntimeError("No chunks produced. Check dataset/segmentation.")

    maxL = max(len(ids) for _,ids in items)
    ys_in = torch.full((len(items), maxL), VOCAB.pad)
    ys_tg = torch.full((len(items), maxL), VOCAB.pad)
    mels  = []
    for i,(mel, ids) in enumerate(items):
        mels.append(mel)
        ys_in[i,:len(ids)-1] = ids[:-1]
        ys_tg[i,:len(ids)-1] = ids[1:]
    mels = nn.utils.rnn.pad_sequence(mels, batch_first=True)
    assert mels.ndim==3 and ys_in.ndim==2 and ys_tg.ndim==2, (mels.shape, ys_in.shape, ys_tg.shape)
    return mels, ys_in, ys_tg

def train_loop(pairs, epochs=5, bs=8, lr=2e-4, device="cuda"):
    ds = AMTDataset(pairs)
    dl = DataLoader(ds, batch_size=bs, shuffle=True,
                    collate_fn=collate, num_workers=2)

    model = MT3Mini(vocab_size=len(VOCAB.itos)).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit= nn.CrossEntropyLoss(ignore_index=VOCAB.pad)
    print(f"dataset size (chunks): {len(ds)}")
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        # tqdmでエポック単位の進捗バー
        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}", unit="batch")
        for mels, y_in, y_tg in pbar:
            mels, y_in, y_tg = mels.to(device), y_in.to(device), y_tg.to(device)
            logits = model(mels, y_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), y_tg.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        print(f"[epoch {ep+1}] avg_loss={running_loss/len(dl):.3f}")
    return model