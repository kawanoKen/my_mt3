# amtx/model.py
import torch, torch.nn as nn, math

import copy

class EMATeacher:
    def __init__(self, student_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.teacher = copy.deepcopy(student_module).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student_module: torch.nn.Module):
        d = self.decay
        msd = student_module.state_dict()
        tsd = self.teacher.state_dict()
        for k in tsd.keys():
            if tsd[k].dtype.is_floating_point:
                tsd[k].mul_(d).add_(msd[k], alpha=1.0 - d)
            else:
                tsd[k].copy_(msd[k])  # int buffers etc
        self.teacher.load_state_dict(tsd, strict=True)


class PosEmb(nn.Module):
    def __init__(self, d, max_len=16384):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2)*(-math.log(10000.0)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # [B,T,D]
        return x + self.pe[:x.size(1)]

class Encoder(nn.Module):
    def __init__(self, n_mels=256, d=384, L=8, nhead=6, ff=1024):
        super().__init__()
        self.proj = nn.Linear(n_mels, d)
        self.pos = PosEmb(d)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, nhead, ff, 0.1, batch_first=True) for _ in range(L)])
    def forward(self, x):  # [B,T,F]
        h = self.pos(self.proj(x))
        for blk in self.blocks: h = blk(h)
        return h  # [B,T,D]

class Decoder(nn.Module):
    def __init__(self, vocab_size, d=384, L=8, nhead=6, ff=1024):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = PosEmb(d)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(d, nhead, ff, 0.1, batch_first=True) for _ in range(L)])
        self.lm = nn.Linear(d, vocab_size)
    def forward(self, y_in, mem):  # y_in: [B,S], mem: [B,T,D]
        tgt = self.pos(self.emb(y_in))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        h=tgt
        for blk in self.blocks: h = blk(h, mem, tgt_mask=tgt_mask)
        return self.lm(h)

class MT3Mini(nn.Module):
    def __init__(self, vocab_size, n_mels=256):
        super().__init__()
        self.enc = Encoder(n_mels=n_mels)
        self.dec = Decoder(vocab_size=vocab_size)
    def forward(self, mel, y_in):
        mem = self.enc(mel)
        logits = self.dec(y_in, mem)
        return logits
