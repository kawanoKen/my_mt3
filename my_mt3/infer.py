import torch, pretty_midi
from .audio import load_wav_mono, wav_to_logmel, chunk_indices, ms_quantize
from .tokenizer import VOCAB
MAX_STEPS = 1024

def greedy_decode(model, mel, device="cuda"):
    model.eval()
    with torch.no_grad():
        mem = model.enc(torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device))
        y = torch.tensor([[list(VOCAB.program.values())[0]]], dtype=torch.long, device=device)  # 例: PRG_0から
        out=[]
        for _ in range(MAX_STEPS):
            logits = model.dec(y, mem)[:,-1,:]  # [B,V]
            nxt = logits.argmax(-1)            # [B]
            tok = nxt.item()
            out.append(tok); 
            if tok == VOCAB.eos: break
            y = torch.cat([y, nxt.unsqueeze(0)], dim=1)
    return out

def to_midi_from_tokens(token_ids, sr=22050, step_ms=10):
    # MVP: 単一プログラムと仮定し、TIM/NOTE_ON/OFFからノートを復元
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    cur_ms=0; onsets={}
    for tid in token_ids:
        tok = VOCAB.itos[tid]
        if tok.startswith("TIM_"):
            cur_ms = int(tok.split("_")[1])*step_ms
        elif tok.startswith("NON_"):
            p = int(tok.split("_")[1]); onsets[p]=cur_ms
        elif tok.startswith("NOF_"):
            p = int(tok.split("_")[1])
            if p in onsets:
                on = onsets.pop(p)
                inst.notes.append(pretty_midi.Note(velocity=80, pitch=p,
                                                   start=on/1000.0, end=cur_ms/1000.0))
        elif tok == "<eos>":
            break
    pm.instruments.append(inst)
    return pm