# run/train_minimal.py

# ==== add this at the very top ====
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# ==================================

import glob, os, torch
from my_mt3.train import train_loop

def collect_pairs():
    wavs = sorted(glob.glob("data/wavs/*.wav"))
    pairs = []
    for w in wavs:
        base = os.path.splitext(os.path.basename(w))[0]
        m = os.path.join("data/midis", base + ".mid")
        if os.path.exists(m):
            # (wav_path, midi_path, program_id=0: piano)
            pairs.append((w, m, 0))
    return pairs

if __name__ == "__main__":
    pairs = collect_pairs()
    print(f"pairs: {len(pairs)}")
    model = train_loop(
        pairs,
        epochs=10,          # まず10周（必要に応じて20〜30）
        bs=16,              # VRAMに応じて 8〜32
        lr=2e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.save(model.state_dict(), "ckpt_piano.pt")
    print("saved -> ckpt_piano.pt")
