# run/make_synth_piano.py
import os, math, random
import numpy as np
import pretty_midi
import torch
import torchaudio

SR = 22050
CHUNK_SEC = 2.048
N_SAMPLES = 50
OUT_WAV = "data/wavs"
OUT_MID = "data/midis"
os.makedirs(OUT_WAV, exist_ok=True)
os.makedirs(OUT_MID, exist_ok=True)

def note_freq(p):
    return 440.0 * (2.0 ** ((p - 69) / 12.0))

def synth_sine_midi(notes, sr=SR, length_sec=CHUNK_SEC):
    """とても簡易なサイン波合成（ADSRもどき）"""
    T = int(length_sec * sr)
    y = np.zeros(T, dtype=np.float32)
    for (on, off, p) in notes:
        f = note_freq(p)
        on_i = int(on * sr)
        off_i = int(off * sr)
        t = np.arange(off_i - on_i, dtype=np.float32) / sr
        # ぷちノイズ防止のADSR（攻0.01s 衰0.02s）
        attack = min(0.01, off - on) 
        decay  = min(0.02, max(0.0, off - on - attack))
        sus    = max(0.0, (off - on) - (attack + decay))
        env = np.ones_like(t)
        a_len = max(1, int(attack * sr))
        d_len = max(1, int(decay  * sr))
        s_len = max(0, int(sus    * sr))
        if a_len > 0:
            env[:a_len] = np.linspace(0.0, 1.0, a_len, endpoint=False)
        if d_len > 0 and a_len + d_len <= len(env):
            env[a_len:a_len+d_len] = np.linspace(1.0, 0.7, d_len, endpoint=False)
        if a_len + d_len + s_len < len(env):
            env[a_len + d_len + s_len:] = np.linspace(0.7, 0.0, len(env)-(a_len+d_len+s_len), endpoint=False)
        wave = 0.3 * env * np.sin(2 * math.pi * f * t)
        y[on_i:off_i] += wave.astype(np.float32)
    # クリップ
    y = np.clip(y, -1.0, 1.0)
    return y

def make_one(idx):
    # ランダム音符（2.048s 内に収まる）
    n_notes = random.randint(6, 12)
    notes = []
    t = 0.0
    for _ in range(n_notes):
        dur = random.uniform(0.12, 0.45)
        gap = random.uniform(0.02, 0.18)
        on  = min(t, CHUNK_SEC - 0.12)
        off = min(on + dur, CHUNK_SEC)
        pitch = random.randint(60, 72)  # C4-D5 あたり
        notes.append((on, off, pitch))
        t = min(off + gap, CHUNK_SEC - 0.06)
    # MIDI
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    for on, off, p in notes:
        inst.notes.append(pretty_midi.Note(velocity=85, pitch=p, start=on, end=off))
    pm.instruments.append(inst)

    # WAV（サイン波簡易合成）
    y = synth_sine_midi(notes)
    torchaudio.save(
        os.path.join(OUT_WAV, f"pno_{idx:04d}.wav"),
        torch.from_numpy(y).unsqueeze(0),
        SR
    )
    pm.write(os.path.join(OUT_MID, f"pno_{idx:04d}.mid"))

def main():
    random.seed(42)
    for i in range(N_SAMPLES):
        make_one(i)
    print(f"Generated {N_SAMPLES} pairs under {OUT_WAV} and {OUT_MID}")

if __name__ == "__main__":
    main()
