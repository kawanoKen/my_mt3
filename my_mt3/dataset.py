# amtx/dataset.py
import torch, pretty_midi, numpy as np
from torch.utils.data import Dataset
from .audio import load_wav_mono, wav_to_logmel, chunk_indices, ms_quantize
from .tokenizer import encode_events, PROGRAMS


def sec_to_frame(t_sec: float, sr: int, hop: int) -> int:
    return int(round(t_sec * sr / hop))

class AMTDataset(Dataset):
    def __init__(self, pairs, sr=22050, hop=256, step_ms=10):
        self.pairs = pairs  # [(wav_path, midi_path, program_id), ...]
        self.sr, self.hop, self.step_ms = sr, hop, step_ms

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        wav, midi, pid = self.pairs[i]
        y, _ = load_wav_mono(wav, sr=self.sr)
        mel_full = wav_to_logmel(y, sr=self.sr, hop=self.hop)  # [T_full, F]
        total_sec = len(y) / self.sr

        # 参照MIDIを読む
        pm = pretty_midi.PrettyMIDI(midi)
        notes = [(n.start, n.end, n.pitch)
                 for inst in pm.instruments for n in inst.notes]

        chunks = []
        # 最大フレーム index（Timeトークンの終端含む定義に対応）
        frame_max_template = int(round(2.048 * 1000 / self.step_ms))  # ≈205
        frame_max_template = max(0, frame_max_template - 1)           # 204

        for s, e in chunk_indices(total_sec):
            # ---- ここが重要：Mel をチャンク範囲にスライス ----
            fs = sec_to_frame(s, self.sr, self.hop)
            fe = sec_to_frame(e, self.sr, self.hop)
            if fe <= fs:   # 念のため
                continue
            mel = mel_full[fs:fe, :]          # [T_chunk, F] だけ渡す

            ev = []
            ties = []
            # チャンク内のノート抽出＋量子化（0..204に収める）
            frame_max = frame_max_template
            for on, off, p in notes:
                if off <= s or on >= e:
                    continue
                on_q  = max(0, min(ms_quantize(on - s, self.step_ms),  frame_max))
                off_q = max(0, min(ms_quantize(off - s, self.step_ms), frame_max))
                if on < s:  # 前チャンクから鳴ってる
                    ties.append((p, ms_quantize(min(off, e) - s, self.step_ms)))
                    on_q = 0
                ev.append((on_q, off_q, p))

            token_ids = encode_events(ev, pid, ties)
            chunks.append((mel, token_ids, (s, e)))

        return chunks
