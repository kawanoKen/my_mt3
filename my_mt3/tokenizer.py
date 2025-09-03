from dataclasses import dataclass
import math
TIME_STEP_MS = 10
CHUNK_SEC = 2.048
NUM_TIME = int(round(CHUNK_SEC * 1000 / TIME_STEP_MS)) + 1   # ≈205

PROGRAMS = ["piano","guitar","bass","drums","vocal"]  # MVP
PITCHES = list(range(128))

@dataclass
class Vocab:
    # 例: [PAD, EOS, END_TIE] + PROGRAM_x + NOTE_ON_p + NOTE_OFF_p + TIME_t
    pad: int; eos: int; end_tie: int
    program: dict; note_on: dict; note_off: dict; time: dict
    itos: list

def build_vocab():
    itos = []
    def add(tok): itos.append(tok); return len(itos)-1
    pad = add("<pad>"); eos = add("<eos>"); end_tie = add("<end_tie>")
    program = {f"PRG_{n}": add(f"PRG_{n}") for n,_ in enumerate(PROGRAMS)}
    note_on = {p: add(f"NON_{p}") for p in PITCHES}
    note_off= {p: add(f"NOF_{p}") for p in PITCHES}
    time    = {t: add(f"TIM_{t}") for t in range(NUM_TIME)}
    return Vocab(pad,eos,end_tie,program,note_on,note_off,time,itos)

VOCAB = build_vocab()

def encode_events(note_events, program_id, ties):
    """
    note_events: [(on_ms, off_ms, pitch), ...]  ※チャンク内絶対msで量子化済み
    program_id: int (PRG)
    ties: [(pitch, remaining_ms), ...] チャンク先頭で鳴り続けている音
    """
    ids = [list(VOCAB.program.values())[program_id]]
    if ties:  # Tie宣言節
        ids += [VOCAB.end_tie]
        # （MVPではTie詳細を持たせず宣言のみ。改良版で詳細符号化しても良い）
    # 時系列をTIM→NON/NOFの順に並べる（同一時刻は[NON*,NOF*]の順で安定）
    timeline = {}
    for on,off,p in note_events:
        timeline.setdefault(on, []).append(("on",p))
        timeline.setdefault(off,[]).append(("off",p))
    for t in sorted(timeline.keys()):
        t_clamped = min(t, NUM_TIME - 1)    # ★保険
        ids.append(VOCAB.time[t])
        for kind,p in sorted(timeline[t], key=lambda x: 0 if x[0]=="on" else 1):
            ids.append(VOCAB.note_on[p] if kind=="on" else VOCAB.note_off[p])
    ids.append(VOCAB.eos)
    return ids

def decode_events(token_ids):
    # 逆変換（MVP: プログラム検出/時間復元/ノート組み立て）
    pass