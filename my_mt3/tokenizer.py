from dataclasses import dataclass
import math

# --------------------
# 基本設定
# --------------------
TIME_STEP_MS = 10
INSTRUMENT_TYPES = ["piano", "guitar", "bass", "drums", "vocal"]
PITCHES = list(range(128))

INPUT_FRAMES = 256

# --------------------
# Vocab 定義
# --------------------
@dataclass
class Vocab:
    pad: int
    eos: int
    end_tie: int | None  
    instrument_type: dict
    note_on: dict
    note_off: dict | None  
    time: dict
    itos: list


def build_vocab(
    *,
    input_frames: int,
    sr: int = 16000,
    hop: int = 256,
    n_fft: int = 2048,
    time_step_ms: int = TIME_STEP_MS,
    instrument_type: str = "piano",  # "piano" | "drum"
    include_note_off: bool = False,   # True で NOF_* を語彙に含める
):
    """
    input_frames に基づいて TIME_x の語彙数を自動決定する
    - instrument_type="piano" の場合: tie トークンを含む
    - instrument_type="drum"  の場合: tie（および note off）は含めない
    """
    # --- window 秒数 ---
    need_samples = (input_frames - 1) * hop + n_fft
    window_sec = need_samples / sr

    # --- TIME 語彙数 ---
    num_time = int(math.ceil(window_sec * 1000 / time_step_ms)) + 1

    itos = []
    def add(tok):
        itos.append(tok)
        return len(itos) - 1

    pad = add("<pad>")
    eos = add("<eos>")
    end_tie: int | None
    if instrument_type == "piano":
        end_tie = add("<end_tie>")
    else:
        # drum 用は tie を使わない
        end_tie = None

    instrument_type = {f"PRG_{n}": add(f"PRG_{n}") for n, _ in enumerate(INSTRUMENT_TYPES)}
    note_on = {p: add(f"NON_{p}") for p in PITCHES}
    note_off = {p: add(f"NOF_{p}") for p in PITCHES} if include_note_off else None
    time = {t: add(f"TIM_{t}") for t in range(num_time)}

    return Vocab(
        pad=pad,
        eos=eos,
        end_tie=end_tie,
        instrument_type=instrument_type,
        note_on=note_on,
        note_off=note_off,
        time=time,
        itos=itos,
    )


def encode_events(note_events, program_id, ties, *, frame_max_token: int, vocab: Vocab):
    """
    note_events: [(on_t, off_t, pitch), ...]
    program_id: int (PRG id)
    ties: [(pitch, remaining_t), ...]
      tie section: <end_tie> NON_p1 NON_p2 ... の形で、前チャンクから
      持ち越されたピッチを明示する。timeline では tie pitch の onset を省略し
      offset(NOF) のみ出力する。
    frame_max_token: int
    """
    # ---- program token ----
    prg_key = f"PRG_{int(program_id)}"
    if prg_key not in vocab.instrument_type:
        raise KeyError(f"{prg_key} not in vocab.instrument_type (keys={list(vocab.instrument_type.keys())[:5]}...)")
    ids = [vocab.instrument_type[prg_key]]

    # ---- tie section: <end_tie> NON_p1 NON_p2 ... ----
    tie_pitches: set[int] = set()
    if ties and getattr(vocab, "end_tie", None) is not None:
        ids.append(vocab.end_tie)
        for p, _ in sorted(ties, key=lambda x: x[0]):
            ids.append(vocab.note_on[int(p)])
            tie_pitches.add(int(p))

    # ---- timeline: TIME index -> [("on", pitch), ...] ----
    timeline: dict[int, list[tuple[str,int]]] = {}
    has_off = getattr(vocab, "note_off", None) is not None
    for on_t, off_t, p in note_events:
        on_t  = int(on_t)
        off_t = int(off_t)
        p     = int(p)
        on_t  = max(0, min(on_t,  frame_max_token))
        off_t = max(0, min(off_t, frame_max_token))
        if p in tie_pitches and on_t == 0:
            # tie pitch: onset は tie section で既出。offset のみ timeline に残す
            if has_off:
                timeline.setdefault(off_t, []).append(("off", p))
        else:
            timeline.setdefault(on_t,  []).append(("on", p))
            if has_off:
                timeline.setdefault(off_t, []).append(("off", p))

    # ---- 時系列順に符号化（同時刻は on のみ）----
    for t in sorted(timeline.keys()):
        ids.append(vocab.time[t])
        if has_off:
            # on を先、off を後に並べる
            for kind, p in sorted(timeline[t], key=lambda x: 0 if x[0] == "on" else 1):
                if kind == "on":
                    ids.append(vocab.note_on[p])
                else:
                    ids.append(vocab.note_off[p])  # type: ignore[index]
        else:
            for _, p in timeline[t]:
                ids.append(vocab.note_on[p])

    ids.append(vocab.eos)
    return ids


def decode_events(token_ids):
    # 逆変換（MVP: プログラム検出/時間復元/ノート組み立て）
    pass

# ---- デフォルト語彙 ----
# ピアノ用（tieあり、Note Offは用途に応じて切替可。推論/学習の互換性のため既定は Note Off なし）
VOCAB_PIANO = build_vocab(input_frames=INPUT_FRAMES, instrument_type="piano", include_note_off=True)
VOCAB_DRUM  = build_vocab(input_frames=INPUT_FRAMES, instrument_type="drum",  include_note_off=False)
VOCAB = VOCAB_PIANO  # 後方互換のためエイリアス
DEFAULT_VOCAB = VOCAB