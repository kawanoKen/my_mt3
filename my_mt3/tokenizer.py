from dataclasses import dataclass
import math

# --------------------
# 基本設定
# --------------------
TIME_STEP_MS = 10
PROGRAMS = ["piano", "guitar", "bass", "drums", "vocal"]
PITCHES = list(range(128))

INPUT_FRAMES = 256

# --------------------
# Vocab 定義
# --------------------
@dataclass
class Vocab:
    pad: int
    eos: int
    end_tie: int
    program: dict
    note_on: dict
    # note_off: dict  # Note Off は無効化（必要なら再有効化）
    time: dict
    itos: list


def build_vocab(
    *,
    input_frames: int,
    sr: int = 16000,
    hop: int = 256,
    n_fft: int = 2048,
    time_step_ms: int = TIME_STEP_MS,
):
    """
    input_frames に基づいて TIME_x の語彙数を自動決定する
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
    end_tie = add("<end_tie>")

    program = {f"PRG_{n}": add(f"PRG_{n}") for n, _ in enumerate(PROGRAMS)}
    note_on = {p: add(f"NON_{p}") for p in PITCHES}
    # note_off = {p: add(f"NOF_{p}") for p in PITCHES}  # 無効化中（Note Off なし仕様）
    time = {t: add(f"TIM_{t}") for t in range(num_time)}

    return Vocab(
        pad=pad,
        eos=eos,
        end_tie=end_tie,
        program=program,
        note_on=note_on,
        # note_off=note_off,  # 無効化中
        time=time,
        itos=itos,
    )

VOCAB = build_vocab(input_frames=INPUT_FRAMES)

def encode_events(note_events, program_id, ties, *, frame_max_token: int):
    """
    note_events: [(on_t, off_t, pitch), ...]
      - on_t/off_t は index（10ms刻み推奨）。Note Off は無視し、Note On のみ符号化します。
    program_id: int (PRG id)
    ties: [(pitch, remaining_t), ...]  ※MVPでは宣言のみ。Note Off 無し仕様では未使用だが互換のため受け取る。
    frame_max_token: int
      - このウィンドウ内で表現可能な最大 TIME index（例: 0..819）
      - Dataset側で window_sec から計算した self.frame_max_token を渡すのが正解
    """
    # ---- program token（辞書順に依存しない）----
    prg_key = f"PRG_{int(program_id)}"
    if prg_key not in VOCAB.program:
        raise KeyError(f"{prg_key} not in VOCAB.program (keys={list(VOCAB.program.keys())[:5]}...)")
    ids = [VOCAB.program[prg_key]]

    # ---- tie（MVP: 宣言だけ）----
    if ties:
        ids.append(VOCAB.end_tie)

    # ---- timeline: TIME index -> [("on", pitch), ...] ----
    # Note Off を使わないため、"off" は登録しない
    timeline: dict[int, list[tuple[str,int]]] = {}
    for on_t, off_t, p in note_events:
        on_t  = int(on_t)
        p     = int(p)
        # clamping（負は0へ、上はframe_maxへ）
        on_t  = max(0, min(on_t,  frame_max_token))
        timeline.setdefault(on_t,  []).append(("on", p))
        # --- 旧実装（Note Off あり）の参考 ---
        # off_t = int(off_t)
        # off_t = max(0, min(off_t, frame_max_token))
        # timeline.setdefault(off_t, []).append(("off", p))

    # ---- 時系列順に符号化（同時刻は on のみ）----
    for t in sorted(timeline.keys()):
        ids.append(VOCAB.time[t])
        for kind, p in timeline[t]:
            # kind は常に "on"
                ids.append(VOCAB.note_on[p])
        # --- 旧実装（Note Off あり）の参考 ---
        # for kind, p in sorted(timeline[t], key=lambda x: 0 if x[0] == "on" else 1):
        #     if kind == "on":
        #         ids.append(VOCAB.note_on[p])
        #     else:
        #         ids.append(VOCAB.note_off[p])

    ids.append(VOCAB.eos)
    return ids


def decode_events(token_ids):
    # 逆変換（MVP: プログラム検出/時間復元/ノート組み立て）
    pass