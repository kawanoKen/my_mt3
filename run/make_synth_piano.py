from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional, List, Tuple
import time

@dataclass
class RenderConfig:
    sr: int = 16000
    gain: float = 0.8              # クリップ回避に少し下げる
    polyphony: int = 256           # 発音数上限（重いなら下げる）
    timeout_sec: int = 600         # 1曲あたり最大時間
    retries: int = 1               # 失敗時リトライ回数
    extra_args: Tuple[str, ...] = ()  # fluidsynthに追加で渡したい引数


def _check_inputs(midi_path: Path, sf2_path: Path, out_wav: Path):
    if not midi_path.exists():
        raise FileNotFoundError(midi_path)
    if midi_path.suffix.lower() not in (".mid", ".midi"):
        raise ValueError(f"Not a MIDI file: {midi_path}")
    if not sf2_path.exists():
        raise FileNotFoundError(sf2_path)
    if sf2_path.suffix.lower() != ".sf2":
        raise ValueError(f"Not a .sf2 SoundFont: {sf2_path}")
    out_wav.parent.mkdir(parents=True, exist_ok=True)


def _which_fluidsynth() -> str:
    exe = shutil.which("fluidsynth")
    if exe is None:
        raise RuntimeError(
            "fluidsynth not found in PATH. Install fluidsynth and ensure it's callable as `fluidsynth`."
        )
    return exe


def render_midi_to_wav(
    midi_path: str | Path,
    sf2_path: str | Path,
    out_wav: str | Path,
    cfg: RenderConfig = RenderConfig(),
    *,
    overwrite: bool = False,
    quiet: bool = True,
) -> Path:
    """
    Safe MIDI->WAV rendering using fluidsynth CLI.
    Uses shell=False, temp file + atomic rename, timeout, retry.
    """
    midi_path = Path(midi_path)
    sf2_path = Path(sf2_path)
    out_wav = Path(out_wav)
    _check_inputs(midi_path, sf2_path, out_wav)

    if out_wav.exists() and not overwrite:
        return out_wav

    fluidsynth = _which_fluidsynth()

    # fluidsynth options:
    #  -ni            : no interactive shell
    #  -F <file>      : render to file
    #  -r <sr>        : sample rate
    #  -g <gain>      : gain
    #  -o synth.polyphony=<n>
    #  -T wav         : output format wav (usually implied by -F, but explicit is fine)
    #
    # Note: We pass args as a list => safe against injection.
    base_cmd = [
        fluidsynth,
        "-ni",
        "-T", "wav",
        "-r", str(cfg.sr),
        "-g", str(cfg.gain),
        "-o", f"synth.polyphony={cfg.polyphony}",
    ]
    if quiet:
        base_cmd += ["-q"]
    # user extra args (already split)
    base_cmd += list(cfg.extra_args)

    # temp file in same dir for atomic rename
    tmp_dir = out_wav.parent
    tmp_fd, tmp_path_str = tempfile.mkstemp(prefix=out_wav.stem + ".", suffix=".tmp.wav", dir=str(tmp_dir))
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)

    # Final command: fluidsynth ... -F tmp.wav sf2.mid file.mid
    cmd = base_cmd + ["-F", str(tmp_path), str(sf2_path), str(midi_path)]

    last_err: Optional[str] = None
    try:
        for attempt in range(cfg.retries + 1):
            try:
                t0 = time.time()
                p = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=cfg.timeout_sec,
                    check=False,
                )
                dt = time.time() - t0

                if p.returncode != 0:
                    last_err = (p.stderr or p.stdout or "").strip()
                    raise RuntimeError(f"fluidsynth failed (code={p.returncode}, {dt:.1f}s): {last_err[:300]}")

                # sanity check: non-empty file
                if not tmp_path.exists() or tmp_path.stat().st_size < 1024:
                    last_err = "output wav is missing or too small"
                    raise RuntimeError(f"fluidsynth produced invalid wav: {tmp_path}")

                # atomic rename
                if out_wav.exists():
                    out_wav.unlink()
                tmp_path.replace(out_wav)
                return out_wav

            except subprocess.TimeoutExpired:
                last_err = f"timeout after {cfg.timeout_sec}s"
                # cleanup partial
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                if attempt >= cfg.retries:
                    raise RuntimeError(f"fluidsynth timeout: {midi_path}") from None
                continue

            except Exception as e:
                # cleanup partial
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                if attempt >= cfg.retries:
                    raise
                continue

        raise RuntimeError(last_err or "unknown error")
    finally:
        # if still exists and not renamed
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def batch_render(
    midi_files: Iterable[str | Path],
    sf2_path: str | Path,
    out_dir: str | Path,
    cfg: RenderConfig = RenderConfig(),
    *,
    overwrite: bool = False,
    max_items: Optional[int] = None,
    rel_root: Optional[str | Path] = None,   # ここが与えられた場合、相対構造を保って出力
) -> List[Tuple[Path, Optional[Path], Optional[str]]]:
    """
    Returns list of (midi_path, out_wav_or_None, error_or_None)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir = out_dir.resolve()
    rel_root_path = Path(rel_root).resolve() if rel_root is not None else None

    results: List[Tuple[Path, Optional[Path], Optional[str]]] = []
    for i, m in enumerate(midi_files):
        if max_items is not None and i >= max_items:
            break
        m = Path(m).resolve()
        if rel_root_path is not None:
            try:
                rel = m.relative_to(rel_root_path)
                out_wav = out_dir.joinpath(rel).with_suffix(".wav")
            except ValueError:
                # 相対にできない場合はフラットに
                out_wav = out_dir / (m.stem + ".wav")
        else:
            out_wav = out_dir / (m.stem + ".wav")
        try:
            w = render_midi_to_wav(m, sf2_path, out_wav, cfg, overwrite=overwrite)
            results.append((m, w, None))
        except Exception as e:
            results.append((m, None, str(e)))
    return results


if __name__ == "__main__":
    # usage examples:
    # - directory: python make_synth_piano.py /path/to.sf2 /path/to/midis /path/to/out_wavs
    # - single:    python make_synth_piano.py /path/to.sf2 /path/to/file.mid /path/to/out.wav (or /path/to/out_dir)
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python make_synth_piano.py <soundfont.sf2> <midi_dir> <out_dir>")
        print("  python make_synth_piano.py <soundfont.sf2> <midi_file.mid> <out.wav|out_dir>")
        sys.exit(1)

    sf2 = Path(sys.argv[1])
    in_path = Path(sys.argv[2])
    out_arg = Path(sys.argv[3])

    cfg = RenderConfig(sr=16000, gain=0.8, polyphony=256, timeout_sec=600, retries=1)

    # Single-file mode
    if in_path.is_file():
        if out_arg.suffix.lower() == ".wav":
            out_wav = out_arg
        else:
            out_wav = out_arg / (in_path.stem + ".wav")
        try:
            w = render_midi_to_wav(in_path, sf2, out_wav, cfg, overwrite=False)
            print(f"done: {in_path} -> {w}")
        except Exception as e:
            print(f"[ERR] {in_path}: {e}")
        sys.exit(0)

    # Directory mode (recursive)
    midi_dir = in_path
    out_dir = out_arg
    mids = sorted(list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi")))

    res = batch_render(mids, sf2, out_dir, cfg, overwrite=False, rel_root=midi_dir)

    n_ok = sum(1 for _, w, e in res if w is not None and e is None)
    n_ng = len(res) - n_ok
    print(f"done: ok={n_ok}, ng={n_ng}")
    if n_ng:
        for m, _, e in res[:20]:
            if e:
                print(f"[ERR] {m}: {e}")
