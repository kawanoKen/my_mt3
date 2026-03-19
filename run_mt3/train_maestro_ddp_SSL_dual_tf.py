import runpy
import sys
from pathlib import Path


def _has_flag(flag: str) -> bool:
    for a in sys.argv[1:]:
        if a == flag or a.startswith(flag + "="):
            return True
    return False


def _append_default(flag: str, value: str) -> None:
    if _has_flag(flag):
        return
    sys.argv.extend([flag, value])


if __name__ == "__main__":
    # 既存SSLスクリプトをそのまま使い、デフォルトで「通常TF + timewise onset TF」を有効化する。
    _append_default("--timewise_onset_tf_weight", "1.0")
    _append_default("--timewise_onset_tf_max_groups", "0")
    _append_default("--timewise_onset_tf_min_onsets", "1")

    target = Path(__file__).with_name("train_maestro_ddp_SSL.py")
    runpy.run_path(str(target), run_name="__main__")
