from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd


def load_cfg(path: str) -> dict:
    txt = pathlib.Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception:
        return json.loads(txt)


def main():
    print("この簡易版では encoder probe 学習は未実装です。必要なら次で追加します。")
    cfg = load_cfg("onset_confidence/conf/default.yaml")
    in_csv = pathlib.Path(cfg["evaluate"]["input_csv"])
    if in_csv.exists():
        df = pd.read_csv(in_csv)
        out = in_csv.with_name(in_csv.stem + "_with_probe_placeholder.csv")
        df["score_encoder_surrogate"] = float("nan")
        df.to_csv(out, index=False)
        print(f"Saved placeholder: {out}")


if __name__ == "__main__":
    main()
