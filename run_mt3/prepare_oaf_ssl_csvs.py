import argparse
import csv
from pathlib import Path


def _read_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_labeled(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio_path", "midi_path"])
        w.writeheader()
        for r in rows:
            w.writerow({"audio_path": r["audio_path"], "midi_path": r["midi_path"]})


def _write_unlabeled(maestro_rows, maestro_root: Path, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio_path"])
        w.writeheader()
        for r in maestro_rows:
            if r.get("split") != "train":
                continue
            audio_rel = r["audio_filename"]
            audio_abs = (maestro_root / audio_rel).resolve()
            w.writerow({"audio_path": str(audio_abs)})


def main():
    ap = argparse.ArgumentParser(
        description="Create train_ssl.py-compatible CSVs for MAPS-labeled + MAESTRO-unlabeled SSL"
    )
    ap.add_argument("--maps_dir", type=str, default="dataset/MAPS")
    ap.add_argument("--maestro_csv", type=str, default="dataset/maestro-v3.0.0/maestro-v3.0.0.csv")
    ap.add_argument("--out_root", type=str, default="onset_and_frames/runs/maps_maestro_ssl_splits")
    args = ap.parse_args()

    maps_dir = Path(args.maps_dir)
    maestro_csv = Path(args.maestro_csv)
    out_root = Path(args.out_root)

    if not maps_dir.exists():
        raise FileNotFoundError(f"maps_dir not found: {maps_dir}")
    if not maestro_csv.exists():
        raise FileNotFoundError(f"maestro_csv not found: {maestro_csv}")

    maestro_root = maestro_csv.parent
    maestro_rows = _read_rows(maestro_csv)

    scenario_csvs = sorted(maps_dir.glob("MAPS_*_scenario.csv"))
    if not scenario_csvs:
        raise FileNotFoundError(f"No MAPS_*_scenario.csv found in: {maps_dir}")

    for sc_csv in scenario_csvs:
        rows = _read_rows(sc_csv)
        train_rows = [r for r in rows if r.get("split") == "train"]
        valid_rows = [r for r in rows if r.get("split") == "validation"]
        scenario_name = sc_csv.stem.replace("_scenario", "")
        out_dir = out_root / scenario_name

        _write_labeled(train_rows, out_dir / "labeled_train.csv")
        _write_labeled(valid_rows, out_dir / "valid.csv")
        _write_unlabeled(maestro_rows, maestro_root, out_dir / "unlabeled_train.csv")

        print(
            f"[{scenario_name}] "
            f"labeled_train={len(train_rows)} valid={len(valid_rows)} "
            f"unlabeled_train={sum(1 for r in maestro_rows if r.get('split') == 'train')} -> {out_dir}"
        )


if __name__ == "__main__":
    main()
