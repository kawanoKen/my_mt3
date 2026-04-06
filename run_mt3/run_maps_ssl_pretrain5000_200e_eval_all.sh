#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_mt3/run_maps_ssl_pretrain5000_200e_eval_all.sh
#   NPROC=1 BS=2 bash run_mt3/run_maps_ssl_pretrain5000_200e_eval_all.sh
#
# This script does all steps in one run:
#   1) Train SSL from pretrained model_ep5000 for 200 epochs
#   2) Save checkpoints every 10 epochs
#   3) Evaluate every saved checkpoint on MAPS validation
#   4) Build one summary CSV with mean metrics per epoch

NPROC="${NPROC:-4}"
ROOT="${ROOT:-dataset/maestro-v3.0.0}"
MAPS_CSV="${MAPS_CSV:-dataset/MAPS/MAPS_Full_scenario.csv}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-checkpoints_MAPS/run_maps_supervised_10000ep/model_ep5000.pt}"

EPOCHS="${EPOCHS:-200}"
BS="${BS:-4}"
SAVE_EVERY="${SAVE_EVERY:-10}"
VAL_EVERY="${VAL_EVERY:-10}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-8}"

PSEUDO_START_EPOCH="${PSEUDO_START_EPOCH:-1}"
PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD:--1.5}"
PSEUDO_NOTE_PROB_THRESHOLD="${PSEUDO_NOTE_PROB_THRESHOLD:--1.0}"

SAVE_DIR="${SAVE_DIR:-checkpoints_MAPS/run_maps_maestro_ssl_from_pretrain_ep5000_200e}"
PSEUDO_DEBUG_DIR="${PSEUDO_DEBUG_DIR:-outputs/pseudo_debug/run_maps_maestro_ssl_from_pretrain_ep5000_200e}"
EVAL_ROOT="${EVAL_ROOT:-outputs/maps_eval_run_maps_maestro_ssl_from_pretrain_ep5000_200e}"

echo "[1/3] Training: ${SAVE_DIR}"
python -m torch.distributed.run --nproc_per_node="${NPROC}" run_mt3/train_maestro_ddp_SSL.py \
  --pretrained_ckpt "${PRETRAINED_CKPT}" \
  --root "${ROOT}" \
  --maps_csv "${MAPS_CSV}" \
  --maps_labeled_maestro_unlabeled \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --pseudo_start_epoch "${PSEUDO_START_EPOCH}" \
  --pseudo_threshold "${PSEUDO_THRESHOLD}" \
  --pseudo_note_target_only \
  --pseudo_note_conf_mode prob \
  --pseudo_note_prob_threshold "${PSEUDO_NOTE_PROB_THRESHOLD}" \
  --pseudo_debug_n 100000000 \
  --pseudo_debug_dir "${PSEUDO_DEBUG_DIR}" \
  --save-every "${SAVE_EVERY}" \
  --val-every "${VAL_EVERY}" \
  --save-dir "${SAVE_DIR}"

echo "[2/3] Evaluating saved checkpoints"
mkdir -p "${EVAL_ROOT}"
for ep in $(seq "${SAVE_EVERY}" "${SAVE_EVERY}" "${EPOCHS}"); do
  ckpt="${SAVE_DIR}/model_ep${ep}.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "  - skip ep${ep}: checkpoint not found (${ckpt})"
    continue
  fi
  out_dir="${EVAL_ROOT}/ep$(printf "%04d" "${ep}")"
  echo "  - eval ep${ep} -> ${out_dir}"
  python run_mt3/infer_maestro.py \
    --ckpt "${ckpt}" \
    --maps_csv "${MAPS_CSV}" \
    --maps_split validation \
    --out_dir "${out_dir}" \
    --batch_size "${BATCH_SIZE_EVAL}" \
    --eval
done

echo "[3/3] Building summary CSV"
SUMMARY_CSV="${EVAL_ROOT}/summary_means.csv"
python - "${EVAL_ROOT}" "${SUMMARY_CSV}" <<'PY'
import re
import sys
from pathlib import Path

import pandas as pd

eval_root = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
rows = []

for d in sorted(eval_root.glob("ep*")):
    m = re.search(r"ep(\d+)$", d.name)
    if m is None:
        continue
    ep = int(m.group(1))
    metric_csv = d / "eval_metrics.csv"
    if not metric_csv.exists():
        continue
    df = pd.read_csv(metric_csv)
    mean_row = df.drop(columns=["stem"], errors="ignore").mean(numeric_only=True).to_dict()
    mean_row["epoch"] = ep
    rows.append(mean_row)

if not rows:
    raise SystemExit("No eval_metrics.csv found under " + str(eval_root))

out = pd.DataFrame(rows).sort_values("epoch")
out.to_csv(summary_csv, index=False)
print("saved:", summary_csv)
show_cols = [c for c in ["epoch", "note_f", "note_vel_f", "onset_f", "onset_pitch_f"] if c in out.columns]
print(out[show_cols].to_string(index=False))
PY

echo "Done."
echo "  train checkpoints: ${SAVE_DIR}"
echo "  eval outputs:      ${EVAL_ROOT}"
echo "  summary csv:       ${SUMMARY_CSV}"
