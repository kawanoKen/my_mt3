#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_mt3/run_ssl_oracle_note_target_only.sh
#   NPROC=8 ROOT=dataset/maestro-v3.0.0 bash run_mt3/run_ssl_oracle_note_target_only.sh

NPROC="${NPROC:-4}"
ROOT="${ROOT:-dataset/maestro-v3.0.0}"
LABEL_FRAC="${LABEL_FRAC:-0.10}"
EPOCHS="${EPOCHS:-200}"
BS="${BS:-4}"
INPUT_FRAMES="${INPUT_FRAMES:-256}"
PSEUDO_START_EPOCH="${PSEUDO_START_EPOCH:-50000}"
ORACLE_METRIC="${ORACLE_METRIC:-note_f}"
ORACLE_THRESHOLD="${ORACLE_THRESHOLD:-0.5}"
SAVE_DIR="${SAVE_DIR:-checkpoints_maestro_SSL/run_oracle_note_target_only_$(date +%Y%m%d_%H%M%S)}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-checkpoints_maestro_SSL/run_20260227_110302_frac10pct/model_ep1500.pt}"

python3 -m torch.distributed.run --nproc_per_node="${NPROC}" run_mt3/train_maestro_ddp_SSL.py \
  --root "${ROOT}" \
  --label_frac "${LABEL_FRAC}" \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --input_frames "${INPUT_FRAMES}" \
  --pseudo_start_epoch "${PSEUDO_START_EPOCH}" \
  --pretrained_ckpt "${PRETRAINED_CKPT}" \
  --oracle_filter \
  --oracle_metric "${ORACLE_METRIC}" \
  --oracle_threshold "${ORACLE_THRESHOLD}" \
  --oracle_note_target_only \
  --save-dir "${SAVE_DIR}"
