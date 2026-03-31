#!/usr/bin/env bash
set -euo pipefail

# 待機対象PID（全て終了するまで待つ）
WAIT_PIDS=(201534 201535 201536 201537)

echo "waiting target PIDs: ${WAIT_PIDS[*]}"
while :; do
  alive=0
  for p in "${WAIT_PIDS[@]}"; do
    if kill -0 "$p" 2>/dev/null; then
      alive=1
      break
    fi
  done
  [[ "$alive" -eq 0 ]] && break
  sleep 30
done
echo "all target PIDs finished. starting queued jobs..."

python -m torch.distributed.run --nproc_per_node=4 --master_port=29511 run_mt3/train_maestro_ddp_SSL.py \
  --root dataset/maestro-v3.0.0 \
  --label_frac 0.1 \
  --epochs 7000 \
  --bs 4 \
  --pseudo_start_epoch 2200 \
  --pseudo_debug_start_epoch 2200 \
  --pseudo_threshold -1.5 \
  --pseudo_note_target_only \
  --pseudo_note_onset_only \
  --pseudo_note_conf_mode prob_and_mask \
  --pseudo_note_prob_threshold -1.0 \
  --pseudo_note_mask_score_metric abs_mask_delta \
  --pseudo_note_mask_threshold 1.0 \
  --pseudo_note_mask_width_ratio 0.2 \
  --pseudo_note_mask_fill zero \
  --pseudo_repair_order \
  --pseudo_debug_n 100000000 \
  --save-dir checkpoints_maestro_SSL/run_frac10pct_prob_and_mask_chunk_note_repair_debugall \
  --pseudo_debug_dir outputs/pseudo_debug/run_frac10pct_prob_and_mask_chunk_note_repair_debugall

python -m torch.distributed.run --nproc_per_node=4 --master_port=29512 run_mt3/train_maestro_ddp_SSL.py \
  --root dataset/maestro-v3.0.0 \
  --label_frac 0.1 \
  --epochs 7000 \
  --bs 4 \
  --pseudo_start_epoch 1 \
  --pseudo_debug_start_epoch 2200 \
  --pseudo_threshold -1.5 \
  --pseudo_note_target_only \
  --pseudo_note_onset_only \
  --pseudo_note_conf_mode prob \
  --pseudo_note_prob_threshold -1.0 \
  --pseudo_repair_order \
  --pseudo_debug_n 100000000 \
  --save-dir checkpoints_maestro_SSL/run_frac10pct_chunk_note_probonly_repair_debugall \
  --pseudo_debug_dir outputs/pseudo_debug/run_frac10pct_chunk_note_probonly_repair_debugall
