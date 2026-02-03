#!/usr/bin/env bash

set -euo pipefail

PYTHON=${PYTHON:-python}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/train/train_l2_sft.py"

# export WANDB_DISABLED=true
# export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}

export WANDB_MODE=disabled

# Outputs
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/train/sft/eval_only"}

# Optional: resume from a checkpoint directory (e.g. .../checkpoint-1234)
RESUME_FROM=${RESUME_FROM:-""}

# Required inputs (not included in this repo). Override via env vars.
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-""}
TRAIN_JSON=${TRAIN_JSON:-"${REPO_ROOT}/data/train.jsonl"}
VAL_JSON=${VAL_JSON:-"${REPO_ROOT}/data/val.jsonl"}
FRAME_ROOT=${FRAME_ROOT:-"${REPO_ROOT}/data/frames"}
ANNOTATION_JSON=${ANNOTATION_JSON:-"${REPO_ROOT}/data/annotations/all_annotations.json"}
PRIORITY_SCORES_JSON=${PRIORITY_SCORES_JSON:-"${REPO_ROOT}/data/annotations/priority_score.json"}

NUM_PROCESSES=${NUM_PROCESSES:-8}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] MODEL_NAME_OR_PATH is not set. Please point it to your base model."
  exit 1
fi

# When resuming, some downstream utilities (e.g. merge_global_metrics) expect
# `<run_dir>/eval_pred` to exist. Make it robust by creating it if missing.
if [[ -n "${RESUME_FROM}" ]]; then
  mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/eval_pred"

  # If OUTPUT_DIR contains per-run subfolders (e.g. model*/),
  # ensure each has an eval_pred/ as well.
  shopt -s nullglob
  for d in "${OUTPUT_DIR}"/model*/ ; do
    mkdir -p "${d%/}/eval_pred"
  done
  shopt -u nullglob

  # Also ensure the run directory that contains the checkpoint has eval_pred/.
  RUN_DIR="$(dirname "${RESUME_FROM}")"
  mkdir -p "${RUN_DIR}/eval_pred"
fi

"${ACCELERATE_BIN}" launch --num_processes "${NUM_PROCESSES}" --mixed_precision "${MIXED_PRECISION}" "${TRAIN_SCRIPT}" \
  --model_name "${MODEL_NAME_OR_PATH}" \
  --train_json "${TRAIN_JSON}" \
  --val_json "${VAL_JSON}" \
  --frame_root "${FRAME_ROOT}" \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 2 \
  --learning_rate 5e-5 --num_train_epochs 5 \
  --lora_rank 32 --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --window_size 5 --window_stride 3 \
  --load_in_4bit \
  --annotation "${ANNOTATION_JSON}" \
  --priority_scores "${PRIORITY_SCORES_JSON}" \
  --record_prompts \
  --eval_every_n_epochs 2 \
  --min_reason_tokens 0 \
  --scores \
  --scores_max_tokens 50 \
  --seed 42 \
  --lazy_loading \
  --dataloader_num_workers 4 \
  --disable_find_unused_parameters \
  --max_image_long_edge 384 \
  --save_total_limit 5 \
  --eval_generation_batch_size 16 \
  --predict_steps 5 \
  --silence \
  --bind_trigger_task \
  --bind_tt_weight 0.3 \
  --bind_task_step \
  --bind_ts_weight 0.5 \
  --bind_trigger_disc_weight 0 \
  ${RESUME_FROM:+--resume_from_checkpoint "${RESUME_FROM}"}