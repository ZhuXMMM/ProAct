#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

METHOD_NAME="ours"
PRED_DIR="${PRED_DIR:-"${REPO_ROOT}/outputs/onestep_planning/qwen2.5_7b_ours"}"

METRICS_SUBPATH="${METRICS_SUBPATH:-"metrics.csv"}"
ACTION_LOG="${ACTION_LOG:-${PRED_DIR}/planning_actions_new.jsonl}"

mkdir -p "${PRED_DIR}"

extra_args=()
if [ -n "${MAX_SAMPLES:-}" ]; then
  extra_args+=(--max_samples "${MAX_SAMPLES}")
fi

python "${REPO_ROOT}/test/onestep_planning/eval_onestep_end2end.py" \
  --pred_path "${PRED_DIR}" \
  --method_name "${METHOD_NAME}" \
  --action_selector entropy \
  --human_mode "${HUMAN_MODE:-hmin}" \
  --entropy_candidate_mode "${ENTROPY_CANDIDATE_MODE:-future}" \
  --append_terminate \
  --out_dir "${PRED_DIR}" \
  --metrics_subpath "${METRICS_SUBPATH}" \
  --also_write_root_metrics \
  --action_log "${ACTION_LOG}" \
  --only_correct_task \
  "${extra_args[@]}"

