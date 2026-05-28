#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../configs/act_calvin_common.env"

RUN_ID="${RUN_ID:-act_calvin_abc_clean_bs64_200k}"
OUT_DIR="${CHECKPOINT_DIR}/${RUN_ID}"
LOG_FILE="${LOG_DIR}/train_${RUN_ID}.log"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"

PYTHONPATH="${LEROBOT_REPO}/src" python "${LEROBOT_REPO}/src/lerobot/scripts/train.py" \
  --dataset.repo_id=local/calvin-task-ABC-train-lerobot \
  --dataset.root="${LEROBOT_DATA_HOME}/local/calvin-task-ABC-train-lerobot" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --batch_size=64 \
  --num_workers=16 \
  --steps=200000 \
  --log_freq=10 \
  --save_freq=5000 \
  --eval_freq=0 \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --output_dir="${OUT_DIR}" 2>&1 | tee "${LOG_FILE}"
