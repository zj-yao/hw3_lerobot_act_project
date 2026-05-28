#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../configs/act_calvin_common.env"

STEPS=(025000 050000 075000 100000 125000 150000 175000 200000)
ACTA_RUN="${ACTA_RUN:-act_calvin_a_clean_20260527_a_clean_bs64_200k_070441}"
ACTABC_RUN="${ACTABC_RUN:-act_calvin_abc_clean_20260526_abc_clean_bs64_200k_022517}"

mkdir -p "${EVAL_DIR}" "${LOG_DIR}"

for MODEL in ACTA ACTABC; do
  if [[ "${MODEL}" == "ACTA" ]]; then
    RUN="${ACTA_RUN}"
  else
    RUN="${ACTABC_RUN}"
  fi

  for STEP in "${STEPS[@]}"; do
    CKPT="${CHECKPOINT_DIR}/${RUN}/checkpoints/${STEP}/pretrained_model"
    OUT="${EVAL_DIR}/offline_eval_${MODEL}_DD_${STEP}_l1_200b.json"
    python "$(dirname "$0")/offline_eval_act_loss.py" \
      --checkpoint "${CKPT}" \
      --repo-id fywang/calvin-task-D-D-lerobot \
      --dataset-root "${LEROBOT_DATA_HOME}/fywang/calvin-task-D-D-lerobot" \
      --batch-size 64 \
      --num-workers 16 \
      --num-batches 200 \
      --device cuda \
      --mode inference_l1 \
      --output-json "${OUT}"
  done
done
