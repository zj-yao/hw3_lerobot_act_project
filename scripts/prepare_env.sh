#!/usr/bin/env bash
set -euo pipefail

export HW3_PROJECT="/data/Vscode_project/Deeplearning/Homework_3"
export HW3_STORAGE="/data/yzj/calvin_lerobot_act_hw2"
export CALVIN_RAW="${HW3_STORAGE}/raw_datasets/calvin"
export LEROBOT_DATA_HOME="${HW3_STORAGE}/processed_datasets/lerobot"
export HF_LEROBOT_HOME="${LEROBOT_DATA_HOME}"
export LEROBOT_REPO="${HW3_PROJECT}/repos/lerobot_v033"
export CHECKPOINT_DIR="${HW3_STORAGE}/checkpoints"
export LOG_DIR="${HW3_STORAGE}/logs"
export EVAL_DIR="${HW3_STORAGE}/eval_outputs"

mkdir -p \
  "${HW3_PROJECT}/configs" \
  "${HW3_PROJECT}/scripts" \
  "${HW3_PROJECT}/reports" \
  "${HW3_PROJECT}/repos" \
  "${HW3_STORAGE}/raw_datasets" \
  "${HW3_STORAGE}/processed_datasets" \
  "${HW3_STORAGE}/checkpoints" \
  "${HW3_STORAGE}/logs" \
  "${HW3_STORAGE}/eval_outputs"

if [[ "${1:-}" == "--check" ]]; then
  echo "HW3_PROJECT=${HW3_PROJECT}"
  echo "HW3_STORAGE=${HW3_STORAGE}"
  echo "CALVIN_RAW=${CALVIN_RAW}"
  echo "LEROBOT_DATA_HOME=${LEROBOT_DATA_HOME}"
  echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
  echo "LEROBOT_REPO=${LEROBOT_REPO}"
  echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "EVAL_DIR=${EVAL_DIR}"
  echo
  conda run -n dl_hw2 python -c 'import sys, torch
print("python:", sys.executable)
print("python_version:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in conda env dl_hw2")
idx = torch.cuda.current_device()
props = torch.cuda.get_device_properties(idx)
print("gpu_name:", torch.cuda.get_device_name(idx))
print("gpu_memory_gib:", round(props.total_memory / 1024**3, 2))'
fi
