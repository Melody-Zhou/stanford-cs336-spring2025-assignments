#!/usr/bin/env bash
set -euo pipefail

# ====== basic settings ======
MODEL_ID="${MODEL_ID:-data/models/Qwen2.5-Math-1.5B}"
TRAIN_PATH="${TRAIN_PATH:-data/MATH/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/MATH/validation.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-cs336_alignment/prompts/r1_zero.prompt}"
LOG_DIR="${LOG_DIR:-runs/grpo_lr_sweep}"

# ====== sweep grid ======
LRS=(3e-6 1e-5 3e-5 1e-4)

# ====== GRPO hypers ======
N_GRPO_STEPS="${N_GRPO_STEPS:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-1024}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-128}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-1}"

SAMPLING_TEMP="${SAMPLING_TEMP:-1.0}"
SAMPLING_MIN_TOKENS="${SAMPLING_MIN_TOKENS:-4}"
SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1024}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SEED="${SEED:-0}"

LOSS_TYPE="${LOSS_TYPE:-reinforce_with_baseline}"
USE_STD_NORM="${USE_STD_NORM:-1}"  # 1 => add --use-std-normalization

# Optional: for grpo_clip
CLIPRANGE="${CLIPRANGE:-0.2}"

mkdir -p "${LOG_DIR}"

echo "=== GRPO LR sweep ==="
echo "model_id=${MODEL_ID}"
echo "train=${TRAIN_PATH}"
echo "val=${VAL_PATH}"
echo "log_dir=${LOG_DIR}"
echo "loss_type=${LOSS_TYPE}"
echo "steps=${N_GRPO_STEPS} rollout_bs=${ROLLOUT_BATCH_SIZE} group=${GROUP_SIZE} train_bs=${TRAIN_BATCH_SIZE} ga=${GRAD_ACC_STEPS} epochs=${EPOCHS_PER_ROLLOUT}"
echo "eval_interval=${EVAL_INTERVAL} eval_max_examples=${EVAL_MAX_EXAMPLES}"
echo "lrs=${LRS[*]}"
echo

for lr in "${LRS[@]}"; do
  echo "---- Running lr=${lr} ----"

  cmd=(uv run python scripts/grpo_experiment.py
    --model-id "${MODEL_ID}"
    --train-path "${TRAIN_PATH}"
    --val-path "${VAL_PATH}"
    --prompt-file "${PROMPT_FILE}"
    --log-dir "${LOG_DIR}"
    --seed "${SEED}"
    --learning-rate "${lr}"
    --n-grpo-steps "${N_GRPO_STEPS}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --group-size "${GROUP_SIZE}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRAD_ACC_STEPS}"
    --epochs-per-rollout-batch "${EPOCHS_PER_ROLLOUT}"
    --sampling-temperature "${SAMPLING_TEMP}"
    --sampling-min-tokens "${SAMPLING_MIN_TOKENS}"
    --sampling-max-tokens "${SAMPLING_MAX_TOKENS}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --loss-type "${LOSS_TYPE}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-max-examples "${EVAL_MAX_EXAMPLES}"
  )

  if [[ "${USE_STD_NORM}" == "1" ]]; then
    cmd+=(--use-std-normalization)
  fi

  if [[ "${LOSS_TYPE}" == "grpo_clip" ]]; then
    cmd+=(--cliprange "${CLIPRANGE}")
  fi

  echo "${cmd[@]}"
  "${cmd[@]}"

  echo "---- Done lr=${lr} ----"
  echo
done

echo "=== Sweep complete ==="