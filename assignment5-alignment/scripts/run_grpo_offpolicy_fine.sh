#!/usr/bin/env bash
set -euo pipefail

ROLLOUT_BS=256

MODEL_ID="${MODEL_ID:-data/models/Qwen2.5-Math-1.5B}"
TRAIN_PATH="${TRAIN_PATH:-data/MATH/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/MATH/validation.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-cs336_alignment/prompts/r1_zero.prompt}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"   # best lr from earlier sweep
SEED="${SEED:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

LOSS_TYPE="grpo_clip"
CLIPRANGE="${CLIPRANGE:-0.2}"

# Fine sweep settings (200 steps as suggested)
N_GRPO_STEPS="${N_GRPO_STEPS:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-1024}"

SAMPLING_TEMP="${SAMPLING_TEMP:-1.0}"
SAMPLING_MIN_TOKENS="${SAMPLING_MIN_TOKENS:-4}"
SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1024}"

MICRO_BS="${MICRO_BS:-2}"

USE_STD_NORM="${USE_STD_NORM:-1}"
LENGTH_NORM="${LENGTH_NORM:-masked_mean}"
CONSTANT_NORMALIZER="${CONSTANT_NORMALIZER:-1024}"

LOG_DIR="${LOG_DIR:-runs/grpo_offpolicy_fine}"
mkdir -p "${LOG_DIR}"

# ---------------------------
# Candidate configs from coarse sweep (edit if needed)
# Format: "epochs train_bs"
# ---------------------------
CANDIDATES=(
  "2 128"   # 4 updates/rollout
  "2 64"    # 8 updates/rollout
  "4 64"    # 16 updates/rollout (more aggressive)
)

echo "=== Off-policy GRPO FINE sweep ==="
echo "rollout_bs=${ROLLOUT_BS} steps=${N_GRPO_STEPS} eval_interval=${EVAL_INTERVAL} eval_max=${EVAL_MAX_EXAMPLES}"
echo "candidates:"
printf '  - %s\n' "${CANDIDATES[@]}"
echo "micro_bs=${MICRO_BS} (grad_acc_steps = train_bs / micro_bs)"
echo "log_dir=${LOG_DIR}"
echo

for cfg in "${CANDIDATES[@]}"; do
  E=$(echo "${cfg}" | awk '{print $1}')
  TB=$(echo "${cfg}" | awk '{print $2}')

  if (( ROLLOUT_BS % TB != 0 )); then
    echo "Skip epochs=${E} train_bs=${TB} (rollout_bs not divisible)"
    continue
  fi
  if (( TB % MICRO_BS != 0 )); then
    echo "Skip epochs=${E} train_bs=${TB} (train_bs not divisible by micro_bs=${MICRO_BS})"
    continue
  fi

  GA=$(( TB / MICRO_BS ))
  UPDATES_PER_EPOCH=$(( ROLLOUT_BS / TB ))
  TOTAL_UPDATES=$(( E * UPDATES_PER_EPOCH ))

  echo "---- Run epochs=${E} train_bs=${TB} grad_acc=${GA} updates/rollout=${TOTAL_UPDATES} ----"

  cmd=(uv run python scripts/grpo_experiment.py
    --model-id "${MODEL_ID}"
    --train-path "${TRAIN_PATH}"
    --val-path "${VAL_PATH}"
    --prompt-file "${PROMPT_FILE}"
    --log-dir "${LOG_DIR}"
    --seed "${SEED}"

    --loss-type "${LOSS_TYPE}"
    --cliprange "${CLIPRANGE}"

    --learning-rate "${LEARNING_RATE}"
    --n-grpo-steps "${N_GRPO_STEPS}"

    --rollout-batch-size "${ROLLOUT_BS}"
    --epochs-per-rollout-batch "${E}"
    --train-batch-size "${TB}"
    --gradient-accumulation-steps "${GA}"

    --sampling-temperature "${SAMPLING_TEMP}"
    --sampling-min-tokens "${SAMPLING_MIN_TOKENS}"
    --sampling-max-tokens "${SAMPLING_MAX_TOKENS}"

    --eval-interval "${EVAL_INTERVAL}"
    --eval-max-examples "${EVAL_MAX_EXAMPLES}"
  )

  if [[ "${USE_STD_NORM}" == "1" ]]; then
    cmd+=(--use-std-normalization)
  fi

  echo "${cmd[@]}"
  "${cmd[@]}"

  echo
done

echo "=== FINE sweep complete ==="