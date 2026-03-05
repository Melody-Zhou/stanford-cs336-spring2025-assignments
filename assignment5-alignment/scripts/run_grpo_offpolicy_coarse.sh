#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Fixed by assignment
# ---------------------------
ROLLOUT_BS=256

# ---------------------------
# Paths / defaults (override via env vars)
# ---------------------------
MODEL_ID="${MODEL_ID:-data/models/Qwen2.5-Math-1.5B}"
TRAIN_PATH="${TRAIN_PATH:-data/MATH/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/MATH/validation.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-cs336_alignment/prompts/r1_zero.prompt}"

# ---------------------------
# Shared hyperparams (override via env vars)
# ---------------------------
LEARNING_RATE="${LEARNING_RATE:-1e-5}"   # set this to best lr from earlier sweep
SEED="${SEED:-0}"

# Off-policy uses GRPO-Clip
LOSS_TYPE="grpo_clip"
CLIPRANGE="${CLIPRANGE:-0.2}"

# Coarse sweep settings (< 50 steps as suggested)
N_GRPO_STEPS="${N_GRPO_STEPS:-40}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-512}"

# Sampling (can shrink for sanity)
SAMPLING_TEMP="${SAMPLING_TEMP:-1.0}"
SAMPLING_MIN_TOKENS="${SAMPLING_MIN_TOKENS:-4}"
SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1024}"

# Keep microbatch size roughly constant to keep VRAM constant:
# micro = train_bs / grad_acc_steps  (we choose micro=2)
MICRO_BS="${MICRO_BS:-2}"

# Use the best choices from prior ablations (override via env vars)
# (These flags may or may not exist in your script; keep if you added them.)
USE_STD_NORM="${USE_STD_NORM:-1}"              # 1 => add --use-std-normalization
LENGTH_NORM="${LENGTH_NORM:-masked_mean}"      # masked_mean or masked_normalize
CONSTANT_NORMALIZER="${CONSTANT_NORMALIZER:-1024}"

# Output
LOG_DIR="${LOG_DIR:-runs/grpo_offpolicy_coarse}"
mkdir -p "${LOG_DIR}"

# ---------------------------
# Sweep grid
# ---------------------------
EPOCHS_LIST=(1 2 4)
TRAIN_BS_LIST=(256 128 64)

echo "=== Off-policy GRPO COARSE sweep ==="
echo "rollout_bs=${ROLLOUT_BS} steps=${N_GRPO_STEPS} eval_interval=${EVAL_INTERVAL} eval_max=${EVAL_MAX_EXAMPLES}"
echo "grid: epochs={${EPOCHS_LIST[*]}} train_bs={${TRAIN_BS_LIST[*]}}"
echo "micro_bs=${MICRO_BS} (grad_acc_steps = train_bs / micro_bs)"
echo "log_dir=${LOG_DIR}"
echo

for E in "${EPOCHS_LIST[@]}"; do
  for TB in "${TRAIN_BS_LIST[@]}"; do
    # Require divisibility for simplicity
    if (( ROLLOUT_BS % TB != 0 )); then
      echo "Skip epochs=${E} train_bs=${TB} (rollout_bs not divisible)"
      continue
    fi
    if (( TB % MICRO_BS != 0 )); then
      echo "Skip epochs=${E} train_bs=${TB} (train_bs not divisible by micro_bs=${MICRO_BS})"
      continue
    fi

    GA=$(( TB / MICRO_BS ))   # keep microbatch size constant

    # Off-policy intensity: total updates per rollout = E * (ROLLOUT_BS/TB)
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

      # helpful tag so you can grep logs later (if your script supports it; optional)
      # --run-name "coarse_e${E}_tb${TB}_u${TOTAL_UPDATES}"
    )

    if [[ "${USE_STD_NORM}" == "1" ]]; then
      cmd+=(--use-std-normalization)
    fi

    echo "${cmd[@]}"
    "${cmd[@]}"

    echo
  done
done

echo "=== COARSE sweep complete ==="