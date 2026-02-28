#!/usr/bin/env bash
set -euo pipefail

# ==========================
# Logging
# ==========================
LOG_DIR="logs/expert_iteration"
mkdir -p "${LOG_DIR}"

# ==========================
# Common defaults
# ==========================
N_EI_STEPS=5
EVAL_MAX_EXAMPLES=500
SAMPLING_MAX_TOKENS=256
SAMPLING_MIN_TOKENS=4
SEED=0

TRAIN_DEVICE="cuda:0"
VLLM_DEVICE="cuda:1"

# ==========================
# Sweep list
# Format:
#   "<name>  <args...>"
# ==========================
declare -a RUNS=(
  # ---- Minimal required coverage (2 rollout+epoch combos) ----
  "G2_E1_D512   --G 2  --epochs 1 --D_i 512"
  "G2_E1_D1024  --G 2  --epochs 1 --D_i 1024"
  "G2_E1_D2048  --G 2  --epochs 1 --D_i 2048"

  "G8_E3_D512   --G 8  --epochs 3 --D_i 512"
  "G8_E3_D1024  --G 8  --epochs 3 --D_i 1024"
  "G8_E3_D2048  --G 8  --epochs 3 --D_i 2048"

  # ---- Optional extra combos (uncomment if you have time/compute) ----
  # "G4_E1_D1024  --G 4  --epochs 1 --D_i 1024"
  # "G8_E1_D1024  --G 8  --epochs 1 --D_i 1024"
  # "G2_E3_D1024  --G 2  --epochs 3 --D_i 1024"
)

echo "=== Starting Expert Iteration sweep at $(date) ==="
echo "Logs will be saved to: ${LOG_DIR}"
echo "n_ei_steps=${N_EI_STEPS}, eval_max_examples=${EVAL_MAX_EXAMPLES}, sampling_max_tokens=${SAMPLING_MAX_TOKENS}"
echo

for item in "${RUNS[@]}"; do
  name=$(echo "$item" | awk '{print $1}')
  args=$(echo "$item" | cut -d' ' -f2-)

  ts=$(date +"%Y%m%d_%H%M%S")
  log_file="${LOG_DIR}/${ts}_${name}.log"

  echo "=== Run: ${name} @ $(date) ==="
  echo "Command: uv run python scripts/expert_iteration_experiment.py ${args} ..."
  echo "Log: ${log_file}"
  echo

  set +e
  {
    echo "===== BEGIN ${name} $(date) ====="
    echo "CMD: uv run python scripts/expert_iteration_experiment.py ${args} \\"
    echo "  --n_ei_steps ${N_EI_STEPS} \\"
    echo "  --eval_max_examples ${EVAL_MAX_EXAMPLES} \\"
    echo "  --sampling_max_tokens ${SAMPLING_MAX_TOKENS} \\"
    echo "  --sampling_min_tokens ${SAMPLING_MIN_TOKENS} \\"
    echo "  --seed ${SEED} \\"
    echo "  --train_device ${TRAIN_DEVICE} \\"
    echo "  --vllm_device ${VLLM_DEVICE} \\"
    echo

    uv run python scripts/expert_iteration_experiment.py \
      ${args} \
      --n_ei_steps "${N_EI_STEPS}" \
      --eval_max_examples "${EVAL_MAX_EXAMPLES}" \
      --sampling_max_tokens "${SAMPLING_MAX_TOKENS}" \
      --sampling_min_tokens "${SAMPLING_MIN_TOKENS}" \
      --seed "${SEED}" \
      --train_device "${TRAIN_DEVICE}" \
      --vllm_device "${VLLM_DEVICE}" \

    exit_code=$?
    echo
    echo "EXIT_CODE: ${exit_code}"
    echo "===== END ${name} $(date) ====="
    exit ${exit_code}
  } 2>&1 | tee "${log_file}"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ "${exit_code}" -ne 0 ]]; then
    echo
    echo "!!! Run ${name} failed with exit code ${exit_code}. Stopping sweep."
    echo "See log: ${log_file}"
    exit "${exit_code}"
  fi

  echo
  echo "=== Run ${name} finished successfully @ $(date) ==="
  echo
done

echo "=== All Expert Iteration runs completed at $(date) ==="