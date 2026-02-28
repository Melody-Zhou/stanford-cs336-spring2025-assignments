#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs/sft_experiment"
mkdir -p "${LOG_DIR}"

EVAL_INTERVAL=200
EVAL_MAX_EXAMPLES=500

declare -a RUNS=(
  "s128   --train_samples 128  --max_steps 2000 --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
  "s256   --train_samples 256  --max_steps 2000 --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
  "s512   --train_samples 512  --max_steps 2000 --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
  "s1024  --train_samples 1024 --max_steps 2000 --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
  "sfull  --train_samples 0    --max_steps 4000 --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
)

echo "=== Starting SFT sweep at $(date) ==="
echo "Logs will be saved to: ${LOG_DIR}"
echo

for item in "${RUNS[@]}"; do
  name=$(echo "$item" | awk '{print $1}')
  args=$(echo "$item" | cut -d' ' -f2-)

  ts=$(date +"%Y%m%d_%H%M%S")
  log_file="${LOG_DIR}/${ts}_${name}.log"

  echo "=== Run: ${name} @ $(date) ==="
  echo "Command: uv run python scripts/sft_experiment.py ${args}"
  echo "Log: ${log_file}"
  echo

  set +e
  {
    echo "===== BEGIN ${name} $(date) ====="
    echo "CMD: uv run python scripts/sft_experiment.py ${args}"
    echo
    uv run python scripts/sft_experiment.py ${args}
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

echo "=== All runs completed at $(date) ==="