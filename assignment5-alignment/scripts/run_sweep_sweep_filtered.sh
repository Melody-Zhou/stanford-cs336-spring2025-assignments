#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs/sft_experiment_filtered"
mkdir -p "${LOG_DIR}"

EVAL_INTERVAL=200
EVAL_MAX_EXAMPLES=500

declare -a RUNS=(
  "s128   --filter_correct --train_samples 128  --max_steps 2000 --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
  "s256   --filter_correct --train_samples 256  --max_steps 2000 --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
  "s512   --filter_correct --train_samples 512  --max_steps 2000 --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
  "s1024  --filter_correct --train_samples 1024 --max_steps 2000 --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
  "sfull  --filter_correct --train_samples 0    --max_steps 4000 --eval_interval ${EVAL_INTERVAL} --eval_max_examples ${EVAL_MAX_EXAMPLES}"
)

echo "=== Starting FILTERED SFT sweep at $(date) ==="
echo "Logs: ${LOG_DIR}"
echo

for item in "${RUNS[@]}"; do
  name=$(echo "$item" | awk '{print $1}')
  args=$(echo "$item" | cut -d' ' -f2-)

  ts=$(date +"%Y%m%d_%H%M%S")
  log_file="${LOG_DIR}/${ts}_${name}.log"

  echo "=== Run: ${name} @ $(date) ==="
  echo "CMD: uv run python scripts/sft_experiment.py ${args}"
  echo "LOG: ${log_file}"
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
    echo "!!! Run ${name} failed (exit ${exit_code}). Stop."
    echo "See: ${log_file}"
    exit "${exit_code}"
  fi

  echo "=== Run ${name} DONE @ $(date) ==="
  echo
done

echo "=== All FILTERED runs finished @ $(date) ==="
