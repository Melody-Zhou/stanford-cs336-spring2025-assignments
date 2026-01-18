#!/usr/bin/env bash
set -e

MODELS=("small" "medium" "large" "xl" "2.7b")
CONTEXT_LENGTH=128
OUT_DIR="runs"

mkdir -p ${OUT_DIR}

echo "== Nsight Systems profiling start =="

for MODEL in "${MODELS[@]}"; do
  OUT_NAME="nsys_a_${MODEL}_s${CONTEXT_LENGTH}"

  echo ""
  echo ">>> Profiling model=${MODEL}, context_length=${CONTEXT_LENGTH}"
  echo ">>> Output: ${OUT_DIR}/${OUT_NAME}.nsys-rep"

  uv run nsys profile \
    -o ${OUT_DIR}/${OUT_NAME} \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    python cs336_systems/benchmark.py \
      --model-size ${MODEL} \
      --context-length ${CONTEXT_LENGTH} \
      --nvtx

done

echo ""
echo "== All profiling finished =="
