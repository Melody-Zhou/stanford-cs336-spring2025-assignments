#!/usr/bin/env bash
set -euo pipefail

OUTDIR="runs/ddp_compare_xl"
GBS=32
CTX=128
WARMUP=2
MEASURE=10
PROFILE_STEPS=3
PROFILE_RANK=0

mkdir -p runs

echo "[1/2] Profiling INITIAL (naive per-parameter, no overlap)"
nsys profile -o runs/nsys_initial \
  --trace=cuda,nvtx \
  --force-overwrite true \
  --cpuctxsw=none --sample=none \
  uv run python cs336_systems/ddp/bench_naive_ddp.py \
    --model-size xl \
    --global-batch-size ${GBS} --context-length ${CTX} \
    --warmup-steps ${WARMUP} --measure-steps ${MEASURE} \
    --backend nccl --world-size 2 --master-addr 127.0.0.1 --master-port 29530 \
    --profile --profile-rank ${PROFILE_RANK} --profile-steps ${PROFILE_STEPS} \
    --nvtx \
    --out-dir ${OUTDIR}

echo "[2/2] Profiling OVERLAP (individual params, async all-reduce)"
nsys profile -o runs/nsys_overlap \
  --trace=cuda,nvtx \
  --force-overwrite true \
  --cpuctxsw=none --sample=none \
  uv run python cs336_systems/ddp/bench_ddp_overlap_individual_parameters.py \
    --global-batch-size ${GBS} --context-length ${CTX} \
    --warmup-steps ${WARMUP} --measure-steps ${MEASURE} \
    --backend nccl --world-size 2 --master-addr 127.0.0.1 --master-port 29540 \
    --profile --profile-rank ${PROFILE_RANK} --profile-steps ${PROFILE_STEPS} \
    --nvtx \
    --out-dir ${OUTDIR}

echo "[OK] Done. Open runs/nsys_initial.nsys-rep and runs/nsys_overlap.nsys-rep"
