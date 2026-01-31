import argparse
import os
import math
import time
from multiprocessing import Manager
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext

from cs336_systems.ddp.naive_ddp import broadcast_model_from_rank0, allreduce_gradients
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy_from_logits
from cs336_systems.utils import NaiveDDPBenchRow, NaiveDDPBenchmarkReporter


def setup(rank: int, world_size: int, backend: str, master_addr: str, master_port: str) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_xl_model(device: torch.device, dtype: torch.dtype) -> nn.Module:
    return TransformerLM(
        vocab_size=10000,
        context_length=128,
        d_model=768,
        num_layers=16,
        num_heads=12,
        d_ff=3200,
        rope_theta=10000.0,
        max_seq_len=128,
        eps=1e-5,
        device=device,
        dtype=dtype,
    )


def make_fake_batch(global_bs: int, ctx: int, vocab_size: int, device: torch.device):
    g = torch.Generator(device="cpu")
    g.manual_seed(123)
    tokens = torch.randint(0, vocab_size, (global_bs, ctx + 1), generator=g, dtype=torch.long)
    x = tokens[:, :-1].to(device)
    y = tokens[:, 1:].to(device)
    return x, y


def worker(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: str,
    model_size: str,
    vocab_size: int,
    global_batch_size: int,
    context_length: int,
    warmup_steps: int,
    measure_steps: int,
    profile: bool,
    profile_rank: int,
    profile_steps: int,
    use_nvtx: bool,
    out_proxy,
) -> None:
    try:
        setup(rank, world_size, backend, master_addr, master_port)

        assert world_size == 2, "This benchmark is standardized to 2 GPUs for the assignment."
        assert backend == "nccl", "This benchmark is intended for NCCL + CUDA (1 node x 2 GPU)."

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = torch.float32

        # Build model and ensure identical init
        model = build_xl_model(device=device, dtype=dtype)
        broadcast_model_from_rank0(model)
        dist.barrier()

        # Fake batch (global) then shard into micro-batch
        x, y = make_fake_batch(global_batch_size, context_length, vocab_size, device=device)
        assert global_batch_size % world_size == 0
        micro_bs = global_batch_size // world_size
        x_local = x[rank * micro_bs : (rank + 1) * micro_bs]
        y_local = y[rank * micro_bs : (rank + 1) * micro_bs]

        # Loss/optimizer
        loss_fn = cross_entropy_from_logits        
        opt = AdamW(model.parameters())

        def nvtx_range(msg: str):
            if not use_nvtx:
                return nullcontext()
            if profile and rank != profile_rank:
                return nullcontext()
            return torch.cuda.nvtx.range(msg)

        def run_one_step() -> Tuple[float, float]:
            """
            Returns: (step_ms, comm_ms)
            step_ms: forward->backward->allreduce grads->opt.step
            comm_ms: only time inside allreduce_gradients(model)
            """
            opt.zero_grad(set_to_none=True)

            sync_if_cuda(device)
            t0 = time.perf_counter()

            # --- forward/backward (placeholder) ---
            with nvtx_range("fwd"):
                logits = model(x_local)          # [micro_bs, S, V]
            with nvtx_range("loss"):
                loss = loss_fn(logits, y_local)
            with nvtx_range("bwd"):
                loss.backward()

            # --- communication timing (naive per-parameter all-reduce) ---
            sync_if_cuda(device)
            c0 = time.perf_counter()
            with nvtx_range("allreduce_grads"):
                allreduce_gradients(model)
            sync_if_cuda(device)
            c1 = time.perf_counter()

            with nvtx_range("opt_step"):
                opt.step()

            sync_if_cuda(device)
            t1 = time.perf_counter()

            return (t1 - t0) * 1e3, (c1 - c0) * 1e3

        # profiling
        if profile:
            dist.barrier()
            for i in range(profile_steps):
                # add an outer per-iteration range
                if use_nvtx and rank == profile_rank:
                    torch.cuda.nvtx.range_push(f"profile_iter_{i}")

                run_one_step()
                
                if use_nvtx and rank == profile_rank:
                    torch.cuda.nvtx.range_pop()
            dist.barrier()
            return

        # warmup
        for _ in range(warmup_steps):
            run_one_step()
        dist.barrier()

        # measure
        step_times: List[float] = []
        comm_times: List[float] = []
        for _ in range(measure_steps):
            s_ms, c_ms = run_one_step()
            step_times.append(s_ms)
            comm_times.append(c_ms)

        # gather per-rank times
        gathered_steps: List[List[float]] = [None for _ in range(world_size)]  # type: ignore
        gathered_comms: List[List[float]] = [None for _ in range(world_size)]  # type: ignore
        dist.all_gather_object(gathered_steps, step_times)
        dist.all_gather_object(gathered_comms, comm_times)                

        if rank == 0:
            # use per-iter max across ranks as synchronized step/comm time
            step_max = [max(gathered_steps[r][i] for r in range(world_size)) for i in range(measure_steps)]
            comm_max = [max(gathered_comms[r][i] for r in range(world_size)) for i in range(measure_steps)]

            def mean_std(xs: List[float]) -> Tuple[float, float]:
                m = sum(xs) / len(xs)
                if len(xs) <= 1:
                    return m, 0.0
                var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
                return m, math.sqrt(var)

            step_mean, step_std = mean_std(step_max)
            comm_mean, comm_std = mean_std(comm_max)
            comm_pct = (comm_mean / step_mean) * 100.0 if step_mean > 0 else 0.0

            out_proxy.append(
                dict(
                    model_size=model_size,
                    backend=backend,
                    device="cuda",
                    world_size=world_size,
                    dtype="fp32",
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_bs,
                    context_length=context_length,
                    warmup_steps=warmup_steps,
                    measure_steps=measure_steps,
                    step_mean_ms=step_mean,
                    step_std_ms=step_std,
                    comm_mean_ms=comm_mean,
                    comm_std_ms=comm_std,
                    comm_pct_mean=comm_pct,
                )
            )

        dist.barrier()
    finally:
        cleanup()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", type=str, default="xl")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=20)

    p.add_argument("--backend", type=str, default="nccl", choices=["nccl"])
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=str, default="29530")

    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-rank", type=int, default=0)
    p.add_argument("--profile-steps", type=int, default=3)
    p.add_argument("--nvtx", action="store_true")

    p.add_argument("--out-dir", type=str, default="runs/naive_ddp_bench")
    args = p.parse_args()

    # standardized: 1 node x 2 GPU
    assert args.world_size == 2, "Use 2 GPUs for consistency with later problems."

    out_dir = Path(args.out_dir)
    reporter = NaiveDDPBenchmarkReporter(
        jsonl_path=out_dir / "metrics.jsonl",
        md_path=out_dir / "table.md",
        title="#### Naive DDP benchmarking (XL, 1 node x 2 GPU)",
    )

    with Manager() as manager:
        out_rows = manager.list()

        mp.spawn(
            worker,
            args=(
                args.world_size,
                args.backend,
                args.master_addr,
                args.master_port,
                args.model_size,
                args.vocab_size,
                args.global_batch_size,
                args.context_length,
                args.warmup_steps,
                args.measure_steps,
                args.profile,
                args.profile_rank,
                args.profile_steps,
                args.nvtx,                
                out_rows,
            ),
            nprocs=args.world_size,
            join=True,
        )

        rows = list(out_rows)
        for r in rows:
            reporter.append(NaiveDDPBenchRow(**r))
        reporter.write_markdown()

        print(f"[OK] wrote {len(rows)} rows to {out_dir/'metrics.jsonl'} and {out_dir/'table.md'}")


if __name__ == "__main__":
    main()