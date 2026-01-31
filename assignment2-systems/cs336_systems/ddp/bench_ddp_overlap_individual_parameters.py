import argparse
import math
import time
from multiprocessing import Manager
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext

from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy_from_logits

from cs336_systems.ddp.bench_naive_ddp import setup, cleanup, sync_if_cuda, build_xl_model, make_fake_batch
from cs336_systems.ddp.ddp_overlap_individual_parameters import DDPIndividualParameters
from cs336_systems.utils import MinimalDDPFlatBenchRow, MinimalDDPFlatBenchmarkReporter


def worker(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: str,
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

        assert world_size == 2, "Standardized to 2 GPUs."
        assert backend == "nccl", "Intended for NCCL + CUDA."
    
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = torch.float32

        # base model
        model = build_xl_model(device=device, dtype=dtype)

        # wrap with overlap-ddp container
        ddp_model = DDPIndividualParameters(model)

        x, y = make_fake_batch(global_batch_size, context_length, 10000, device=device)
        micro_bs = global_batch_size // world_size
        x_local = x[rank * micro_bs : (rank + 1) * micro_bs]
        y_local = y[rank * micro_bs : (rank + 1) * micro_bs]

        loss_fn = cross_entropy_from_logits
        opt = AdamW(ddp_model.parameters())  # optimizer sees same params

        def nvtx_range(msg: str):
            if not use_nvtx:
                return nullcontext()
            if profile and rank != profile_rank:
                return nullcontext()
            return torch.cuda.nvtx.range(msg)

        def run_one_step() -> Tuple[float, float]:
            opt.zero_grad(set_to_none=True)

            sync_if_cuda(device)
            t0 = time.perf_counter()

            with nvtx_range("fwd"):
                logits = ddp_model(x_local)

            with nvtx_range("loss"):
                loss = loss_fn(logits, y_local)

            with nvtx_range("bwd"):
                loss.backward()

            sync_if_cuda(device)
            c0 = time.perf_counter()

            with nvtx_range("finish_grad_sync"):
                ddp_model.finish_gradient_synchronization()

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
                # outer range per iteration
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
            s, c = run_one_step()
            step_times.append(s)
            comm_times.append(c)

        gathered_steps: List[List[float]] = [None for _ in range(world_size)]  # type: ignore
        gathered_comms: List[List[float]] = [None for _ in range(world_size)]  # type: ignore
        dist.all_gather_object(gathered_steps, step_times)
        dist.all_gather_object(gathered_comms, comm_times)

        if rank == 0:
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
                    variant="overlap",
                    model_size="xl",
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
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=20)
    p.add_argument("--backend", type=str, default="nccl", choices=["nccl"])
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=str, default="29540")
    p.add_argument("--profile", action="store_true", help="Enable cuda profiler start/stop for Nsight Systems.")
    p.add_argument("--profile-rank", type=int, default=0, help="Which rank triggers torch.cuda.profiler.start/stop.")
    p.add_argument("--profile-steps", type=int, default=3, help="How many measured steps to capture in trace (rank==profile-rank).")
    p.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges for fwd/bwd/comm/step.")
    p.add_argument("--out-dir", type=str, default="runs/ddp_compare_xl")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    reporter = MinimalDDPFlatBenchmarkReporter(
        jsonl_path=out_dir / "metrics.jsonl",
        md_path=out_dir / "table.md",
        title="#### DDP benchmarking: per-parameter vs flat vs overlap",
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

        for r in list(out_rows):
            reporter.append(MinimalDDPFlatBenchRow(**r))
        reporter.write_markdown()

        print(f"[OK] wrote results to {out_dir/'metrics.jsonl'} and {out_dir/'table.md'}")


if __name__ == "__main__":
    main()