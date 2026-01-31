import argparse
import math
import time
from multiprocessing import Manager
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy_from_logits

from cs336_systems.ddp.bench_naive_ddp import setup, cleanup, sync_if_cuda, build_xl_model, make_fake_batch
from cs336_systems.ddp.ddp_overlap_individual_parameters import DDPIndividualParameters
from cs336_systems.ddp.ddp_overlap_bucketed import DDPBucketed

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
    bucket_sizes_mb: List[float],
    out_proxy,
) -> None:
    try:
        setup(rank, world_size, backend, master_addr, master_port)

        assert world_size == 2, "Standardized to 2 GPUs for this assignment."
        assert backend == "nccl", "Intended for NCCL + CUDA."

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = torch.float32  # keep consistent with your overlap benchmark

        # Prepare fixed fake batch (same across variants for fairness)
        x, y = make_fake_batch(global_batch_size, context_length, 10000, device=device)
        micro_bs = global_batch_size // world_size
        x_local = x[rank * micro_bs : (rank + 1) * micro_bs]
        y_local = y[rank * micro_bs : (rank + 1) * micro_bs]

        loss_fn = cross_entropy_from_logits

        def run_variant(variant: str, bucket_mb: Optional[float]) -> None:
            # Fresh model per variant to avoid cross-contamination
            model = build_xl_model(device=device, dtype=dtype)

            if variant == "overlap":
                ddp_model = DDPIndividualParameters(model)
            elif variant == "bucketed":
                assert bucket_mb is not None
                ddp_model = DDPBucketed(model, bucket_size_mb=float(bucket_mb))
            else:
                raise ValueError(f"unknown variant: {variant}")

            opt = AdamW(ddp_model.parameters())

            def run_one_step() -> Tuple[float, float]:
                opt.zero_grad(set_to_none=True)

                sync_if_cuda(device)
                t0 = time.perf_counter()

                logits = ddp_model(x_local)
                loss = loss_fn(logits, y_local)
                loss.backward()

                sync_if_cuda(device)
                c0 = time.perf_counter()

                # measure "communication tail" after backward
                ddp_model.finish_gradient_synchronization()

                sync_if_cuda(device)
                c1 = time.perf_counter()

                opt.step()

                sync_if_cuda(device)
                t1 = time.perf_counter()
                return (t1 - t0) * 1e3, (c1 - c0) * 1e3
            
            # ---------- warmup ----------
            for _ in range(warmup_steps):
                run_one_step()
            dist.barrier()

            # ---------- measure ----------
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
                # distributed iteration time is governed by the slowest rank
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

                # encode bucket size into variant string to avoid changing utils.py schema
                if variant == "bucketed":
                    variant_name = f"bucketed_{int(bucket_mb)}mb"
                else:
                    variant_name = "overlap"

                out_proxy.append(
                    dict(
                        variant=variant_name,
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

        # ---- Run baseline + bucket sweeps ----
        run_variant("overlap", None)
        for b in bucket_sizes_mb:
            run_variant("bucketed", float(b))        

    finally:
        cleanup()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=20)
    p.add_argument("--bucket-sizes-mb", type=float, nargs="+", default=[1, 10, 100, 1000])

    p.add_argument("--backend", type=str, default="nccl", choices=["nccl"])
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=str, default="29560")

    p.add_argument("--out-dir", type=str, default="runs/ddp_bucketed_xl")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    reporter = MinimalDDPFlatBenchmarkReporter(
        jsonl_path=out_dir / "metrics.jsonl",
        md_path=out_dir / "table.md",
        title="#### DDP bucketed benchmarking: overlap vs bucketed (sweep bucket sizes)",
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
                args.bucket_sizes_mb,
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
    