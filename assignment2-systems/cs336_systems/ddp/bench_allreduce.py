import math
import time
import os
import argparse
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.utils import DDPCommRow, DDPCommBenchmarkReporter


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


def worker(
    rank: int,
    world_size: int,
    backend: str,
    size_bytes_list: List[int],
    warmup: int,
    iters: int,
    master_addr: str,
    master_port: str,
    out_rows_proxy,  # Manager().list()
) -> None:
    try:
        setup(rank, world_size, backend, master_addr, master_port)

        use_cuda = (backend == "nccl")
        if use_cuda:
            assert torch.cuda.is_available(), "CUDA not available but backend=nccl"
            assert world_size <= torch.cuda.device_count(), (f"world_size={world_size} > cuda_device_count={torch.cuda.device_count()}")
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        dtype = torch.float32
        elem_size = torch.tensor([], dtype=dtype).element_size()  # 4 bytes

        for size_bytes in size_bytes_list:
            numel = size_bytes // elem_size
            if numel <= 0:
                raise ValueError(f"Invalid size_bytes={size_bytes}")
            
            torch.manual_seed(1234 + rank)
            x = torch.rand((numel,), device=device, dtype=dtype)

            # warmup
            for _ in range(warmup):
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
                sync_if_cuda(device)
            
            # timed
            times_ms: List[float] = []
            for _ in range(iters):
                sync_if_cuda(device)
                t0 = time.perf_counter()
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
                sync_if_cuda(device)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1e3)
            
            gathered: List[List[float]] = [None for _ in range(world_size)]  # type: ignore
            dist.all_gather_object(gathered, times_ms)

            if rank == 0:
                # per-iter max across ranks ~= synchronized step latency
                per_iter_max = [max(gathered[r][i] for r in range(world_size)) for i in range(iters)]
                mean = sum(per_iter_max) / len(per_iter_max)
                var = sum((t -mean) ** 2 for t in per_iter_max) / max(1, (len(per_iter_max) - 1))
                std = math.sqrt(var)
                max_ms = max(per_iter_max)

                out_rows_proxy.append(
                    dict(
                        backend=backend,
                        device=("cuda" if use_cuda else "cpu"),
                        world_size=world_size,
                        op="all_reduce",
                        size_bytes=size_bytes,
                        dtype="float32",
                        warmup_steps=warmup,
                        measure_steps=iters,
                        mean_ms=float(mean),
                        std_ms=float(std),
                        max_ms=float(max_ms)
                    )
                )
        
        dist.barrier()
    finally:
        cleanup()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, choices=["gloo", "nccl"], required=True)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=str, default="29501")
    p.add_argument("--out-dir", type=str, default="runs/ddp_comm_test")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Required sizes: 1MB / 10MB / 100MB / 1GB
    size_bytes_list = [
        1 * 1024 * 1024,
        10 * 1024 * 1024,
        100 * 1024 * 1024,
        1024 * 1024 * 1024,
    ]

    out_dir = Path(args.out_dir)
    reporter = DDPCommBenchmarkReporter(
        jsonl_path=out_dir / "metrics.jsonl",
        md_path=out_dir / "table.md",
        title="#### DDP communication benchmark (op=all_reduce)"
    )

    with Manager() as manager:
        out_rows = manager.list()

        mp.spawn(
            worker,
            args=(
                args.world_size,
                args.backend,
                size_bytes_list,
                args.warmup,
                args.iters,
                args.master_addr,
                args.master_port,
                out_rows,
            ),
            nprocs=args.world_size,
            join=True,
        )

        # rank0 results were appended into out_rows
        rows: List[Dict[str, Any]] = list(out_rows)
        for r in rows:
            reporter.append(DDPCommRow(**r))
        reporter.write_markdown()

        print(f"[OK] wrote {len(rows)} rows -> {out_dir/'metrics.jsonl'} and {out_dir/'table.md'}")


if __name__ == "__main__":
    main()
