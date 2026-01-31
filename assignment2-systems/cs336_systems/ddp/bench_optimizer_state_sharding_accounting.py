import argparse
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy_from_logits

from cs336_systems.ddp.bench_naive_ddp import setup, cleanup, sync_if_cuda, build_xl_model, make_fake_batch
from cs336_systems.ddp.ddp_overlap_bucketed import DDPBucketed

from cs336_systems.utils import OptimShardMemRow, OptimShardTimeRow, OptimShardMemReporter, OptimShardTimeReporter


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def module_param_bytes(module: torch.nn.Module) -> int:
    seen = set()
    total = 0
    for p in module.parameters():
        if p in seen:
            continue
        seen.add(p)
        total += p.numel() * p.element_size()
    return total


def module_grad_bytes(module: torch.nn.Module) -> int:
    seen = set()
    total = 0
    for p in module.parameters():
        if p in seen:
            continue
        seen.add(p)
        if p.grad is not None:
            total += tensor_bytes(p.grad)
    return total


def optimizer_state_bytes(optim) -> int:
    """
    Count the visible tensor bytes in optimizer.state.
    """
    state = getattr(optim, "state", None)
    if state is None:
        return 0
    total = 0
    for _, st in state.items():
        if isinstance(st, dict):
            for _, v in st.items():
                if torch.is_tensor(v):
                    total += tensor_bytes(v)
    return total


def max_memory_mb() -> float:
    # max_memory_allocated 是 bytes
    return float(torch.cuda.max_memory_allocated()) / (1024**2)


def worker(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: str,
    global_batch_size: int,
    context_length: int,
    bucket_size_mb: float,
    dtype_str: str,
    mode: str,
    warmup_steps: int,
    measure_steps: int,    
    out_proxy,
) -> None:
    try:
        setup(rank, world_size, backend, master_addr, master_port)
        assert world_size == 2, "Standardized to 2 GPUs for this assignment."
        assert backend == "nccl", "Intended for NCCL + CUDA."

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        if dtype_str == "fp32":
            dtype = torch.float32
        elif dtype_str == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"unsupported dtype: {dtype_str}")

        # same fake batch across variants
        x, y = make_fake_batch(global_batch_size, context_length, 10000, device=device)
        micro_bs = global_batch_size // world_size
        x_local = x[rank * micro_bs : (rank + 1) * micro_bs]
        y_local = y[rank * micro_bs : (rank + 1) * micro_bs]

        loss_fn = cross_entropy_from_logits

        def time_one_iter_ms(fn):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            sync_if_cuda(device)
            start.record()
            fn()
            end.record()
            sync_if_cuda(device)
            return float(start.elapsed_time(end))

        def run_variant(variant: str) -> Dict[str, float]:
            # ---- init model & ddp wrapper ----
            model = build_xl_model(device=device, dtype=dtype)
            ddp_model = DDPBucketed(model, bucket_size_mb=bucket_size_mb)

            # ---- build optimizer ----
            if variant == "baseline":
                opt = AdamW(ddp_model.parameters())
                local_opt_for_state = opt
            elif variant == "sharded":
                from cs336_systems.ddp.shared_optimizer import SharedOptimizer
                opt = SharedOptimizer(ddp_model.parameters(), AdamW)
                local_opt_for_state = opt._local_optimizer  # type: ignore[attr-defined]
            else:
                raise ValueError(variant)

            def one_iter():
                opt.zero_grad(set_to_none=True)
                logits = ddp_model(x_local)
                loss = loss_fn(logits, y_local)
                loss.backward()
                ddp_model.finish_gradient_synchronization()
                opt.step()
            
            if mode == "time":
                # warmup
                for _ in range(warmup_steps):
                    one_iter()

                # measure
                times = [time_one_iter_ms(one_iter) for _ in range(measure_steps)]
                return dict(
                    step_mean_ms=statistics.mean(times),
                    step_std_ms=statistics.pstdev(times),
                )            

            # ---- record after init peak ----
            sync_if_cuda(device)
            torch.cuda.reset_peak_memory_stats(device)
            sync_if_cuda(device)
            peak_after_init = max_memory_mb()

            # ---- forward/backward (before step) ----
            opt.zero_grad(set_to_none=True)

            sync_if_cuda(device)
            torch.cuda.reset_peak_memory_stats(device)

            logits = ddp_model(x_local)
            loss = loss_fn(logits, y_local)
            loss.backward()
            ddp_model.finish_gradient_synchronization()

            sync_if_cuda(device)
            peak_before_step = max_memory_mb()

            # ---- optimizer step (after step) ----
            sync_if_cuda(device)
            torch.cuda.reset_peak_memory_stats(device)

            opt.step()

            sync_if_cuda(device)
            peak_after_step = max_memory_mb()

            # ---- estimates ----
            param_mb = module_param_bytes(ddp_model) / (1024**2)
            grad_mb = module_grad_bytes(ddp_model) / (1024**2)
            optim_mb = optimizer_state_bytes(local_opt_for_state) / (1024**2)

            return dict(
                peak_after_init_mb=peak_after_init,
                peak_before_step_mb=peak_before_step,
                peak_after_step_mb=peak_after_step,
                param_mb=param_mb,
                grad_mb=grad_mb,
                optim_state_mb=optim_mb,
            )

        res_baseline = run_variant("baseline")
        dist.barrier()
        res_sharded = run_variant("sharded")
        dist.barrier()

        gathered: List[Tuple[Dict[str, float], Dict[str, float]]] = [None for _ in range(world_size)]  # type: ignore
        dist.all_gather_object(gathered, (res_baseline, res_sharded))

        if rank == 0:
            def reduce_max(key: str, which: int) -> float:
                # which: 0 baseline, 1 sharded
                vals = []
                for r in range(world_size):
                    vals.append(gathered[r][which][key])
                return max(vals)

            common = dict(
                model_size="xl",
                backend=backend,
                device="cuda",
                world_size=world_size,
                dtype=dtype_str,
                global_batch_size=global_batch_size,
                micro_batch_size=micro_bs,
                context_length=context_length,
            )

            def reduce_max(key: str, which: int) -> float:
                vals = [gathered[r][which][key] for r in range(world_size)]
                return max(vals)

            if mode == "mem":
                out_proxy.append({
                    **common,
                    "variant": "baseline",
                    "peak_after_init_mb": reduce_max("peak_after_init_mb", 0),
                    "peak_before_step_mb": reduce_max("peak_before_step_mb", 0),
                    "peak_after_step_mb": reduce_max("peak_after_step_mb", 0),
                    "param_mb": reduce_max("param_mb", 0),
                    "grad_mb": reduce_max("grad_mb", 0),
                    "optim_state_mb": reduce_max("optim_state_mb", 0),
                })

                out_proxy.append({
                    **common,
                    "variant": "sharded",
                    "peak_after_init_mb": reduce_max("peak_after_init_mb", 1),
                    "peak_before_step_mb": reduce_max("peak_before_step_mb", 1),
                    "peak_after_step_mb": reduce_max("peak_after_step_mb", 1),
                    "param_mb": reduce_max("param_mb", 1),
                    "grad_mb": reduce_max("grad_mb", 1),
                    "optim_state_mb": reduce_max("optim_state_mb", 1),
                })
            else:
                out_proxy.append({
                    **common,
                    "variant": "baseline",
                    "warmup_steps": warmup_steps,
                    "measure_steps": measure_steps,
                    "step_mean_ms": reduce_max("step_mean_ms", 0),
                    "step_std_ms": reduce_max("step_std_ms", 0),
                })

                out_proxy.append({
                    **common,
                    "variant": "sharded",
                    "warmup_steps": warmup_steps,
                    "measure_steps": measure_steps,
                    "step_mean_ms": reduce_max("step_mean_ms", 1),
                    "step_std_ms": reduce_max("step_std_ms", 1),
                })


        dist.barrier()

    finally:
        cleanup()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--bucket-size-mb", type=float, default=100.0)
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])

    p.add_argument("--mode", type=str, default="mem", choices=["mem", "time"])
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=20)

    p.add_argument("--backend", type=str, default="nccl", choices=["nccl"])
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=str, default="29570")

    p.add_argument("--out-dir", type=str, default="")
    args = p.parse_args()

    if args.out_dir.strip():
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("runs/optim_state_sharding_xl_mem" if args.mode == "mem"
                    else "runs/optim_state_sharding_xl_time")

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "mem":
        reporter = OptimShardMemReporter(
            jsonl_path=out_dir / "metrics.jsonl",
            md_path=out_dir / "table.md",
            title="#### Optimizer state sharding accounting (XL, 1 node x 2 GPU): peak memory at key timestamps",
        )
    else:
        reporter = OptimShardTimeReporter(
            jsonl_path=out_dir / "metrics.jsonl",
            md_path=out_dir / "table.md",
            title="#### Optimizer state sharding accounting (XL, 1 node x 2 GPU): iteration time",
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
                args.bucket_size_mb,
                args.dtype,
                args.mode,
                args.warmup_steps,
                args.measure_steps,
                out_rows,
            ),
            nprocs=args.world_size,
            join=True,
        )

        for r in list(out_rows):
            if args.mode == "mem":
                reporter.append(OptimShardMemRow(**r))
            else:
                reporter.append(OptimShardTimeRow(**r))
        reporter.write_markdown()

    print(f"[OK] wrote results to {out_dir/'metrics.jsonl'} and {out_dir/'table.md'}")


if __name__ == "__main__":
    main()