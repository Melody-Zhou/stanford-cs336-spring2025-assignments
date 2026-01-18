import torch
import timeit
import argparse
import statistics as stats
from contextlib import nullcontext
from cs336_basics.transformer_lm import TransformerLM
from cs336_systems.utils import BenchmarkReporter, BenchmarkRow
import torch.cuda.nvtx as nvtx
from contextlib import contextmanager


# Table 1 defaults
MODEL_SPECS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

MODEL_SPECS_RTX4060 = {
    "small":  dict(d_model=384, d_ff=1536, num_layers=6,  num_heads=6),   # 384/6=64
    "medium": dict(d_model=512, d_ff=2048, num_layers=8,  num_heads=8),   # 512/8=64
    "large":  dict(d_model=640, d_ff=2560, num_layers=10, num_heads=10),  # 640/10=64
    "xl":     dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),  # 768/12=64
    "2.7b":   dict(d_model=768, d_ff=3072, num_layers=16, num_heads=12),  # 768/12=64
}


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        nvtx.range_push(name)
        try:
            yield
        finally:
            nvtx.range_pop()
    else:
        yield


def torch_dtype_from_string(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {s}")


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_model(args, device: torch.device) -> torch.nn.Module:
    spec = MODEL_SPECS_RTX4060[args.model_size]
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=spec["d_model"],
        num_layers=spec["num_layers"],
        num_heads=spec["num_heads"],
        d_ff=spec["d_ff"],
        rope_theta=args.rope_theta,
        max_seq_len=args.context_length,
        eps=args.rmsnorm_eps,
        device=device,
        dtype=torch.float32  # default
    ).to(device)
    model.train()
    return model


def make_batch(args, device: torch.device) -> torch.Tensor:
    # Random token IDs (batch, seq)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device, dtype=torch.long)
    return x


def step_forward(model, x, autocast_ctx):
    with autocast_ctx:
        logits = model(x)
    return logits


def step_backward(model, logits):
    # cheap scalar loss, but builds backward
    loss = logits.float().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()


def measure_forward_backward_ms(model, x, device, autocast_ctx, nvtx_enabled: bool = False):
    """
    Return (forward_ms, backward_ms) for one step.
    Uses CUDA events when on GPU, otherwise falls back to timeit.
    """
    if device.type == "cuda":
        # CUDA event timing is more accurate than timeit + sync for segments.
        start_f = torch.cuda.Event(enable_timing=True)
        end_f = torch.cuda.Event(enable_timing=True)
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)

        start_f.record()
        with nvtx_range("forward", nvtx_enabled):
            logits = step_forward(model, x, autocast_ctx)
        end_f.record()

        start_b.record()
        with nvtx_range("backward", nvtx_enabled):
            step_backward(model, logits)
        end_b.record()

        torch.cuda.synchronize()
        f_ms = start_f.elapsed_time(end_f)
        b_ms = start_b.elapsed_time(end_b)
        return f_ms, b_ms

    # CPU fallback
    t0 = timeit.default_timer()
    logits = step_forward(model, x, autocast_ctx)
    t1 = timeit.default_timer()
    step_backward(model, logits)
    t2 = timeit.default_timer()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3


def run_benchmark_split(args, device, autocast_ctx):
    """
    Warmup then measure forward/backward separately.
    Return:
      forward_times_ms, backward_times_ms
    """
    model = make_model(args, device)
    x = make_batch(args, device)

    with nvtx_range("benchmark", args.nvtx):
        # Warmup
        with nvtx_range("warmup", args.nvtx):
            for _ in range(args.warmup_steps):
                with nvtx_range("step", args.nvtx):
                    with nvtx_range("forward_backward_time", args.nvtx):
                        f_ms, b_ms = measure_forward_backward_ms(model, x, device, autocast_ctx, nvtx_enabled=args.nvtx)
                # Already synchronized in CUDA path; keep this for safety/CPU
                sync_if_cuda(device)

        # Measure
        with nvtx_range("measure", args.nvtx):
            f_times = []
            b_times = []
            for _ in range(args.measure_steps):
                with nvtx_range("step", args.nvtx):
                    with nvtx_range("forward_backward_timed", args.nvtx):
                        f_ms, b_ms = measure_forward_backward_ms(model, x, device, autocast_ctx, nvtx_enabled=args.nvtx)
                f_times.append(f_ms)
                b_times.append(b_ms)

            return f_times, b_times


def stats_ms(times_ms):
    mean = stats.mean(times_ms)
    std = stats.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    return mean, std


def emit_row(args, reporter, device: torch.device, mode: str, mean_ms: float, std_ms: float, tok_s: float):
    row = BenchmarkRow(
        specs="RTX4060",
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        amp=args.amp,
        mode=mode,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mean_ms=mean_ms,
        std_ms=std_ms,
        tok_per_step=args.batch_size * args.context_length,
        tok_per_s=tok_s,
        device=str(device),
    )

    if reporter is not None:
        reporter.append_and_maybe_write(row, write_md=args.write_md)


def run_one_setting(args, reporter, device: torch.device, autocast_ctx):
    spec = MODEL_SPECS_RTX4060[args.model_size]

    f_times_ms, b_times_ms = run_benchmark_split(args, device, autocast_ctx)

    f_mean, f_std = stats_ms(f_times_ms)
    b_mean, b_std = stats_ms(b_times_ms)

    tok_per_step = args.batch_size * args.context_length
    f_tok_s = tok_per_step / (f_mean / 1e3) if f_mean > 0 else float("inf")
    b_tok_s = tok_per_step / (b_mean / 1e3) if b_mean > 0 else float("inf")

    print(f"[device={device}] specs=RTX4060 amp={args.amp}")
    print(f"model={args.model_size} B={args.batch_size} S={args.context_length} V={args.vocab_size}")
    print(f"warmup={args.warmup_steps} measure={args.measure_steps}")
    print(f"forward : mean={f_mean:.3f} ms, std={f_std:.3f} ms, tok/s={f_tok_s:.1f}")
    print(f"backward: mean={b_mean:.3f} ms, std={b_std:.3f} ms, tok/s={b_tok_s:.1f}")

    emit_row(args, reporter, device, "forward", f_mean, f_std, f_tok_s)
    emit_row(args, reporter, device, "backward", b_mean, b_std, b_tok_s)


def run_sweep(args, reporter, device: torch.device, autocast_ctx):
    models = [s.strip() for s in args.sweep_models.split(",") if s.strip()]
    contexts = [int(x.strip()) for x in args.sweep_contexts.split(",") if x.strip()]

    for m in models:
        for s in contexts:
            args.model_size = m
            args.context_length = s
            try:
                run_one_setting(args, reporter, device, autocast_ctx)
            except torch.OutOfMemoryError:
                print(f"[OOM] model={m} S={s}")
                emit_row(args, reporter, device, "forward", float("nan"), float("nan"), float("nan"))
                emit_row(args, reporter, device, "backward", float("nan"), float("nan"), float("nan"))
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", choices=list(MODEL_SPECS_RTX4060.keys()), default="small")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=10_000)

    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=10)

    # Mixed precision hook
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")

    # Model misc
    p.add_argument("--rope-theta", type=float, default=10_000.0)
    p.add_argument("--rmsnorm-eps", type=float, default=1e-5)

    # Determinism / matmul settings
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-tf32", action="store_true")

    # Reporter
    p.add_argument("--out-jsonl", type=str, default=None)
    p.add_argument("--out-md", type=str, default=None)
    p.add_argument("--write-md", action="store_true", help="If set, refresh markdown after appending a row.")

    # Sweep runner
    p.add_argument("--sweep", action="store_true",
                   help="Run sweep over model sizes and context lengths.")
    p.add_argument("--sweep-models", type=str, default="small,medium,large,xl,2.7b")
    p.add_argument("--sweep-contexts", type=str, default="128,256,512,1024")
    
    # NVTX
    p.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges for Nsight Systems profiling.")

    # Profiling (nsys) mode
    p.add_argument("--profile", action="store_true",
                help="Run a short workload for Nsight Systems profiling (does not affect normal benchmarking).")
    p.add_argument("--profile-mode", choices=["inference", "train_step"], default="inference",
                help="Profiling workload: inference (forward only) or train_step (forward+loss+backward+adamw).")
    p.add_argument("--profile-steps", type=int, default=1,
                help="How many measured steps to run in profiling mode.")

    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    # Autocast context (no-op on CPU)
    if device.type == "cuda" and args.amp != "none":
        amp_dtype = torch_dtype_from_string(args.amp)
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    reporter = None
    if args.out_jsonl:
        reporter = BenchmarkReporter(
            jsonl_path=args.out_jsonl,
            md_path=args.out_md,
            title="#### Forward / Backward timing"
        )

    if args.sweep:
        run_sweep(args, reporter, device, autocast_ctx)
    else:
        run_one_setting(args, reporter, device, autocast_ctx)


if __name__ == "__main__":
    main()

    # ---------------------------------------------------------------------
    # (b) Forward / backward runtime benchmarking
    #
    # Forward + backward (measures forward pass time):
    #   uv run python cs336_systems/benchmark.py \
    #     --out-jsonl runs/bench.jsonl \
    #     --out-md runs/bench.md \
    #     --write-md \
    #     --sweep \
    #     --sweep-contexts 128
    # 
    # 
    # (a) Nsys profile 
    # 
    # bash scripts/profile_nsys_models.sh
    # ---------------------------------------------------------------------
