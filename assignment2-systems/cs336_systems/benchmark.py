import torch
import timeit
import argparse
import statistics as stats
from contextlib import nullcontext
from cs336_basics.transformer_lm import TransformerLM


# Table 1 defaults
MODEL_SPECS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def torch_dtype_from_string(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {s}")


def sycn_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_model(args, device: torch.device) -> torch.nn.Module:
    spec = MODEL_SPECS[args.model_size]
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


def run_one_step(model, x, args, autocast_ctx):
    # Forward
    with autocast_ctx:
        logits = model(x)   # (B, S, V)
        if args.forward_only:
            loss = None
        else:
            # Use a cheap scalar loss that still builds a full backward graph
            loss = logits.float().mean()
    
    # Backward
    if not args.forward_only:
        model.zero_grad(set_to_none=True)
        loss.backward()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", choices=list(MODEL_SPECS.keys()), default="small")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=10_000)

    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=10)
    p.add_argument("--forward-only", action="store_true")

    # Mixed precision hook
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")

    # Model misc
    p.add_argument("--rope-theta", type=float, default=10_000.0)
    p.add_argument("--rmsnorm-eps", type=float, default=1e-5)

    # Determinism / matmul settings
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-tf32", action="store_true")

    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    model = make_model(args, device)
    x = make_batch(args, device)

    # Autocast context (no-op on CPU)
    if device.type == "cuda" and args.amp != "none":
        amp_dtype = torch_dtype_from_string(args.amp)
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    # Warm-up
    for _ in range(args.warmup_steps):
        run_one_step(model, x, args, autocast_ctx)
        sycn_if_cuda(device)

    # Timed steps
    times = []
    for _ in range(args.measure_steps):
        t0 = timeit.default_timer()
        run_one_step(model, x, args, autocast_ctx)
        sycn_if_cuda(device)
        t1 = timeit.default_timer()
        times.append(t1 - t0)

    mean_s = stats.mean(times)
    std_s = stats.pstdev(times) if len(times) > 1 else 0.0

    tok_per_step = args.batch_size * args.context_length
    tok_s = tok_per_step / mean_s if mean_s > 0 else float("inf")

    mode = "forward" if args.forward_only else "forward+backward"
    print(f"[device={device}] mode={mode} amp={args.amp}")
    print(f"model={args.model_size} B={args.batch_size} S={args.context_length} V={args.vocab_size}")
    print(f"warmup={args.warmup_steps} measure={args.measure_steps}")
    print(f"step time: mean={mean_s*1e3:.3f} ms, std={std_s*1e3:.3f} ms, tok/s={tok_s:.1f}")

if __name__ == "__main__":
    main()