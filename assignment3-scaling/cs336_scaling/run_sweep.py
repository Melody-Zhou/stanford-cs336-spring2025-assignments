import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

from api_client import LossQuery, ScalingAPIClient, ScalingAPIError


# -----------------------------
# Utilities
# -----------------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def jsonl_append(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def iter_unique(seq: Iterable[LossQuery]) -> List[LossQuery]:
    seen = set()
    out: List[LossQuery] = []
    for q in seq:
        key = (
            q.d_model,
            q.num_layers,
            q.num_heads,
            q.batch_size,
            float(q.learning_rate),
            int(q.train_flops),
        )
        if key in  seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def estimate_nonemb_params(d_model: int, num_layers: int) -> float:
    # Handout tip: non-embedding params ≈ 12 * n_layer * d_model^2
    return 12.0 * num_layers * (d_model ** 2)


# -----------------------------
# Grid generator (coarse -> refine)
# -----------------------------
def coarse_grid(
    train_flops: List[int],
    batch_sizes: List[int],
    d_models: List[int],
    num_layers: List[int],
    num_heads: List[int],
    learning_rates: List[float],
) -> List[LossQuery]:
    """
    Coarse exploration:
      - fewer shapes
      - a couple lrs
      - multiple compute levels
    """
    qs: List[LossQuery] = []
    for C in train_flops:
        for bs in batch_sizes:
            for d in d_models:
                for nl in num_layers:
                    for nh in num_heads:
                        # require d_model divisible by num_heads (Transformer constraint)
                        if d % nh != 0:
                            continue
                        for lr in learning_rates:
                            qs.append(
                                LossQuery(
                                    d_model=d,
                                    num_layers=nl,
                                    num_heads=nh,
                                    batch_size=bs,
                                    learning_rate=lr,
                                    train_flops=int(C),
                                )
                            )
    return iter_unique(qs)


def refine_grid_around_best(
    best: LossQuery,
    train_flops: List[int],
    batch_sizes: List[int],
    d_model_mults: List[float],
    layer_deltas: List[int],
    head_candidates: List[int],
    lr_mults: List[float],
) -> List[LossQuery]:
    """
    Local refinement around a "best" config (by loss at some compute).
    """
    qs: List[LossQuery] = []
    for C in train_flops:
        for bs in batch_sizes:
            for dm in d_model_mults:
                d = int(round(best.d_model * dm))
                d = max(64, min(1024, d))  # API range :contentReference[oaicite:9]{index=9}
                for dl in layer_deltas:
                    nl = best.num_layers + dl
                    nl = max(2, min(24, nl))  # API range :contentReference[oaicite:10]{index=10}
                    for nh in head_candidates:
                        if d % nh != 0:
                            continue
                        for lm in lr_mults:
                            lr = float(best.learning_rate * lm)
                            # clamp to API range [1e-4, 1e-3] :contentReference[oaicite:11]{index=11}
                            lr = max(1e-4, min(1e-3, lr))
                            qs.append(
                                LossQuery(
                                    d_model=d,
                                    num_layers=nl,
                                    num_heads=nh,
                                    batch_size=bs,
                                    learning_rate=lr,
                                    train_flops=int(C),
                                )
                            )
    return iter_unique(qs)


# -----------------------------
# Budget-aware runner
# -----------------------------
def would_consume_budget(client: ScalingAPIClient, q: LossQuery) -> bool:
    """
    If cache already has this exact request, then it's free (no extra FLOPs):contentReference[oaicite:12]{index=12}.
    We check client.cache directly via the endpoint+params mapping used in api_client.py.
    """
    endpoint = "/loss"
    params = {
        "d_model": q.d_model,
        "num_layers": q.num_layers,
        "num_heads": q.num_heads,
        "batch_size": q.batch_size,
        "learning_rate": q.learning_rate,
        "train_flops": int(q.train_flops),
        "api_key": client.api_key,        
    }
    hit = client.cache.get(endpoint, params)
    return hit is None


def run_queries(
    client: ScalingAPIClient,
    queries: List[LossQuery],
    max_fit_budget_flops: float = 2e18,
    results_path: Path = Path("runs/sweep_results.jsonl"),
    dry_run: bool = False,
    sleep_s: float = 0.0,
) -> None:
    """
    Executes queries until (estimated) budget would exceed max_fit_budget_flops.
    Notes:
      - total_flops_used is returned by API and can be fetched anytime:contentReference[oaicite:13]{index=13}.
      - If we exceed the 2e18 scaling-law budget, API will refuse future requests:contentReference[oaicite:14]{index=14},
        so we stop conservatively.
    """
    # starting point from API
    try:
        used0 = float(client.total_flops_used())
    except ScalingAPIError:
        # If key has never queried, the endpoint may 422; but in that case used0=0 is safe.
        used0 = 0.0

    planned_new = 0.0
    n_new = 0
    n_cached = 0

    # Pre-pass: compute how many are cached & estimated extra FLOPs
    for q in queries:
        if would_consume_budget(client, q):
            planned_new += float(q.train_flops)
            n_new += 1
        else:
            n_cached += 1

    print("=== Sweep plan ===")
    print(f"queries_total: {len(queries)}")
    print(f"cached_free:   {n_cached}")
    print(f"new_queries:   {n_new}")
    print(f"api_used_now:  {used0:.3e} FLOPs")
    print(f"est_new_cost:  {planned_new:.3e} FLOPs")
    print(f"est_total:     {(used0 + planned_new):.3e} FLOPs")
    print(f"budget_limit:  {max_fit_budget_flops:.3e} FLOPs")

    if dry_run:
        print("\n[dry-run] not executing API calls.")
        return

    # Execute with conservative budget guard
    used = used0
    for i, q in enumerate(queries):
        is_new = would_consume_budget(client, q)
        est_after = used + (float(q.train_flops) if is_new else 0.0)

        if est_after > max_fit_budget_flops:
            print(
                f"[STOP] Budget guard: would exceed limit if running next query. "
                f"used={used:.3e}, next_cost={(q.train_flops if is_new else 0):.3e}, "
                f"limit={max_fit_budget_flops:.3e}"
            )
            return

        rec = {
            "ts_ms": now_ms(),
            "index": i,
            "query": asdict(q),
            "nonemb_params_est": estimate_nonemb_params(q.d_model, q.num_layers),
            "was_cached": (not is_new),
        }

        try:
            out = client.loss(q, use_cache=True)
            rec["response"] = out
            # Use authoritative used FLOPs if API returns it in /loss response:contentReference[oaicite:15]{index=15}
            if isinstance(out, dict) and "total_flops_used" in out:
                used = float(out["total_flops_used"])
            else:
                # fallback estimate
                used = est_after
            rec["api_used_after"] = used
            rec["status"] = "ok"
            print(
                f"[{i+1}/{len(queries)}] ok "
                f"C={q.train_flops:.1e} d={q.d_model} L={q.num_layers} H={q.num_heads} "
                f"bs={q.batch_size} lr={q.learning_rate:g} "
                f"loss={out.get('loss', None)} used={used:.3e}"
            )
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = repr(e)
            print(f"[{i+1}/{len(queries)}] error: {e!r}")

        jsonl_append(results_path, rec)

        if sleep_s > 0:
            time.sleep(sleep_s)


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("CS336_API_KEY", ""))
    p.add_argument("--base-url", default="http://hyperturing.stanford.edu:8000")
    p.add_argument("--cache", default="runs/api_cache.jsonl")
    p.add_argument("--out", default="runs/sweep_results.jsonl")
    p.add_argument("--budget", type=float, default=2e18, help="scaling-law fit budget cap (FLOPs)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sleep", type=float, default=0.0)

    sub = p.add_subparsers(dest="mode", required=True)

    # coarse mode
    c = sub.add_parser("coarse")
    c.add_argument("--train-flops", nargs="+", type=float, default=[1e13, 1e14, 1e15, 1e16, 1e17, 1e18])
    c.add_argument("--batch-sizes", nargs="+", type=int, default=[128])
    c.add_argument("--d-models", nargs="+", type=int, default=[128, 256, 512, 768, 1024])
    c.add_argument("--num-layers", nargs="+", type=int, default=[2, 4, 8, 12, 16, 24])
    c.add_argument("--num-heads", nargs="+", type=int, default=[2, 4, 8, 16])
    c.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])

    # refine mode: requires a seed config
    r = sub.add_parser("refine")
    r.add_argument("--seed-d-model", type=int, required=True)
    r.add_argument("--seed-num-layers", type=int, required=True)
    r.add_argument("--seed-num-heads", type=int, required=True)
    r.add_argument("--seed-batch-size", type=int, required=True)
    r.add_argument("--seed-learning-rate", type=float, required=True)
    r.add_argument("--train-flops", nargs="+", type=float, default=[1e16, 3e16, 1e17, 3e17, 1e18])
    r.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 256])
    r.add_argument("--d-model-mults", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    r.add_argument("--layer-deltas", nargs="+", type=int, default=[-2, 0, 2])
    r.add_argument("--head-candidates", nargs="+", type=int, default=[2, 4, 8, 16])
    r.add_argument("--lr-mults", nargs="+", type=float, default=[0.5, 1.0, 2.0])

    args = p.parse_args()
    if not args.api_key:
        raise SystemExit("Missing --api-key or env CS336_API_KEY")

    client = ScalingAPIClient(
        api_key=args.api_key,
        base_url=args.base_url,
        cache_path=args.cache,
    )

    out_path = Path(args.out)

    if args.mode == "coarse":
        queries = coarse_grid(
            train_flops=[int(x) for x in args.train_flops],
            batch_sizes=args.batch_sizes,
            d_models=args.d_models,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            learning_rates=args.learning_rates,
        )

    else:
        seed = LossQuery(
            d_model=args.seed_d_model,
            num_layers=args.seed_num_layers,
            num_heads=args.seed_num_heads,
            batch_size=args.seed_batch_size,
            learning_rate=args.seed_learning_rate,
            train_flops=int(1e13),  # placeholder; replaced by --train-flops below
        )
        queries = refine_grid_around_best(
            best=seed,
            train_flops=[int(x) for x in args.train_flops],
            batch_sizes=args.batch_sizes,
            d_model_mults=args.d_model_mults,
            layer_deltas=args.layer_deltas,
            head_candidates=args.head_candidates,
            lr_mults=args.lr_mults,
        )

    run_queries(
        client=client,
        queries=queries,
        max_fit_budget_flops=float(args.budget),
        results_path=out_path,
        dry_run=bool(args.dry_run),
        sleep_s=float(args.sleep),
    )


if __name__ == "__main__":
    main()
