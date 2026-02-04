import argparse
import os
from pprint import pprint

from api_client import LossQuery, ScalingAPIClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("CS336_API_KEY", ""))
    p.add_argument("--base-url", default="http://hyperturing.stanford.edu:8000")
    p.add_argument("--cache", default="runs/api_cache.jsonl")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("total_flops_used")
    sub.add_parser("previous_runs")

    q = sub.add_parser("loss")
    q.add_argument("--d-model", type=int, required=True)
    q.add_argument("--num-layers", type=int, required=True)
    q.add_argument("--num-heads", type=int, required=True)
    q.add_argument("--batch-size", type=int, required=True)
    q.add_argument("--learning-rate", type=float, required=True)
    q.add_argument("--train-flops", type=int, required=True)

    args = p.parse_args()
    if not args.api_key:
        raise SystemExit("Missing --api-key or env CS336_API_KEY")

    client = ScalingAPIClient(
        api_key=args.api_key,
        base_url=args.base_url,
        cache_path=args.cache,
    )

    if args.cmd == "total_flops_used":
        print(client.total_flops_used())
        return

    if args.cmd == "previous_runs":
        pprint(client.previous_runs())
        return

    if args.cmd == "loss":
        out = client.loss(
            LossQuery(
                d_model=args.d_model,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                train_flops=args.train_flops,
            )
        )
        pprint(out)
        return


if __name__ == "__main__":
    main()
