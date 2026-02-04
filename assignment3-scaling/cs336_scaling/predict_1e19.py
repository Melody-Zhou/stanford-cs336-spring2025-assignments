import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from api_client import LossQuery
from scaling_data import approx_nonemb_params


ALLOWED_D_MODEL = [64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 896, 1024]
ALLOWED_LAYERS = list(range(2, 25))  # 2..24
ALLOWED_HEADS = [2, 4, 8, 16]
ALLOWED_BATCH = [128, 256]


def find_closest_arch(target_N: float) -> Tuple[Dict, float]:
    """
    brute-force search over allowed ranges to find (d_model, num_layers, num_heads)
    that yields N_est closest to target_N, with d_model % num_heads == 0.
    """
    best = None
    best_err = float("inf")
    best_N = None

    for d in ALLOWED_D_MODEL:
        for nl in ALLOWED_LAYERS:
            for nh in ALLOWED_HEADS:
                if d % nh != 0:
                    continue
                N = approx_nonemb_params(d, nl)
                err = abs(N - target_N) / target_N
                if err < best_err:
                    best_err = err
                    best = {"d_model": d, "num_layers": nl, "num_heads": nh}
                    best_N = N
    return best, float(best_N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, default=Path("runs/scaling_fit/scaling_fit.json"))
    ap.add_argument("--budget", type=float, default=1e19)
    ap.add_argument("--batch", type=int, choices=ALLOWED_BATCH, default=256)
    ap.add_argument("--lr", type=float, default=None, help="override learning rate; if None, use lr from best point at max C")
    args = ap.parse_args()

    data = json.loads(args.fit.read_text())
    nfit = data["fit"]["n_opt"]
    lfit = data["fit"]["l_opt"]

    C = float(args.budget)

    # Predictions
    N_pred = float(nfit["k"]) * (C ** float(nfit["a"]))
    L_pred = float(lfit["L_inf"]) + float(lfit["k"]) * (C ** (-float(lfit["a"])))

    # pick lr from the best point at maximum observed C (usually 1e18) unless overridden
    best_points = data["best_points"]
    maxC_point = max(best_points, key=lambda x: x["train_flops"])
    lr = float(args.lr) if args.lr is not None else float(maxC_point["learning_rate"])

    arch, N_arch = find_closest_arch(N_pred)

    suggested = LossQuery(
        d_model=int(arch["d_model"]),
        num_layers=int(arch["num_layers"]),
        num_heads=int(arch["num_heads"]),
        batch_size=int(args.batch),
        learning_rate=float(lr),
        train_flops=int(1e18),  # API only supports up to 1e18; 1e19 is for your final report prediction
    )

    print("=== Scaling-law prediction at 1e19 FLOPs ===")
    print(f"Predicted N_opt (non-emb est) : {N_pred:.3e}")
    print(f"Predicted L_opt               : {L_pred:.6f}")
    print()
    print("=== Closest feasible architecture (API domain) ===")
    print(f"arch = {arch}, N_est={N_arch:.3e}, rel_err={abs(N_arch-N_pred)/N_pred:.3%}")
    print()
    print("=== Suggested training hyperparams (submit) ===")
    print(f"batch_size must be 128 or 256 (you chose {args.batch}).")  # handout requirement:contentReference[oaicite:6]{index=6}
    print(f"learning_rate = {lr:g}  (default from best @ max observed compute)")
    print()
    print("NOTE: API train_flops max is 1e18; 1e19 values are extrapolated for the report/submission.")
    print("Suggested config (for Google form):")
    print(json.dumps({
        "model_size_nonemb_params_est": N_arch,   # what you'll report as "model size"
        "arch": arch,
        "batch_size": args.batch,
        "learning_rate": lr,
        "predicted_loss_at_1e19": L_pred,
    }, indent=2))


if __name__ == "__main__":
    main()
