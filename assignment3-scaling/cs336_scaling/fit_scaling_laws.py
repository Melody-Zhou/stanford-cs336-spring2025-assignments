import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scaling_data import approx_nonemb_params, group_best_by_compute, load_sweep_jsonl


def fit_powerlaw(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit y = k * x^a using log-log linear regression.
    Returns (k, a).
    """
    lx = np.log(x)
    ly = np.log(y)
    a, logk = np.polyfit(lx, ly, 1)
    return float(np.exp(logk)), float(a)


def fit_loss_with_floor(C: np.ndarray, L: np.ndarray) -> Dict[str, float]:
    """
    Fit L(C) = L_inf + k * C^{-a}
    via a simple grid search over L_inf and log-log fit on (L - L_inf).
    This is robust and dependency-free.
    """
    # L_inf must be below min(L)
    Lmin = float(np.min(L))
    # a conservative grid: from Lmin-2.0 down to Lmin-0.01
    # (you can widen if needed)
    candidates = np.linspace(Lmin - 2.0, Lmin - 0.01, 200)

    best = {"L_inf": None, "k": None, "a": None, "mse": float("inf")}
    for Linf in candidates:
        y = L - Linf
        if np.any(y <= 0):
            continue
        k, a = fit_powerlaw(C, y)          # y = k * C^a, but we need y = k * C^{-aL}
        # In our parameterization: y = k * C^{-aL} => log y = log k - aL log C
        # So slope returned is a = -aL
        aL = -a
        pred = Linf + k * (C ** (-aL))
        mse = float(np.mean((pred - L) ** 2))
        if mse < best["mse"]:
            best = {"L_inf": float(Linf), "k": float(k), "a": float(aL), "mse": mse}

    if best["L_inf"] is None:
        raise RuntimeError("Failed to fit L(C)=L_inf+k*C^{-a}: no valid Linf candidate.")
    return best


def plot_loglog_points_and_fit(x, y, fit_fn, out_path: Path, title: str, ylab: str):
    xs = np.logspace(np.log10(min(x)), np.log10(max(x)), 300)
    ys = fit_fn(xs)
    plt.figure()
    plt.loglog(x, y, marker="o", linestyle="None", label="best points")
    plt.loglog(xs, ys, linestyle="-", label="fit")
    plt.xlabel("Compute budget C (FLOPs)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=Path("runs/sweep_results.jsonl"))
    ap.add_argument("--outdir", type=Path, default=Path("runs/scaling_fit"))
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    rows = load_sweep_jsonl(args.sweep)
    best = group_best_by_compute(rows)

    # Sort by compute
    Cs = np.array(sorted(best.keys()), dtype=np.float64)
    Ns = np.array([approx_nonemb_params(best[int(C)].d_model, best[int(C)].num_layers) for C in Cs], dtype=np.float64)
    Ls = np.array([best[int(C)].loss for C in Cs], dtype=np.float64)

    # Fit N_opt(C) = kN * C^aN
    kN, aN = fit_powerlaw(Cs, Ns)

    # Fit L_opt(C) = L_inf + kL * C^{-aL}
    loss_fit = fit_loss_with_floor(Cs, Ls)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Save points (for writeup tables / plots)
    csv_path = args.outdir / "scaling_fit_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train_flops", "loss_best", "d_model", "num_layers", "num_heads", "batch_size", "learning_rate", "n_nonemb_params_est"])
        for C in Cs:
            r = best[int(C)]
            w.writerow([
                int(C), r.loss, r.d_model, r.num_layers, r.num_heads, r.batch_size, r.learning_rate,
                approx_nonemb_params(r.d_model, r.num_layers),
            ])

    # Save fit params
    out = {
        "best_points": [
            {
                "train_flops": int(C),
                "loss": float(best[int(C)].loss),
                "d_model": int(best[int(C)].d_model),
                "num_layers": int(best[int(C)].num_layers),
                "num_heads": int(best[int(C)].num_heads),
                "batch_size": int(best[int(C)].batch_size),
                "learning_rate": float(best[int(C)].learning_rate),
                "n_nonemb_params_est": float(approx_nonemb_params(best[int(C)].d_model, best[int(C)].num_layers)),
            }
            for C in Cs
        ],
        "fit": {
            "n_opt": {"k": kN, "a": aN, "form": "N_opt(C)=k*C^a"},
            "l_opt": {
                "L_inf": loss_fit["L_inf"],
                "k": loss_fit["k"],
                "a": loss_fit["a"],
                "mse": loss_fit["mse"],
                "form": "L_opt(C)=L_inf + k*C^{-a}",
            },
        },
    }
    (args.outdir / "scaling_fit.json").write_text(json.dumps(out, indent=2))

    print("=== Fit results ===")
    print(f"N_opt(C) = {kN:.6g} * C^{aN:.6f}")
    print(f"L_opt(C) = {loss_fit['L_inf']:.6f} + {loss_fit['k']:.6g} * C^(-{loss_fit['a']:.6f})")
    print(f"Saved: {args.outdir/'scaling_fit.json'}")
    print(f"Saved: {csv_path}")

    if args.make_plots:
        plot_loglog_points_and_fit(
            Cs, Ns,
            lambda x: kN * (x ** aN),
            args.outdir / "nopt_vs_c.png",
            title="Compute-optimal model size (from best-per-C points)",
            ylab="N_nonemb_params_est",
        )
        # For loss, loglog doesn't work with floor directly; plot (L-L_inf) for loglog visualization
        Linf = loss_fit["L_inf"]
        y = Ls - Linf
        plot_loglog_points_and_fit(
            Cs, y,
            lambda x: loss_fit["k"] * (x ** (-loss_fit["a"])),
            args.outdir / "lopt_minus_linf_vs_c.png",
            title="Compute-optimal loss gap (L - L_inf)",
            ylab="L_opt - L_inf",
        )
        print("Saved plots to outdir.")


if __name__ == "__main__":
    main()
