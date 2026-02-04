import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Run:
    parameters: float       # N
    compute_budget: float   # C
    final_loss: float       # L


def load_runs(path: Path) -> List[Run]:
    data = json.loads(path.read_text())
    runs: List[Run] = []
    for r in data:
        runs.append(
            Run(
                parameters=float(r["parameters"]),
                compute_budget=float(r["compute_budget"]),
                final_loss=float(r["final_loss"]),
            )
        )
    return runs


def select_opt_points(runs: List[Run]) -> Dict[float, Run]:
    """For each compute budget C, pick the run with the lowest final_loss"""
    best: Dict[float, Run] = {}
    for r in runs:
        C = r.compute_budget
        if C not in best or r.final_loss < best[C].final_loss:
            best[C] = r
    return best


def fit_power_law(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    """
    Fit y = k * x^a via log-log linear regression:
      log(y) = log(k) + a * log(x)
    Returns (k, a).
    """
    if np.any(xs <= 0) or np.any(ys <= 0):
        raise ValueError("x and y must be positive for log-log fit.")
    lx = np.log(xs)
    ly = np.log(ys)
    a, logk = np.polyfit(lx, ly, deg=1)  # slope=a, intercept=logk
    k = float(np.exp(logk))
    return k, float(a)


def predict_power_law(k: float, a: float, x: np.ndarray) -> np.ndarray:
    return k * (x ** a)


def plot_scaling(
    x_points: np.ndarray,
    y_points: np.ndarray,
    k: float,
    a: float,
    out_path: Path,
    title: str,
    y_label: str,
    x_min: float,
    x_max: float
):
    xs = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    ys = predict_power_law(k, a, xs)

    plt.figure()
    plt.loglog(x_points, y_points, marker="o", linestyle="None", label="opt points")
    plt.loglog(xs, ys, linestyle="-", label=f"fit: y = {k:.3g} * C^{a:.3f}")
    plt.xlabel("Compute budget C (FLOPs)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/isoflops_curves.json"), help="Path to data/isoflops_curves.json",)
    ap.add_argument("--outdir", type=Path, default=Path("runs/isoflops"), help="Directory to write plots/results")
    args = ap.parse_args()

    runs = load_runs(args.data)
    best = select_opt_points(runs)

    # Sort by compute budget
    budgets = np.array(sorted(best.keys()), dtype=np.float64)
    n_opt = np.array([best[C].parameters for C in budgets], dtype=np.float64)
    d_opt = budgets / (6.0 * n_opt)

    # Fit power laws
    kN, aN = fit_power_law(budgets, n_opt)
    kD, aD = fit_power_law(budgets, d_opt)

    # Predictions required by the problem
    targets = np.array([1e23, 1e24], dtype=np.float64)
    pred_N = predict_power_law(kN, aN, targets)
    pred_D = predict_power_law(kD, aD, targets)

    # Print results
    print("=== IsoFLOPs opt points (C, N_opt, D_opt, loss) ===")
    for C in budgets:
        r = best[C]
        print(f"C={C:.3e}  N_opt={r.parameters:.3e}  D_opt={C/(6*r.parameters):.3e}  loss={r.final_loss:.6f}")

    print("\n=== Power-law fits ===")
    print(f"N_opt(C) = {kN:.6g} * C^{aN:.6f}")
    print(f"D_opt(C) = {kD:.6g} * C^{aD:.6f}")

    print("\n=== Extrapolated predictions ===")
    for C, Np, Dp in zip(targets, pred_N, pred_D):
        print(f"C={C:.1e}:  N_opt≈{Np:.3e} params,  D_opt≈{Dp:.3e} tokens")

    # Plot range: cover observed budgets and extrapolate to 1e24
    x_min = float(min(budgets.min(), 1e16))  # just in case; won't hurt
    x_max = 1e24

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_scaling(
        x_points=budgets,
        y_points=n_opt,
        k=kN,
        a=aN,
        out_path=args.outdir / "n_opt_vs_compute.png",
        title="Compute-optimal model size (IsoFLOPs)",
        y_label="N_opt (parameters)",
        x_min=x_min,
        x_max=x_max,
    )
    plot_scaling(
        x_points=budgets,
        y_points=d_opt,
        k=kD,
        a=aD,
        out_path=args.outdir / "d_opt_vs_compute.png",
        title="Compute-optimal dataset size (IsoFLOPs)",
        y_label="D_opt (tokens)",
        x_min=x_min,
        x_max=x_max,
    )

    # Save a small json for writeup convenience
    result = {
        "opt_points": [
            {
                "compute_budget": float(C),
                "n_opt": float(best[C].parameters),
                "d_opt": float(C / (6.0 * best[C].parameters)),
                "loss": float(best[C].final_loss),
            }
            for C in budgets
        ],
        "fit": {
            "n_opt": {"k": kN, "a": aN},
            "d_opt": {"k": kD, "a": aD},
        },
        "predictions": [
            {"compute_budget": float(C), "n_opt": float(Np), "d_opt": float(Dp)}
            for C, Np, Dp in zip(targets, pred_N, pred_D)
        ],
    }
    (args.outdir / "isoflops_fit.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote plots + json to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
