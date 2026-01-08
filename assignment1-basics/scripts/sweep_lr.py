import gc
import time
import torch
from typing import Sequence
import cs336_basics.config as config_mod
import cs336_basics.train as train_mod

def run_one(lr_max: float, *, lr_min_ratio: float = 0.1, tag: str = "") -> None:
    """
    Run one training job by patching train.get_default_config() to return a modified config.
    """
    cfg = config_mod.get_default_config()

    # ---- override what you need for sweep ----
    cfg.optim.lr_max = float(lr_max)
    cfg.optim.lr_min = float(lr_max * lr_min_ratio)  # IMPORTANT: set explicitly (don't rely on dataclass default)
    lr_tag = "lr" + f"{lr_max:.4g}".replace(".", "p").replace("-", "m")
    cfg.run.run_name = f"{lr_tag}{('_' + tag) if tag else ''}"

    # cfg.wandb.enable = False

    # ---- patch train_mod.get_default_config ----
    def _patched_get_default_config():
        return cfg

    train_mod.get_default_config = _patched_get_default_config

    print("\n" + "=" * 80)
    print(f"[SWEEP] start run: lr_max={cfg.optim.lr_max} lr_min={cfg.optim.lr_min} run_name={cfg.run.run_name}")
    print("=" * 80)

    # run training
    train_mod.main()

    # cleanup to reduce GPU memory fragmentation across runs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main(lrs: Sequence[float]) -> None:
    t0 = time.time()
    for lr in lrs:
        run_one(lr, lr_min_ratio=0.1)
    dt = time.time() - t0
    print(f"\n[SWEEP] all done. total wall time: {dt/60:.1f} min")

if __name__ == "__main__":
    # coarse search
    # lrs = [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1e-2]
    # refinement
    lrs = [5e-4, 8e-4, 1.0e-3, 1.2e-3, 1.5e-3, 1.8e-3, 2.0e-3, 2.2e-3]

    main(lrs)
