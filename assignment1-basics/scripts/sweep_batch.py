import gc
import time
import torch
from typing import Sequence
import cs336_basics.config as config_mod
import cs336_basics.train as train_mod


def run_one(batch_size: int, tag: str = "") -> None:
    """
    Run one training job by patching train.get_default_config() to return
    a config with a fixed LR and varying batch size.
    """
    cfg = config_mod.get_default_config()

    # ---- fixed LR (from your LR sweep result) ----
    cfg.optim.lr_max = 1e-3
    cfg.optim.lr_min = 1e-4

    # ---- variable: batch size ----
    cfg.train.batch_size = int(batch_size)

    # short sweep run (keep cost low)
    cfg.train.max_steps = 3000
    cfg.optim.warmup_iters = 200
    cfg.optim.cosine_cycle_iters = 3000
    cfg.train.eval_interval = 500

    # name = bs only (single variable!)
    cfg.run.run_name = f"bs{batch_size}{('_' + tag) if tag else ''}"

    # cfg.wandb.enable = False

    # ---- patch train_mod.get_default_config ----
    def _patched_get_default_config():
        return cfg

    train_mod.get_default_config = _patched_get_default_config

    print("\n" + "=" * 80)
    print(
        f"[SWEEP] start run: "
        f"batch_size={cfg.train.batch_size} "
        f"lr_max={cfg.optim.lr_max} "
        f"run_name={cfg.run.run_name}"
    )
    print("=" * 80)

    train_mod.main()

    # cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(batch_sizes: Sequence[int]) -> None:
    t0 = time.time()
    for bs in batch_sizes:
        run_one(bs, tag="batch_sweep")
    dt = time.time() - t0
    print(f"\n[SWEEP] all done. total wall time: {dt/60:.1f} min")


if __name__ == "__main__":
    # Given your hardware limit
    batch_sizes = [1, 2, 4, 8, 12]
    main(batch_sizes)
