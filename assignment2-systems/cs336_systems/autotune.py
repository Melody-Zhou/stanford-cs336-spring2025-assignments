import torch

import cs336_systems.flash_triton as ft
from cs336_systems.flash_triton import FlashAttention2Triton


def flash(q, k, v):
    return FlashAttention2Triton.apply(q, k, v, True)


def main():
    B = 1
    S = 4096
    H = 16
    Dh = 64
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(B * H, S, Dh, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B * H, S, Dh, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B * H, S, Dh, device=device, dtype=dtype, requires_grad=True)

    _ = flash(q, k, v)
    torch.cuda.synchronize()

    kern = ft.flash_fwd_kernel

    print("kernel type:", type(kern))
    print("keys:", getattr(kern, "keys", None))
    print("cache size:", len(getattr(kern, "cache", {})))

    bc = getattr(kern, "best_config", None)
    if bc is not None:
        print("best.kwargs:", bc.kwargs)
        print("best.num_warps:", bc.num_warps, "best.num_stages:", bc.num_stages)

    cache = getattr(kern, "cache", {})
    if cache:
        for key, best in cache.items():
            print("\nkey:", key)
            print("  best.kwargs:", best.kwargs)
            print("  best.num_warps:", best.num_warps, "best.num_stages:", best.num_stages)

    timings = getattr(kern, "configs_timings", None)
    if timings:
        print("\nconfigs_timings:")
        for cfg, ts in timings.items():
            print("  cfg.kwargs:", cfg.kwargs, "warps:", cfg.num_warps, "stages:", cfg.num_stages, "times:", ts)


if __name__ == "__main__":
    main()
