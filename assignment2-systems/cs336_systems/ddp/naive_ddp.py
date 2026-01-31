import argparse
import os
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    

def setup_process_group(rank: int, world_size: int, backend: str, master_addr: str, master_port: str) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def broadcast_model_from_rank0(model: nn.Module) -> None:
    """Make all ranks start from identical parameters (rank0 as source)."""
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def allreduce_gradients(model: nn.Module) -> None:
    """Naive DDP: all-reduce every parameter's gradient, then average."""
    world_size = dist.get_world_size()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def make_random_dataset(
    seed: int, n: int, in_dim: int, out_dim: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a fixed random regression dataset (same across ranks)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    y = torch.randn(n, out_dim, generator=g)
    return x.to(device), y.to(device)


def single_process_train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float
) -> nn.Module:
    """Baseline: single process trains on the full dataset each step."""
    model = deepcopy(model).to(x.device)
    loss_fn = nn.MSELoss()
    opt = optim.SGD(model.parameters(), lr=lr)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    return model


def ddp_worker(
    rank: int,
    world_size: int,
    backend: str,
    use_cuda: bool,
    master_addr: str,
    master_port: str,
    seed_model: int,
    seed_data: int,
    n: int,
    in_dim: int,
    out_dim: int,
    steps: int,
    lr: float,
    return_dict,
) -> None:
    try:
        setup_process_group(rank, world_size, backend, master_addr, master_port)

        # device mapping: rank -> cuda:rank
        if use_cuda:
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        # Build model with rank-dependent seed first (to prove broadcast works)
        torch.manual_seed(seed_model + rank)
        model = ToyModel(in_dim, out_dim).to(device)

        # make all ranks start from rank0 params
        broadcast_model_from_rank0(model)
        dist.barrier()

        # Prepare identical random dataset on each rank, then shard it
        x_all, y_all = make_random_dataset(seed=seed_data, n=n, in_dim=in_dim, out_dim=out_dim, device=device)
        assert n % world_size == 0
        local_bs = n // world_size
        start = rank * local_bs
        x_local = x_all[start : start + local_bs]
        y_local = y_all[start : start + local_bs]

        loss_fn = nn.MSELoss()
        opt = optim.SGD(model.parameters(), lr=lr)

        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            out = model(x_local)
            loss = loss_fn(out, y_local)
            loss.backward()

            # Key point: After backpropagation, perform an all-reduce operation (and average) on the gradients of each parameter
            allreduce_gradients(model)

            sync_if_cuda(device)
            opt.step()
            sync_if_cuda(device)

        # Return final weights to rank0 for verification
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return_dict[rank] = state

        dist.barrier()
    finally:
        cleanup_process_group()

    
def compare_state_dicts(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Tuple[bool, List[str]]:
    mismatched = []
    for k in a.keys():
        if not torch.allclose(a[k], b[k]):
            mismatched.append(k)
    return (len(mismatched) == 0), mismatched


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=str, default="29510")
    p.add_argument("--seed-model", type=int, default=0)
    p.add_argument("--seed-data", type=int, default=123)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--in-dim", type=int, default=16)
    p.add_argument("--out-dim", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1)
    args = p.parse_args()

    # Device device/backend
    use_cuda = torch.cuda.is_available() and args.backend == "nccl"
    if args.backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("backend=nccl requires CUDA, but CUDA is not available.")
    if use_cuda and args.world_size > torch.cuda.device_count():
        raise RuntimeError(f"world_size={args.world_size} > cuda_device_count={torch.cuda.device_count()}")

    # Build rank0 baseline model + baseline dataset on rank0 device
    torch.manual_seed(args.seed_model + 0)
    baseline_device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    baseline_model = ToyModel(args.in_dim, args.out_dim).to(baseline_device)
    x_full, y_full = make_random_dataset(args.seed_data, args.n, args.in_dim, args.out_dim, baseline_device)

    baseline_trained = single_process_train(
        model=baseline_model,
        x=x_full,
        y=y_full,
        steps=args.steps,
        lr=args.lr
    )
    baseline_state = {k: v.detach().cpu() for k, v in baseline_trained.state_dict().items()}

    # Run naive DDP
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        ddp_worker,
        args=(
            args.world_size,
            args.backend,
            use_cuda,
            args.master_addr,
            args.master_port,
            args.seed_model,
            args.seed_data,
            args.n,
            args.in_dim,
            args.out_dim,
            args.steps,
            args.lr,
            return_dict,
        ),
        nprocs=args.world_size,
        join=True,
    )

    # Verify: all ranks match baseline
    ok_all = True
    for r in range(args.world_size):
        ddp_state = return_dict[r]
        ok, mismateched = compare_state_dicts(baseline_state, ddp_state)
        if not ok:
            ok_all = False
            print(f"[FALL] rank {r} differs from single-process baseline. mismatched keys: {mismateched}")
        else:
            print(f"[OK] rank {r} matches single-process baseline exactly.")
        
    
    if ok_all:
        print("\nNaive DDP correctness check passed: DDP weights == single-process weights.")
    else:
        raise SystemExit("\nNaive DDP correctness check failed.")

    
if __name__ == "__main__":
    main()