import os
import time
import math
import torch
import numpy as np
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy_from_logits, load_checkpoint, save_checkpoint, clip_grad_norm
from cs336_basics.config import get_default_config
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.optimizer import AdamW, lr_cosine_schedule_with_warmup
from cs336_basics.experiment_tracking import ExperimentTracker

def open_memmap_1d(path: str, np_dtype: str) -> np.memmap:
    """
    Open a 1D token memmap file. The file is assumed to be a raw binary array.
    """
    dtype = np.dtype(np_dtype)
    itemsize = dtype.itemsize
    nbytes = os.path.getsize(path)
    if nbytes % itemsize != 0:
        raise ValueError(f"File size is not divisible by dtype size: {path} ({nbytes} bytes, itemsize={itemsize})")
    length = nbytes // itemsize
    return np.memmap(path, mode="r", dtype=dtype, shape=(length,))

def torch_dtype_from_string(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype string: {s}")

def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, data: np.memmap, cfg) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.train.eval_batches):
        xb, yb = get_batch(
            dataset=data,
            batch_size=cfg.train.batch_size,
            context_length=cfg.data.context_length,
            device=cfg.data.device
        )
        logits = model(xb)  # (B, S, V)
        B, S, V = logits.shape
        loss = cross_entropy_from_logits(logits.reshape(B * S, V), yb.reshape(B * S))
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))

def main() -> None:
    # 1. Load configuration and set random seed
    cfg = get_default_config()

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # 2. Optional experiment tracking (weights & biases)
    if cfg.run.run_name is not None:
        run_name = f"{cfg.run.run_name_prefix}_{cfg.run.run_name}"
    else:
        run_name = f"{cfg.run.run_name_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"

    run_dir = os.path.join(cfg.run.runs_dir, run_name)

    tracker = ExperimentTracker(
        run_dir=run_dir,
        config=cfg,  # dataclass will be serialized
        use_wandb=getattr(cfg.wandb, "enable", False),
        wandb_project=getattr(cfg.wandb, "project", "cs336-a1"),
        wandb_run_name=getattr(cfg.wandb, "run_name", run_name)
    )

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "ckpt.best.pt")

    # 3. Load datasets (memory-mapped)
    train_mm = open_memmap_1d(cfg.data.train_data_path, cfg.data.np_dtype)
    val_mm = open_memmap_1d(cfg.data.val_data_path, cfg.data.np_dtype)

    # 4. Create model and move it to the target device
    device = torch.device(cfg.data.device)
    model_dtype = torch_dtype_from_string(cfg.model.torch_dtype)

    d_ff = cfg.model.d_ff if cfg.model.d_ff is not None else 4 * cfg.model.d_model 

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=d_ff,
        rope_theta=cfg.model.rope_theta,
        max_seq_len=cfg.model.max_seq_len,
        eps=cfg.model.rmsnorm_eps,
        device=device,
        dtype=model_dtype
    ).to(device)

    # 5. Create optimizer and (optionally) resume from checkpoint
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr_max,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay
    )

    start_it = 0
    resume_path = cfg.train.resume_from
    if resume_path is None:
        resume_path = ckpt_path
    elif not os.path.isabs(resume_path):
        resume_path = os.path.join(run_dir, resume_path)

    if resume_path is not None and os.path.exists(resume_path):
        start_it = load_checkpoint(resume_path, model, optimizer)
        print(f"[resume] loaded checkpoint: {resume_path} (start_it={start_it})")

    # 6. Training loop initialization
    best_val = float("inf")
    last_log_t = time.time()

    # 7. Main training loop
    for it in range(start_it, cfg.train.max_steps):
        # 7.1 Update learning rate according to schedule
        lr = lr_cosine_schedule_with_warmup(
            t=it, 
            alpha_max=cfg.optim.lr_max,
            alpha_min=cfg.optim.lr_min,
            T_w=cfg.optim.warmup_iters,
            T_c=cfg.optim.cosine_cycle_iters
        )
        set_optimizer_lr(optimizer, lr)

        # 7.2 Sample a batch of training data
        xb, yb = get_batch(
            train_mm,
            batch_size=cfg.train.batch_size,
            context_length=cfg.data.context_length,
            device=cfg.data.device
        )

        # 7.3 Forward pass and loss computation
        logits = model(xb)  # (B, S, V)
        B, S, V = logits.shape
        loss = cross_entropy_from_logits(logits.reshape(B * S, V), yb.reshape(B * S))

        # 7.4 Backward pass (gradient computation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 7.5 Gradient clipping for training stability
        if cfg.optim.grad_clip > 0:
            clip_grad_norm(model.parameters(), cfg.optim.grad_clip, eps=1e-6)
        
        # 7.6 Optimizer step (parameter update)
        optimizer.step()

        # 7.7 Periodic training metrics logging
        if (it + 1) % cfg.train.log_interval == 0:
            now = time.time()
            dt = max(now - last_log_t, 1e-9)
            tok_s = (cfg.train.batch_size * cfg.data.context_length * cfg.train.log_interval) / dt
            msg = f"it={it+1} loss={loss.item():.4f} lr={lr:.3e} tok/s={tok_s:.1f}"
            print(msg)
            tracker.log(step=it + 1, metrics={"train/loss": float(loss.item()), "train/lr": float(lr), "train/tok_s": float(tok_s)})
            last_log_t = now
        
        # 7.8 Periodic evaluation on validation set
        if (it + 1) % cfg.train.eval_interval == 0:
            val_loss = estimate_loss(model, val_mm, cfg)
            val_ppl = float(math.exp(val_loss))
            print(f"[eval] it={it+1} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
            tracker.log(step=it + 1, metrics={"val/loss": float(val_loss), "val/ppl": float(val_ppl)})

            # Save the best-performing checkpoint
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, it + 1, best_ckpt_path)

        # 7.9 Periodic checkpointing
        if (it + 1) % cfg.train.ckpt_interval == 0:
            save_checkpoint(model, optimizer, it + 1, ckpt_path)

    # 8. Final checkpoint adn cleanup 
    save_checkpoint(model, optimizer, cfg.train.max_steps, ckpt_path)
    tracker.close()

if __name__ == "__main__":
    main()