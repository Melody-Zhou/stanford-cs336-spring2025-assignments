from typing import Callable, Literal
import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Returns:
        normalized_rewards: (B,)
        raw_rewards: (B,)
        metadata: dict
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have same length")

    B = len(rollout_responses)
    if B % group_size != 0:
        raise ValueError(f"rollout_batch_size={B} must be divisible by group_size={group_size}")

    # 1) raw rewards
    raw = []
    fmt = []
    ans = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        out = reward_fn(resp, gt)
        r = float(out["reward"])
        raw.append(r)
        fmt.append(float(out.get("format_reward", r)))
        ans.append(float(out.get("answer_reward", r)))

    raw_rewards = torch.tensor(raw, dtype=torch.float32)  # (B,)

    # 2) group normalization
    n_groups = B // group_size
    grouped = raw_rewards.view(n_groups, group_size)  # (G, group_size)
    mean = grouped.mean(dim=1, keepdim=True)          # (G, 1)
    centered = grouped - mean                         # (G, group_size)

    if normalize_by_std:
        std = grouped.std(dim=1, keepdim=True)  # populaiton std
        normalized = centered / (std + advantage_eps)
    else:
        normalized = centered
    
    normalized_rewards = normalized.reshape(-1)  # (B,)

    metadata = {
        "raw_reward_mean": float(raw_rewards.mean().item()),
        "raw_reward_std": float(raw_rewards.std().item()),
        "raw_reward_min": float(raw_rewards.min().item()),
        "raw_reward_max": float(raw_rewards.max().item()),
        "format_reward_mean": float(torch.tensor(fmt).mean().item()),
        "answer_reward_mean": float(torch.tensor(ans).mean().item()),
    }
    return normalized_rewards, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        raw_rewards_or_advantages: (batch_size, 1)
        policy_log_probs: (batch_size, seq_len)

    Returns:
        per-token loss: (batch_size, seq_len)
    """
    assert raw_rewards_or_advantages.ndim == 2 and raw_rewards_or_advantages.shape[1] == 1, \
        f"expected raw_rewards_or_advantages shape (B,1), got {tuple(raw_rewards_or_advantages.shape)}"
    assert policy_log_probs.ndim == 2, \
        f"expected policy_log_probs shape (B,T), got {tuple(policy_log_probs.shape)}"
    assert raw_rewards_or_advantages.shape[0] == policy_log_probs.shape[0], \
        "batch size mismatch"

    # broadcast (B,1) -> (B,T)
    advantages = raw_rewards_or_advantages.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)
    return -advantages * policy_log_probs


def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,          # (B, 1)
    policy_log_probs: torch.Tensor,    # (B, T)
    old_log_probs: torch.Tensor,       # (B, T)
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert advantages.ndim == 2 and advantages.shape[1] == 1, \
        f"expected advantages (B,1), got {tuple(advantages.shape)}"
    assert policy_log_probs.ndim == 2, \
        f"expected policy_log_probs (B,T), got {tuple(policy_log_probs.shape)}"
    assert policy_log_probs.shape == old_log_probs.shape, \
        "policy_log_probs and old_log_probs must have same shape"
    assert advantages.shape[0] == policy_log_probs.shape[0], \
        "batch size mismatch"

    adv = advantages.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Unclipped surrogate: - ratio * A
    loss = -(ratio * adv)

    metadata = {
        "ratio": ratio.detach(),
        "log_ratio": log_ratio.detach(),
        "ratio_mean": ratio.detach().mean(),
        "ratio_max": ratio.detach().max(),
        "ratio_min": ratio.detach().min(),
        "approx_kl": (-(log_ratio)).detach().mean(),
    }
    return loss, metadata


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: (B, 1)
        policy_log_probs: (B, T)
        old_log_probs: (B, T)
        cliprange: epsilon

    Returns:
        loss: (B, T)  per-token GRPO-Clip loss
        metadata: dict[str, torch.Tensor]
    """
    assert advantages.ndim == 2 and advantages.shape[1] == 1, \
        f"expected advantages shape (B,1), got {tuple(advantages.shape)}"
    assert policy_log_probs.shape == old_log_probs.shape, \
        "policy_log_probs and old_log_probs must have the same shape"
    assert policy_log_probs.ndim == 2, \
        f"expected log_probs shape (B,T), got {tuple(policy_log_probs.shape)}"
    assert advantages.shape[0] == policy_log_probs.shape[0], \
        "batch size mismatch"
    
    # broadcast (B, 1) -> (B, T)
    adv = advantages.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)

    # ratio = pi_theta / pi_old = exp(logpi - logpi_old)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # unclipped and clipped objectives
    pg_unclipped = ratio * adv
    ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_clipped = ratio_clipped * adv

    # GRPO-Clip loss: -min(unclipped, clipped)
    min_obj = torch.minimum(pg_unclipped, pg_clipped)
    loss = -min_obj

    # metadata
    clipped_mask = (pg_clipped < pg_unclipped)

    metadata = {
        "ratio": ratio.detach(),
        "ratio_clipped": ratio_clipped.detach(),
        "clipped_mask": clipped_mask.detach(),
        "clipfrac": clipped_mask.float().mean().detach(),
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert policy_log_probs.ndim == 2, \
        f"expected policy_log_probs shape (B,T), got {tuple(policy_log_probs.shape)}"

    metadata: dict[str, torch.Tensor] = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for loss_type='no_baseline'"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        metadata["loss_type"] = torch.tensor(0)

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages is required for loss_type='reinforce_with_baseline'"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        metadata["loss_type"] = torch.tensor(1)

    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages is required for loss_type='grpo_clip'"
        assert old_log_probs is not None, "old_log_probs is required for loss_type='grpo_clip'"
        assert cliprange is not None, "cliprange is required for loss_type='grpo_clip'"
        loss, submeta = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=float(cliprange),
        )
        metadata.update(submeta)
        metadata["loss_type"] = torch.tensor(2)

    elif loss_type == "grpo_no_clip":
        assert advantages is not None
        assert old_log_probs is not None
        loss, submeta = compute_grpo_no_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
        )
        metadata.update(submeta) 

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata            


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Mean over elements where mask == 1.

    Args:
        tensor: any shape
        mask: same shape as tensor, 0/1 or bool
        dim: dimension to reduce; if None, reduce over all masked elements

    Returns:
        masked mean with the same semantics as tensor.mean(dim=dim)
    """
    assert tensor.shape == mask.shape, (
        f"tensor and mask must have the same shape, got {tuple(tensor.shape)} vs {tuple(mask.shape)}"
    )

    # Convert mask to some dtype/device for arithmetic
    m = mask.to(dtype=tensor.dtype, device=tensor.device)

    num = (tensor * m).sum(dim=dim)
    denom = m.sum(dim=dim)

    return num / denom


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Performs forward+backward for one microbatch.

    Returns:
        loss: scalar tensor (already scaled by grad_acc_steps) for logging
        metadata: dict[str, torch.Tensor]
    """
    assert policy_log_probs.ndim == 2, f"policy_log_probs must be (B,T), got {tuple(policy_log_probs.shape)}"
    assert response_mask.shape == policy_log_probs.shape, (
        f"response_mask must match policy_log_probs shape, got {tuple(response_mask.shape)}"
    )
    assert isinstance(gradient_accumulation_steps, int) and gradient_accumulation_steps >= 1, \
        "gradient_accumulation_steps must be >= 1"
    
    # 1) per-token PG loss (B,T) + metadata
    token_loss, meta = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # 2) per-sample loss: masked mean over sequence dim -> (B,)
    per_sample_loss = masked_mean(
        tensor=token_loss,
        mask=response_mask,
        dim=1,
    )

    # 3) batch mean -> scalar
    batch_loss = per_sample_loss.mean()

    # 4) scale for gradient accumulation
    scaled_loss = batch_loss / float(gradient_accumulation_steps)

    # 5) backward
    scaled_loss.backward()

    # metadata
    meta = dict(meta)
    meta["batch_loss_unscaled"] = batch_loss.detach()
    meta["batch_loss_scaled"] = scaled_loss.detach()
    return scaled_loss.detach(), meta    