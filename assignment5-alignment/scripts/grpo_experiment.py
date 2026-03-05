import json
import os
import time
import random
from pathlib import Path
from typing import Any
from enum import Enum

import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step, masked_mean


app = typer.Typer()


class LossType(str, Enum):
    no_baseline = "no_baseline"
    reinforce_with_baseline = "reinforce_with_baseline"
    grpo_clip = "grpo_clip"
    grpo_no_clip = "grpo_no_clip"


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_path = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_path, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def load_math_jsonl(path: str, limit: int = 0, seed: int = 0) -> list[dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    if limit and limit > 0:
        rnd = random.Random(seed)
        rnd.shuffle(data)
        data = data[:limit]
    return data


def build_prompts_and_gts(examples: list[dict[str, Any]], prompt_file: str) -> tuple[list[str], list[str]]:
    template = Path(prompt_file).read_text(encoding="utf-8")
    prompts, gts = [], []
    dropped = 0

    for ex in examples:
        q = ex.get("problem") or ex.get("question") or ex.get("prompt")
        gt = ex.get("answer") or ex.get("ground_truth") or ex.get("target")
        if q is None or gt is None:
            dropped += 1
            continue
        
        if isinstance(gt, str) and gt.strip() == "":
            dropped += 1
            continue
        prompts.append(template.format(question=q))
        gts.append(gt)

    if len(prompts) == 0:
        raise RuntimeError("No valid examples found after filtering. Check dataset format.")

    if dropped > 0:
        print(f"[build_prompts_and_gts] dropped {dropped} examples with missing/empty fields")

    return prompts, gts


@torch.inference_mode()
def eval_rewards_with_vllm(
    llm: LLM,
    val_prompts: list[str],
    val_gts: list[str],
    reward_fn,
    max_examples: int,
    temperature: float,
    min_tokens: int,
    max_tokens: int,
    stop: list[str],
) -> dict[str, float]:
    prompts = val_prompts[:max_examples]
    gts = val_gts[:max_examples]

    sp = SamplingParams(
        temperature=temperature,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        stop=stop,
    )
    outs = llm.generate(prompts, sp)
    # outs[i].outputs[0].text
    total, fmt, ans = 0.0, 0.0, 0.0
    for o, gt in zip(outs, gts):
        resp = o.outputs[0].text
        r = reward_fn(resp, gt)
        total += float(r["reward"])
        fmt += float(r["format_reward"])
        ans += float(r["answer_reward"])
    n = len(prompts)
    return {
        "val_reward": total / n,
        "val_format_reward": fmt / n,
        "val_answer_reward": ans / n,
    }


@app.command()
def main(
    model_id: str = "data/models/Qwen2.5-Math-1.5B",
    train_path: str = "data/MATH/train.jsonl",
    val_path: str = "data/MATH/validation.jsonl",
    prompt_file: str = "cs336_alignment/prompts/r1_zero.prompt",
    seed: int = 0,

    # ===== GRPO hypers =====
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: LossType = LossType.reinforce_with_baseline,
    use_std_normalization: bool = True,
    cliprange: float = 0.2,

    # ===== logging / eval =====
    eval_interval: int = 10,
    eval_max_examples: int = 1024,
    log_dir: str = "runs/grpo",
    save_interval: int = 50,
    stop_at: str = "</answer>",
):
    torch.manual_seed(seed)
    random.seed(seed)

    # -------- sanity checks (handout suggested) --------
    assert train_batch_size % gradient_accumulation_steps == 0, \
        "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps

    assert rollout_batch_size % group_size == 0, \
        "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    assert train_batch_size >= group_size, \
        "train_batch_size must be >= group_size"

    assert rollout_batch_size % train_batch_size == 0, \
        "For simplicity, require rollout_batch_size divisible by train_batch_size"
    n_optimizer_updates_per_epoch = rollout_batch_size // train_batch_size

    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / "train_log.jsonl"

    # -------- load data --------
    train_examples = load_math_jsonl(train_path, limit=0, seed=seed)
    val_examples = load_math_jsonl(val_path, limit=0, seed=seed)
    train_prompts, train_gts = build_prompts_and_gts(train_examples, prompt_file)
    val_prompts, val_gts = build_prompts_and_gts(val_examples, prompt_file)

    # -------- init models --------
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()
    policy.train()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))

    llm = init_vllm(model_id=model_id, device="cuda:1", seed=seed, gpu_memory_utilization=gpu_memory_utilization)

    load_policy_into_vllm_instance(policy, llm)

    # -------- training loop --------
    global_step = 0
    t0 = time.time()
    loss_type = loss_type.value

    for grpo_step in range(n_grpo_steps):
        # ========== 1) sample prompts ==========
        # on-policy: rollout is regenerated at every step
        idxs = [random.randrange(0, len(train_prompts)) for _ in range(n_prompts_per_rollout_batch)]
        batch_prompts = [train_prompts[i] for i in idxs]
        batch_gts_prompt = [train_gts[i] for i in idxs]

        # ========== 2) rollout via vLLM ==========
        sp = SamplingParams(
            temperature=sampling_temperature,
            min_tokens=sampling_min_tokens,
            max_tokens=sampling_max_tokens,
            n=group_size,
            stop=[stop_at],
        )
        outs = llm.generate(batch_prompts, sp)

        rollout_prompts = []
        rollout_responses = []
        repeated_gts = []
        for p, gt, out in zip(batch_prompts, batch_gts_prompt, outs):
            # out.outputs is a list with length group_size
            for o in out.outputs:
                rollout_prompts.append(p)
                rollout_responses.append(o.text)
                repeated_gts.append(gt)

        assert len(rollout_responses) == rollout_batch_size

        # ========== 3) compute group-normalized rewards (advantages) ==========
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        # ========== 4) tokenize prompt+response for scoring ==========
        toks = tokenize_prompt_and_output(
            prompt_strs=rollout_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        # toks: input_ids, labels, response_mask
        input_ids = toks["input_ids"].cuda()
        labels = toks["labels"].cuda()
        response_mask = toks["response_mask"].cuda()

        # ========== 5) (optional) old_log_probs for off-policy grpo_clip ==========
        old_log_probs = None
        if loss_type == "grpo_clip" or epochs_per_rollout_batch > 1 or train_batch_size != rollout_batch_size:
            # typical off-policy case: multiple epochs / multiple updates
            with torch.inference_mode():
                scored_old = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                old_log_probs = scored_old["log_probs"].detach()  # (B, T)
                # disable gradients for old policy logprobs
                old_log_probs.requires_grad_(False)
        
        # ========== 6) gradient updates on this rollout batch ==========
        perm = torch.randperm(rollout_batch_size, device=input_ids.device)

        # move reward/advantage to GPU and reorder according to perm
        advantages_gpu = advantages.cuda()[perm].unsqueeze(-1)      # (B, 1)
        raw_rewards_gpu = raw_rewards.cuda()[perm].unsqueeze(-1)    # (B, 1)
        input_ids = input_ids[perm]
        labels = labels[perm]
        response_mask = response_mask[perm]
        if old_log_probs is not None:
            old_log_probs = old_log_probs.cuda()[perm]

        # actual optimization
        policy.train()
        for epoch in range(epochs_per_rollout_batch):
            for upd in range(n_optimizer_updates_per_epoch):
                start = upd * train_batch_size
                end = start + train_batch_size

                mb_input_ids = input_ids[start:end]
                mb_labels = labels[start:end]
                mb_mask = response_mask[start:end]
                mb_adv = advantages_gpu[start:end]
                mb_raw = raw_rewards_gpu[start:end]
                mb_old = old_log_probs[start:end] if old_log_probs is not None else None

                # one train_batch is split into gradient_accumulation_steps microbatches
                optimizer.zero_grad(set_to_none=True)

                # used for logging
                loss_accum = 0.0
                entropies = []

                for k in range(gradient_accumulation_steps):
                    ms = k * micro_train_batch_size
                    me = ms + micro_train_batch_size

                    micro_input_ids = mb_input_ids[ms:me]
                    micro_labels = mb_labels[ms:me]
                    micro_mask = mb_mask[ms:me]
                    micro_adv = mb_adv[ms:me]
                    micro_raw = mb_raw[ms:me]
                    micro_old = mb_old[ms:me] if mb_old is not None else None

                    scored = get_response_log_probs(
                        model=policy,
                        input_ids=micro_input_ids,
                        labels=micro_labels,
                        return_token_entropy=True,
                    )
                    policy_log_probs = scored["log_probs"]          # (microB, T)
                    token_entropy = scored["token_entropy"]         # (microB, T)

                    # training step (includes masked_mean + /grad_acc_steps scaling + backward)
                    micro_loss, meta = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=micro_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=micro_raw if loss_type == "no_baseline" else None,
                        advantages=micro_adv if loss_type != "no_baseline" else None,
                        old_log_probs=micro_old if loss_type == "grpo_clip" else None,
                        cliprange=cliprange if loss_type == "grpo_clip" else None,
                    )

                    loss_accum += float(micro_loss.detach().cpu())
                    # average token entropy (only over response tokens)
                    ent = masked_mean(token_entropy.detach(), micro_mask, dim=None)
                    entropies.append(float(ent.cpu()))

                # gradient clipping + step
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                global_step += 1

                # ===== logging =====
                log_obj = {
                    "grpo_step": grpo_step,
                    "global_step": global_step,
                    "epoch": epoch,
                    "update_in_epoch": upd,
                    "loss": loss_accum,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "train_reward": float(mb_raw.mean().detach().cpu()),
                    "train_adv": float(mb_adv.mean().detach().cpu()),
                    "token_entropy": sum(entropies) / max(1, len(entropies)),
                    "wall_time_sec": time.time() - t0,
                }
                # clip fraction (if recorded in grpo_clip metadata)
                if "clip_fraction" in meta:
                    log_obj["clip_fraction"] = float(meta["clip_fraction"].detach().cpu())

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_obj) + "\n")                

                # ===== periodic eval =====
                if global_step % eval_interval == 0:
                    # sync latest policy into vLLM before evaluation
                    load_policy_into_vllm_instance(policy, llm)
                    val_metrics = eval_rewards_with_vllm(
                        llm=llm,
                        val_prompts=val_prompts,
                        val_gts=val_gts,
                        reward_fn=r1_zero_reward_fn,
                        max_examples=eval_max_examples,
                        temperature=0.0,          # greedy decoding for eval
                        min_tokens=sampling_min_tokens,
                        max_tokens=sampling_max_tokens,
                        stop=[stop_at],
                    )
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"global_step": global_step, **val_metrics}) + "\n")

                # ===== periodic save =====
                if global_step % save_interval == 0:
                    save_dir = Path(log_dir) / f"ckpt_step_{global_step}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)

        # before next rollout: synchronize latest policy weights into vLLM
        load_policy_into_vllm_instance(policy, llm)


if __name__ == "__main__":
    app()