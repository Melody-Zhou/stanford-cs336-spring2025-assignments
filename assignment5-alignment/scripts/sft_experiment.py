import argparse
import json
import time
from typing import Dict, Any
import os
import random
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_baseline import evaluate_vllm

from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step, log_generations

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed


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


class SFTDataset(Dataset):
    def __init__(self, path: str, limit: int = 0, seed: int = 0):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        if limit and limit > 0:
            rnd = random.Random(seed)
            rnd.shuffle(self.data)
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return ex["prompt"], ex["response"], ex
    

def collate_fn(batch, tokenizer):
    prompts = [x[0] for x in batch]
    outputs = [x[1] for x in batch]

    toks = tokenize_prompt_and_output(prompts, outputs, tokenizer)
    return toks


def build_math_val_prompts_and_gts(val_path: str, prompt_file: str, max_examples: int = 0):
    prompt_template = Path(prompt_file).read_text(encoding="utf-8")

    val = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            val.append(json.loads(line))

    if max_examples and max_examples > 0:
        val = val[:max_examples]

    prompts, gts = [], []
    for ex in val:
        q = ex.get("problem") or ex.get("question") or ex.get("prompt")
        gt = ex.get("answer") or ex.get("ground_truth") or ex.get("target")
        if q is None or gt is None:
            raise KeyError(f"Validation example missing question/answer fields: keys={list(ex.keys())}")
        prompts.append(prompt_template.format(question=q))
        gts.append(gt)

    return prompts, gts    


def filter_correct_sft_samples(data_path: str, out_path: str):
    """
    Filtering: Only retain SFT samples that can produce the correct answer.
    """
    kept = []
    total = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            total += 1
            gt = ex.get("answer") or ex.get("ground_truth")
            if gt is None:
                raise RuntimeError(
                    "sft.jsonl does not contain ground-truth fields (answer/ground_truth). "
                )
            resp = ex["response"]
            scores = r1_zero_reward_fn(resp, gt)
            if float(scores.get("answer_reward", 0.0)) >= 1.0:
                kept.append(ex)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for ex in kept:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    return {"filtered/kept": len(kept), "filtered/total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="data/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--sft_path", default="data/MATH/sft.jsonl")
    ap.add_argument("--val_path", default="data/MATH/validation.jsonl")
    ap.add_argument("--prompt_file", default="cs336_alignment/prompts/r1_zero.prompt")

    ap.add_argument("--train_device", default="cuda:0")
    ap.add_argument("--vllm_device", default="cuda:1")

    ap.add_argument("--train_samples", type=int, default=0, help="0 means full dataset")
    ap.add_argument("--filter_correct", action="store_true")

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--micro_batch_size", type=int, default=2)
    ap.add_argument("--grad_acc_steps", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs/sft_experiment")
    ap.add_argument("--eval_max_examples", type=int, default=500)
    args = ap.parse_args()

    # logging
    run_dir = Path(args.out_dir) / f"samples{args.train_samples or 'full'}_{'filtered' if args.filter_correct else 'all'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    opt_step = 0  # counts optimizer updates
    step = 0
    micro_idx = 0    

    def log_event(event: Dict[str, Any], *, also_print: bool = True):
        """
        Append one json line to log.jsonl (and optionally print a readable message).
        """
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "time": time.time(),
            "step": step,
            "micro_idx": micro_idx,
            "opt_step": opt_step,
            **event,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()

        if also_print:
            # keep terminal readable
            if "msg" in event:
                print(event["msg"])
            else:
                print(payload)    

    # seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # tokenizer/model on train device
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.train_device)
    policy.train()

    # vLLM on eval device
    llm = init_vllm(args.model_id, device=args.vllm_device, seed=args.seed)

    eval_prompts, eval_gts = build_math_val_prompts_and_gts(
        args.val_path, args.prompt_file, max_examples=args.eval_max_examples
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # optionally filter dataset
    data_path = args.sft_path
    if args.filter_correct:
        filtered_path = str(Path(args.out_dir) / "filtered_sft.jsonl")
        stats = filter_correct_sft_samples(args.sft_path, filtered_path)
        log_event({"type": "filter_stats", "stats": stats, "msg": f"Filter stats: {stats}"})
        data_path = filtered_path

    dataset = SFTDataset(data_path, limit=args.train_samples, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        drop_last=True,
    )

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # training loop
    opt.zero_grad(set_to_none=True)

    for epoch in range(10_000_000):
        for batch in loader:
            step += 1
            micro_idx += 1

            input_ids = batch["input_ids"].to(args.train_device)
            labels = batch["labels"].to(args.train_device)
            response_mask = batch["response_mask"].to(args.train_device)

            # get per-token log_probs (B, T)
            out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)
            policy_log_probs = out["log_probs"]            

            # microbatch train step: does backward inside
            loss, meta = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.grad_acc_steps,
                normalize_constant=1.0,
            )

            # optimizer step each grad_acc_steps
            if micro_idx % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 10 == 0:
                    log_event({"type": "train_loss", "loss": float(loss.detach())}, also_print=False)

            # periodic eval
            if step % args.eval_interval == 0:
                policy.eval()
                with torch.no_grad():
                    load_policy_into_vllm_instance(policy, llm)
                    rows = evaluate_vllm(
                        vllm_model=llm,
                        reward_fn=r1_zero_reward_fn,
                        prompts=eval_prompts,
                        ground_truths=eval_gts,
                        eval_sampling_params=eval_sampling_params,
                        request_batch_size=64,
                    )

                # generation log records
                # gen_log = log_generations(
                #     model=policy,
                #     tokenizer=tokenizer,
                #     prompts=eval_prompts[:8],
                #     ground_truths=eval_gts[:8],
                #     reward_fn=r1_zero_reward_fn,
                #     num_log=8,
                #     step=step,
                #     stop_str="</answer>",
                #     max_new_tokens=512,
                #     temperature=0.0,
                # )

                # log_event({"type": "gen_stats", "gen_stats": gen_log["stats"], "msg": f"gen stats: {gen_log['stats']}"})

                n = len(rows)
                eval_acc = sum(r.answer_reward for r in rows) / n if n else 0.0
                eval_format = sum(r.format_reward for r in rows) / n if n else 0.0
                eval_reward = sum(r.reward for r in rows) / n if n else 0.0
                metrics = {
                    "eval/accuracy": eval_acc,
                    "eval/format_rate": eval_format,
                    "eval/avg_reward": eval_reward,
                    "eval/n": n,                    
                }
                log_event({"type": "eval_metrics", "loss": float(loss.detach()), "metrics": metrics,
                        "msg": f"[step={step}] loss={float(loss.detach()):.4f} {metrics}"})
                policy.train()

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    # save
    policy.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    log_event({"type": "save", "out_dir": str(run_dir), "msg": f"Saved: {run_dir}"})


if __name__ == "__main__":
    main()
