import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


@dataclass
class EvalRow:
    idx: int
    problem_id: Optional[str]
    prompt: str
    ground_truth: Any
    response: str
    reward: float
    format_reward: float
    answer_reward: float
    category: str  # "F1A1", "F1A0", "F0A0"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_r1_zero_prompt(prompt_template: str, question: str) -> str:
    # prompt_template should contain "{question}"
    return prompt_template.format(question=question)


def categorize(format_reward: float, answer_reward: float) -> str:
    if format_reward == 1.0 and answer_reward == 1.0:
        return "F1A1"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "F1A0"
    return "F0A0"


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, Any], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[Any],
    eval_sampling_params: SamplingParams,
    request_batch_size: int = 64,
) -> List[EvalRow]:
    """
    Evaluate a language model on a list of prompts, compute rewards, and return per-example rows.
    """
    assert len(prompts) == len(ground_truths)

    rows: List[EvalRow] = []
    idx_base = 0

    # vLLM can take list prompts directly
    for start in range(0, len(prompts), request_batch_size):
        end = min(len(prompts), start + request_batch_size)
        batch_prompts = prompts[start:end]
        batch_gts = ground_truths[start:end]

        outputs = vllm_model.generate(batch_prompts, eval_sampling_params)

        # outputs aligns with input prompts order
        for i, out in enumerate(outputs):
            prompt = out.prompt
            text = out.outputs[0].text  # generated continuation
            full_response = text

            gt = batch_gts[i]
            scores = reward_fn(full_response, gt)  # dict with reward/format_reward/answer_reward

            fr = float(scores.get("format_reward", 0.0))
            ar = float(scores.get("answer_reward", 0.0))
            rr = float(scores.get("reward", 0.0))
            cat = categorize(fr, ar)

            rows.append(
                EvalRow(
                    idx=idx_base + i,
                    problem_id=None,
                    prompt=prompt,
                    ground_truth=gt,
                    response=full_response,
                    reward=rr,
                    format_reward=fr,
                    answer_reward=ar,
                    category=cat
                )
            )
        
        idx_base += (end - start)

    return rows


def write_jsonl(path: str, rows: List[EvalRow]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def summarize(rows: List[EvalRow]) -> Dict[str, Any]:
    n = len(rows)
    c = {"F1A1": 0, "F1A0": 0, "F0A0": 0}
    for r in rows:
        c[r.category] += 1

    format_rate = sum(r.format_reward for r in rows) / n if n else 0.0
    acc = sum(r.answer_reward for r in rows) / n if n else 0.0
    avg_reward = sum(r.reward for r in rows) / n if n else 0.0

    return {
        "n": n,
        "counts": c,
        "format_rate": format_rate,
        "answer_accuracy": acc,
        "avg_reward": avg_reward,
    }


def sample_examples(rows: List[EvalRow], category: str, k: int = 10) -> List[EvalRow]:
    picked = [r for r in rows if r.category == category]
    return picked[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--data", default="data/MATH/validation.jsonl")
    ap.add_argument("--prompt_file", default="cs336_alignment/prompts/r1_zero.prompt")
    ap.add_argument("--out_dir", default="runs/math_baseline")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=int, default=1.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    prompt_template = read_text(args.prompt_file)

    examples = load_jsonl(args.data)
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    # MATH jsonl usually has "problem" / "question" and "answer" or similar
    def get_question(ex: Dict[str, Any]) -> str:
        for k in ["problem", "question", "prompt"]:
            if k in ex and isinstance(ex[k], str):
                return ex[k]
        raise KeyError(f"Cannot find question field in example keys={list(ex.keys())}")

    def get_ground_truth(ex: Dict[str, Any]) -> Any:
        for k in ["answer", "ground_truth", "target"]:
            if k in ex:
                return ex[k]
        raise KeyError(f"Cannot find answer field in example keys={list(ex.keys())}")

    prompts, gts = [], []
    for ex in examples:
        q = get_question(ex)
        gt = get_ground_truth(ex)
        prompts.append(format_r1_zero_prompt(prompt_template, q))
        gts.append(gt)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
    )

    rows = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=gts,
        eval_sampling_params=sampling_params,
        request_batch_size=args.batch_size,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # full per-example
    write_jsonl(str(out_dir / "predications.jsonl"), rows)

    # summary json
    summary = summarize(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # samples for write-up
    samples = {
        "F1A1": [asdict(r) for r in sample_examples(rows, "F1A1", 10)],
        "F1A0": [asdict(r) for r in sample_examples(rows, "F1A0", 10)],
        "F0A0": [asdict(r) for r in sample_examples(rows, "F0A0", 10)],
    }
    with open(out_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print("Saved to:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()