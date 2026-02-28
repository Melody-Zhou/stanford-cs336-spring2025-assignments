from typing import Dict, List, Tuple, Any, Callable

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch.nn.functional as F


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Tensor]:
    """
    Tokenize prompt and output separately, concatenate them, then build:
      - input_ids: concat[:-1]
      - labels:    concat[1:]
      - response_mask: 1 on label positions corresponding to output tokens, else 0
    Shapes: (batch_size, max_len - 1)
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError(
            f"prompt_strs and output_strs must have same length, got {len(prompt_strs)} vs {len(output_strs)}"
        )
    
    batch_size = len(prompt_strs)
    if batch_size == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return {"input_ids": empty, "labels": empty, "response_mask": empty}
    
    # pad id fallback
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # tokenize separately then concat (no special tokens)
    prompt_ids_list: List[List[int]] = []
    output_ids_list: List[List[int]] = []
    concat_ids_list: List[List[int]] = []
    prompt_lens: List[int] = []
    output_lens: List[int] = []

    for p, o in zip(prompt_strs, output_strs):
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        o_ids = tokenizer(o, add_special_tokens=False).input_ids
        prompt_ids_list.append(list(p_ids))
        output_ids_list.append(list(o_ids))
        prompt_lens.append(len(p_ids))
        output_lens.append(len(o_ids))
        concat_ids_list.append(list(p_ids) + list(o_ids))

    max_len = max(len(x) for x in concat_ids_list)  # length of full concat (prompt+output)

    # full concatenated and padded (batch, max_len)
    full = torch.full((batch_size, max_len), pad_id, dtype=torch.long)

    # response_mask aligns with labels (= full[:, 1:]) => (batch, max_len - 1)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.long)

    for i, ids in enumerate(concat_ids_list):
        L = len(ids)
        full[i, :L] = torch.tensor(ids, dtype=torch.long)

        P = prompt_lens[i]
        O = output_lens[i]

        # output tokens are full indices: [P, P+O-1]
        # labels are full shifted left => label indices: [P-1, P+O-2] (length O)
        if O > 0:
            start = max(P - 1, 0)
            end = min(start + O, max_len - 1)  # cap to mask length
            response_mask[i, start:end] = 1

    input_ids = full[:, :-1].contiguous()
    labels = full[:, 1:].contiguous()

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy of next-token distribution (over vocab dim).

    Args:
        logits: (batch_size, sequence_length, vocab_size)

    Returns:
        entropies: (batch_size, sequence_length)
    """
    # if logits.ndim != 3:
    #     raise ValueError(f"logits must have shape (B, T, V), got {tuple(logits.shape)}")
    if logits.ndim < 1:
        raise ValueError(f"logits must have at least 1 dim, got {tuple(logits.shape)}")

    # log_probs = logits - logsumexp(logits)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)   # (B, T, 1)
    log_probs = logits - log_z                              # (B, T, V)
    probs = torch.exp(log_probs)                            # (B, T, V)

    # H(p) = - sum_v p(v) * log p(v)
    entropy = -(probs * log_probs).sum(dim=-1)              # (B, T)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-token conditional log-probabilities for a causal LM.

    Args:
        model: HF causal LM, already on correct device.
        input_ids: (B, T)
        labels:    (B, T) token ids
        return_token_entropy: if True, also return per-token entropy (B, T) computed from logits.

    Returns:
        dict with:
          - "log_probs": (B, T)
          - "token_entropy": (B, T) if requested
    """
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError(
            f"input_ids and labels must be (B, T). Got {tuple(input_ids.shape)} and {tuple(labels.shape)}"
        )
    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids and labels must have same shape. Got {tuple(input_ids.shape)} vs {tuple(labels.shape)}"
        )

    # forward
    logits = model(input_ids=input_ids).logits  # (B, T, V)

    # stable log-probs over vocab
    log_probs_vocab = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # gather log p(label_t | prefix) for each position
    # labels: (B, T) -> (B, T, 1) for gather
    gathered = torch.gather(log_probs_vocab, dim=-1, index=labels.unsqueeze(-1))  # (B, T, 1)
    token_log_probs = gathered.squeeze(-1)  # (B, T)

    out: Dict[str, torch.Tensor] = {"log_probs": token_log_probs}

    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)  # (B, T)

    return out


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None
) -> torch.Tensor:
    """
    Sum tensor over dim using mask (mask==1 contributes),
    then divide by normalize_constant.
    """
    if tensor.shape != mask.shape:
        raise ValueError(
            f"tensor and mask must have same shape, got {tensor.shape} vs {mask.shape}"
        )

    # convert mask to same dtype as tensor for multiplication
    mask = mask.to(dtype=tensor.dtype)

    masked_tensor = tensor * mask

    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)

    return summed / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Perform one SFT microbatch train step:
      - compute masked negative log-likelihood
      - normalize by normalize_constant
      - scale by gradient_accumulation_steps
      - call backward()

    Args:
        policy_log_probs: (B, T) log p(x_t | x_<t)
        response_mask:    (B, T) 1 for response tokens else 0
        gradient_accumulation_steps: number of microbatches per optimizer step
        normalize_constant: divisor for masked sum before grad-acc scaling

    Returns:
        loss: scalar tensor (already scaled for grad accumulation)
        metadata: dict of useful stats (unscaled loss, token counts, etc.)
    """
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"policy_log_probs and response_mask must have same shape, "
            f"got {tuple(policy_log_probs.shape)} vs {tuple(response_mask.shape)}"
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    # masked sum of negative log-probs, then divide by normalize_constant
    nll = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=1,
    ).mean() # scalar

    # scale for gradient accumulation
    loss = nll / float(gradient_accumulation_steps)

    # backward pass
    loss.backward()

    # some lightweight logging stats
    with torch.no_grad():
        # number of response tokens in this microbatch
        resp_tokens = response_mask.to(policy_log_probs.dtype).sum()
        metadata = {
            "nll": nll.detach(),
            "loss_unscaled": nll.detach(),  # alias (sometimes handy)
            "response_tokens": resp_tokens.detach(),
            "mean_log_prob_on_response": (
                (policy_log_probs * response_mask.to(policy_log_probs.dtype)).sum()
                / torch.clamp(resp_tokens, min=1.0)
            ).detach(),
        }

    return loss, metadata


@torch.no_grad()
def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool | None = None,
    num_log: int = 8,
    step: int | None = None,
    stop_str: str | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    Generate responses for a few prompts and log:
      - prompt / response / ground_truth
      - reward: format_reward, answer_reward, reward
      - avg token entropy over generated tokens
      - length stats (avg, correct avg, wrong avg)
    
    Returns a dict with:
      - "samples": list[dict]
      - "stats": dict
    """
    assert len(prompts) == len(ground_truths), "prompts and ground_truths must align"

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    n = min(num_log, len(prompts))
    prompts = prompts[:n]
    ground_truths = ground_truths[:n]

    # decide sampling
    if do_sample is None:
        do_sample = temperature > 0

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # batch tokenize to get attention mask
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)    

    # generate in one batch
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    sequences = gen_out.sequences  # (B, T_total)
    prompt_lens = attention_mask.sum(dim=1).tolist()

    samples: list[dict[str, Any]] = []
    lengths: list[int] = []
    lengths_correct: list[int] = []
    lengths_wrong: list[int] = []
    entropies: list[float] = []

    # Compute per-step entropies from scores
    avg_ent_per_sample = [0.0 for _ in range(n)]
    if gen_out.scores is not None and len(gen_out.scores) > 0:
        # accumulate entropies per step per sample
        acc = [0.0 for _ in range(n)]
        for step_logits in gen_out.scores:
            for i in range(n):
                acc[i] += float(compute_entropy(step_logits[i]).item())
        denom = float(len(gen_out.scores))
        avg_ent_per_sample = [x / denom for x in acc]

    for i in range(n):
        prompt = prompts[i]
        gt = ground_truths[i]
        pl = int(prompt_lens[i])

        full_ids = sequences[i]
        gen_ids = full_ids[pl:]  # generated part

        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        if stop_str is not None and stop_str in response_text:
            response_text = response_text.split(stop_str)[0] + stop_str

        reward_dict = reward_fn(response_text, gt)

        gen_len = int(gen_ids.numel())
        avg_ent = float(avg_ent_per_sample[i])

        samples.append(
            {
                "step": step,
                "prompt": prompt,
                "response": response_text,
                "ground_truth": gt,
                "reward": float(reward_dict.get("reward", 0.0)),
                "format_reward": float(reward_dict.get("format_reward", 0.0)),
                "answer_reward": float(reward_dict.get("answer_reward", 0.0)),
                "avg_token_entropy": avg_ent,
                "response_len": gen_len,
            }
        )

        lengths.append(gen_len)
        entropies.append(avg_ent)
        is_correct = float(reward_dict.get("answer_reward", 0.0)) >= 1.0
        if is_correct:
            lengths_correct.append(gen_len)
        else:
            lengths_wrong.append(gen_len)

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    stats = {
        "step": step,
        "avg_response_len": _mean([float(x) for x in lengths]),
        "avg_response_len_correct": _mean([float(x) for x in lengths_correct]),
        "avg_response_len_wrong": _mean([float(x) for x in lengths_wrong]),
        "avg_token_entropy": _mean(entropies),
        "avg_reward": _mean([s["reward"] for s in samples]),
        "avg_format_reward": _mean([s["format_reward"] for s in samples]),
        "avg_answer_reward": _mean([s["answer_reward"] for s in samples]),
        "n_logged": len(samples),
    }

    return {"samples": samples, "stats": stats}