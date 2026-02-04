import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class RunRow:
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    learning_rate: float
    train_flops: int
    loss: float


def approx_nonemb_params(d_model: int, num_layers: int) -> float:
    # Handout tip: non-embedding params ≈ 12 * n_layer * d_model^2
    return 12.0 * num_layers * (d_model ** 2)


def load_sweep_jsonl(path: str | Path) -> List[RunRow]:
    path = Path(path)
    rows: List[RunRow] = []
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("status") != "ok":
                continue
            q = obj.get("query", {})
            resp = obj.get("response", {})
            if "loss" not in resp:
                continue
            rows.append(
                RunRow(
                    d_model=int(q["d_model"]),
                    num_layers=int(q["num_layers"]),
                    num_heads=int(q["num_heads"]),
                    batch_size=int(q["batch_size"]),
                    learning_rate=float(q["learning_rate"]),
                    train_flops=int(q["train_flops"]),
                    loss=float(resp["loss"]),
                )
            )


def group_best_by_compute(rows: Iterable[RunRow]) -> Dict[int, RunRow]:
    best: Dict[int, RunRow] = {}
    for r in rows:
        C = r.train_flops
        if (C not in best) or (r.loss < best[C].loss):
            best[C] = r
    return best
