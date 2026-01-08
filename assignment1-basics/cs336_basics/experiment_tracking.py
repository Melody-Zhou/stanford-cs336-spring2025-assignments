import os
import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

class ExperimentTracker:
    """
    A lightweight experiment tracker that logs scalar metrics against:
      - global step (gradient update count)
      - wall-clock time (seconds since run start)

    Logs are written to:
      runs/<run_name>/metrics.jsonl
      runs/<run_name>/config.json    
    """

    def __init__(
        self,
        run_dir: str,
        config: Any,
        *,
        use_wandb: bool = False,
        wandb_project: str = "cs336-a1",
        wandb_run_name: Optional[str] = None
    ) -> None:
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.config_path = os.path.join(self.run_dir, "config.json")

        self.t0 = time.time()

        # Write config once for reproducibility
        cfg_obj = self._to_jsonable(config)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_obj, f, indent=2, sort_keys=True)

        self._wandb = None
        self._wandb_enabled = use_wandb
        if use_wandb:
            import wandb
            self._wandb = wandb
            self._wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=cfg_obj,
                dir=self.run_dir
            )

    def wall_time_s(self) -> float:
        return time.time() - self.t0

    def log(self, step: int, metrics: dict[str, float | int]) -> None:
        """
        Log a dictionary of scalar metrics at a given step.
        A wall_time_s field will be automatically added.
        """
        record = {"step": int(step), "wall_time_s": float(self.wall_time_s())}
        for k, v in metrics.items():
            record[k] = float(v) if isinstance(v, (int, float)) else v

        # Append to JSONL
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Optional wandb
        if self._wandb_enabled and self._wandb is not None:
            self._wandb.log(record, step=int(step))

    def close(self) -> None:
        if self._wandb_enabled and self._wandb is not None:
            self._wandb.finish()

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return {k: ExperimentTracker._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ExperimentTracker._to_jsonable(x) for x in obj]
        # Basic types
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        # Fallback
        return str(obj)