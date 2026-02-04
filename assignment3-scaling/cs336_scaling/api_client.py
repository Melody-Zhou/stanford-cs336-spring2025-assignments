from dataclasses import dataclass
from typing import Any, Dict

import requests

from cache import JsonlCache


@dataclass(frozen=True)
class LossQuery:
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    learning_rate: float
    train_flops: int


class ScalingAPIError(RuntimeError):
    pass


class ScalingAPIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://hyperturing.stanford.edu:8000",
        cache_path: str = "runs/api_cache.jsonl",
        timeout_s: int = 60,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache = JsonlCache(cache_path)
        self.timeout_s = timeout_s
    
    # -------------------------
    # Local validation (matches the handout)
    # -------------------------
    def _validate_loss_query(self, q: LossQuery) -> None:
        # Ranges from the handout: d_model[64,1024], layers[2,24], heads[2,16],
        # batch_size[128,256], lr[1e-4,1e-3], train_flops in a fixed set. :contentReference[oaicite:5]{index=5}
        if not (64 <= q.d_model <= 1024):
            raise ValueError(f"d_model out of range: {q.d_model}")
        if not (2 <= q.num_layers <= 24):
            raise ValueError(f"num_layers out of range: {q.num_layers}")
        if not (2 <= q.num_heads <= 16):
            raise ValueError(f"num_heads out of range: {q.num_heads}")
        if not (128 <= q.batch_size <= 256):
            raise ValueError(f"batch_size out of range: {q.batch_size}")
        if not (1e-4 <= q.learning_rate <= 1e-3):
            raise ValueError(f"learning_rate out of range: {q.learning_rate}")

        allowed = {
            int(1e13), int(3e13), int(6e13),
            int(1e14), int(3e14), int(6e14),
            int(1e15), int(3e15), int(6e15),
            int(1e16), int(3e16), int(6e16),
            int(1e17), int(3e17), int(6e17),
            int(1e18),
        }
        if int(q.train_flops) not in allowed:
            raise ValueError(f"train_flops not allowed: {q.train_flops}")

    def _get_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        r = requests.get(url, params=params, timeout=self.timeout_s)
        # API error examples return {"message": "..."} :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
        try:
            payload = r.json()
        except Exception as e:
            raise ScalingAPIError(f"Non-JSON response: status={r.status_code}, text={r.text[:200]}") from e

        if r.status_code != 200:
            msg = payload.get("message", payload)
            raise ScalingAPIError(f"API error {r.status_code} @ {url}: {msg}")
        return payload

    # -------------------------
    # Public endpoints
    # -------------------------
    def total_flops_used(self) -> float:
        endpoint = "/total_flops_used"            
        params = {"api_key": self.api_key}
        hit = self.cache.get(endpoint, params)
        if hit:
            return float(hit.value)
        out = self._get_json(endpoint, params)
        # sample shows it returns a number (JSON scalar) :contentReference[oaicite:8]{index=8}
        self.cache.put(endpoint, params, out)
        return float(out)

    def previous_runs(self) -> Dict[str, Any]:
        endpoint = "/previous_runs"
        params = {"api_key": self.api_key}
        hit = self.cache.get(endpoint, params)
        if hit:
            return hit.value
        out = self._get_json(endpoint, params)
        self.cache.put(endpoint, params, out)
        return out

    def loss(self, q: LossQuery, use_cache: bool = True) -> Dict[str, Any]:
        self._validate_loss_query(q)

        endpoint = "/loss"
        params = {
            "d_model": q.d_model,
            "num_layers": q.num_layers,
            "num_heads": q.num_heads,
            "batch_size": q.batch_size,
            "learning_rate": q.learning_rate,
            "train_flops": int(q.train_flops),
            "api_key": self.api_key,            
        }

        if use_cache:
            hit = self.cache.get(endpoint, params)
            if hit:
                return hit.value
        
        out = self._get_json(endpoint, params)
        # example output: {"loss": ..., "total_flops_used": ...} :contentReference[oaicite:9]{index=9}
        self.cache.put(endpoint, params, out)
        return out
