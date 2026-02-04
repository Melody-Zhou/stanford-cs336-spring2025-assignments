import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_key(endpoint: str, params: Dict[str, Any]) -> str:
    payload = {"endpoint": endpoint, "params": params}
    h = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return h


@dataclass
class CacheHit:
    key: str
    value: Dict[str, Any]


class JsonlCache:
    """
    Append-only JSONL cache.
    Each line: {"key":..., "endpoint":..., "params":..., "response":...}
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = obj.get("key")
                if key:
                    self._index[key] = obj
    
    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[CacheHit]:
        key = make_key(endpoint, params)
        obj = self._index.get(key)
        if obj is None:
            return None
        return CacheHit(key=key, value=obj["response"])
    
    def put(self, endpoint: str, params: Dict[str, Any], response: Dict[str, Any]) -> str:
        key = make_key(endpoint, params)
        record = {
            "key": key,
            "endpoint": endpoint,
            "params": params,
            "response": response,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(_stable_json(record) + "\n")
        self._index[key] = record
        return key
