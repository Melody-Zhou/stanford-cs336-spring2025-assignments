import os
import re
from functools import lru_cache
from typing import Tuple

import fasttext

# Default model path
_MODEL_PATH = os.environ.get("CS336_QUALITY_MODEL_PATH", "runs/quality_fasttext.bin")


@lru_cache(maxsize=1)
def _get_model():
    """Cache the model so we don't reload it on every call."""
    return fasttext.load_model(_MODEL_PATH)


def _to_single_line(text: str) -> str:
    """fastText expects a single line per prediction."""
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def quality_score(text: str) -> float:
    """
    Returns a numeric quality score in [0, 1], defined as P(__label__hq | text).
    """
    if not text:
        return 0.0
    
    model = _get_model()
    text = _to_single_line(text)

    labels, probs = model.predict(text, k=2)
    score_map = {lab.replace("__label__", ""): float(p) for lab, p in zip(labels, probs)}

    # In training we used labels: __label__hq / __label__lq
    score = score_map.get("hq", 0.0)
    return max(0.0, min(1.0, score))


def classify_quality(text: str, threshold: float = 0.5) -> Tuple[str, float]:
    """
    Classify text as high-quality(wiki) vs non-high-quality(cc) using the trained fastText model.

    Returns:
      (label, confidence)
    where confidence is the model score P(hq|text) in [0, 1].

    Note: threshold defaults to 0.5 for a sanity-check classifier.
    """
    score = quality_score(text)

    label = "wiki" if score >= threshold else "cc"

    return label, score