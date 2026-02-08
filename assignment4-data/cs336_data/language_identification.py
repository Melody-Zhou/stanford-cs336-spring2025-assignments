import re
from functools import lru_cache

import fasttext

_MODEL_PATH = "data/classifiers/lid.176.bin"


@lru_cache(maxsize=1)
def _get_model():
    # Cache the model so we don't reload it on every call.
    return fasttext.load_model(_MODEL_PATH)


def _to_single_line(text: str) -> str:
    # fastText Python binding requires a single line per prediction.
    # Replace newlines/tabs with spaces and collapse repeated whitespace.
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def identify_language(text: str) -> tuple[str, float]:
    """
    Identify the main language of a Unicode string using fastText lid.176

    Returns:
      (language_id, confidence) where confidence is in [0, 1]
    """
    if not text:
        return "unknown", 0.0
    
    model = _get_model()
    text = _to_single_line(text)

    labels, probs = model.predict(text, k=1)
    lang = labels[0].replace("__label__", "")
    score = float(probs[0])

    if lang.startswith("zh"):
        lang = "zh"

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))
    return lang, score