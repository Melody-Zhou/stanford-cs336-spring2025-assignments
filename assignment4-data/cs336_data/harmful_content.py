import re
from functools import lru_cache
from typing import Tuple

import fasttext

# Model paths
_NSFw_MODEL_PATH = "data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin"
_TOXIC_MODEL_PATH = "data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin"


def _to_single_line(text: str) -> str:
    """
    fastText's Python binding expects a single line per prediction.
    Replace newlines/tabs with spaces and collapse repeated whitespace.
    """
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=1)
def _get_nsfw_model():
    """Cache the NSFW model so we don't reload it on every call."""
    return fasttext.load_model(_NSFw_MODEL_PATH)


@lru_cache(maxsize=1)
def _get_toxic_model():
    """Cache the toxic/hate-speech model so we don't reload it on every call."""
    return fasttext.load_model(_TOXIC_MODEL_PATH)


def _strip_label(prefix_label: str) -> str:
    """Remove fastText label prefix."""
    return prefix_label.replace("__label__", "")


def _map_binary_label(raw: str, positive: str, negative: str) -> str:
    """
    Map fastText binary labels to expected strings.

    Handles common fastText conventions:
    - __label__1 / __label__0
    - __label__pos / __label__neg
    - already-correct labels (e.g., nsfw / non-nsfw)
    """
    lab = _strip_label(raw).lower()

    # If the model already outputs human-readable labels.
    if lab in {positive, negative}:
        return lab

    # Common numeric binary labels.
    if lab == "1":
        return positive
    if lab == "0":
        return negative

    # Heuristic: if label string contains the positive keyword.
    if positive.replace("-", "") in lab.replace("-", ""):
        return positive
    if negative.replace("-", "") in lab.replace("-", ""):
        return negative

    # Fallback: treat unknown as negative (conservative for filtering).
    return negative


def classify_nsfw(text: str) -> Tuple[str, float]:
    """
    Classify whether the input text is NSFW.

    Returns:
      (label, confidence) where label is "nsfw" or "non-nsfw",
      and confidence is in [0, 1].
    """
    if not text:
        return "non-nsfw", 0.0

    model = _get_nsfw_model()
    text = _to_single_line(text)

    labels, probs = model.predict(text, k=1)
    raw_label = labels[0]
    score = float(probs[0])

    label = _map_binary_label(raw_label, positive="nsfw", negative="non-nsfw")
    score = max(0.0, min(1.0, score))
    return label, score


def classify_toxic_speech(text: str) -> Tuple[str, float]:
    """
    Classify whether the input text is toxic speech.

    Returns:
      (label, confidence) where label is "toxic" or "non-toxic",
      and confidence is in [0, 1].
    """
    if not text:
        return "non-toxic", 0.0

    model = _get_toxic_model()
    text = _to_single_line(text)

    labels, probs = model.predict(text, k=1)
    raw_label = labels[0]
    score = float(probs[0])

    label = _map_binary_label(raw_label, positive="toxic", negative="non-toxic")
    score = max(0.0, min(1.0, score))
    return label, score
