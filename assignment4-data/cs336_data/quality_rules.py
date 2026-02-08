import re

# Tokenize into "words" using a simple regex, intentionally lightweight and does not require NLTK.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

# A word is considered to "contain a letter" if it has at least one ASCII letter.
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")

# Match lines ending with three dots "..." (ASCII ellipsis), ignoring trailing whitespace.
_ELLIPSIS_LINE_RE = re.compile(r"\.\.\.\s*$")


def gopher_quality_filter(text: str) -> bool:
    """
    Return True if `text` passes a subset of the Gopher quality rules.

    Filters out any document that satisfies ANY of the following:
      - word count < 50 OR > 100,000
      - average word length not in [3, 10]
      - more than 30% of lines end with "..."
      - fewer than 80% of words contain at least one letter
    """
    if not text:
        return False

    # --- Rule 1: word count ---
    words = _WORD_RE.findall(text)
    num_words = len(words)
    if num_words < 50 or num_words > 100_000:
        return False

    # --- Rule 2: average word length in [3, 10] ---
    # Use raw token lengths (includes digits if present).
    avg_len = sum(len(w) for w in words) / max(1, num_words)
    if avg_len < 3.0 or avg_len > 10.0:
        return False

    # --- Rule 3: >30% of lines end with "..." ---
    # Only consider non-empty lines to avoid penalizing trailing newlines.
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if lines:
        ellipsis_lines = sum(1 for ln in lines if _ELLIPSIS_LINE_RE.search(ln) is not None)
        if (ellipsis_lines / len(lines)) > 0.30:
            return False

    # --- Rule 4: <80% of words contain at least one letter ---
    letter_words = sum(1 for w in words if _HAS_LETTER_RE.search(w) is not None)
    if (letter_words / num_words) < 0.80:
        return False

    return True
