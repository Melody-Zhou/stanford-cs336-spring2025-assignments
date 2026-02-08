import re
from typing import Tuple

EMAIL_TOKEN = "|||EMAIL_ADDRESS|||"
PHONE_TOKEN = "|||PHONE_NUMBER|||"
IP_TOKEN = "|||IP_ADDRESS|||"

# A reasonably robust email regex for common cases.
# - Avoids matching trailing punctuation via boundary handling.
# - Covers typical local-part and domain patterns.
_EMAIL_RE = re.compile(
    r"""
    (?<![\w.+-])                           # left boundary (not part of an email-ish token)
    [A-Za-z0-9._%+-]+                      # local part
    @
    (?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}       # domain
    (?![\w.+-])                            # right boundary
    """,
    re.VERBOSE,
)

# US-centric phone patterns with some robustness:
# - Optional country code (+1 or 1)
# - Optional parentheses around area code
# - Separators: space, dot, dash
# - Optional extension: ext, x, ext.
_PHONE_RE = re.compile(
    r"""
    (?<!\w)                                # left boundary
    (?:\+?1[\s.-]?)?                       # optional country code
    (?:\(\s*\d{3}\s*\)|\d{3})              # area code
    [\s.-]?                                # separator
    \d{3}                                  # prefix
    [\s.-]?                                # separator
    \d{4}                                  # line number
    (?:\s*(?:ext\.?|x)\s*\d{1,6})?         # optional extension
    (?!\w)                                 # right boundary
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Strict IPv4 octet: 0-255
_OCTET = r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
_IP_RE = re.compile(
    rf"""
    (?<!\d)                                # left boundary: previous char is not a digit
    (?:{_OCTET}\.){{3}}{_OCTET}            # IPv4
    (?!\d)                                 # right boundary: next char is not a digit
    (?!\.\d)
    """,
    re.VERBOSE,
)


def mask_emails(text: str) -> Tuple[str, int]:
    """
    Replace all email addresses in `text` with EMAIL_TOKEN.

    Returns: (new_text, num_masked)
    """
    if not text:
        return text, 0
    new_text, n = _EMAIL_RE.subn(EMAIL_TOKEN, text)
    return new_text, n


def mask_phone_numbers(text: str) -> Tuple[str, int]:
    """
    Replace (mostly US-style) phone numbers in `text` with PHONE_TOKEN.

    Returns: (new_text, num_masked)
    """
    if not text:
        return text, 0
    new_text, n = _PHONE_RE.subn(PHONE_TOKEN, text)
    return new_text, n


def mask_ips(text: str) -> Tuple[str, int]:
    """
    Replace IPv4 addresses in `text` with IP_TOKEN.

    Returns: (new_text, num_masked)
    """
    if not text:
        return text, 0
    new_text, n = _IP_RE.subn(IP_TOKEN, text)
    return new_text, n
