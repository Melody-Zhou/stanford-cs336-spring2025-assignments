from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    Extract plain text from raw HTML bytes.

    Strategy:
    1) Try UTF-8 decode first (fast path).
    2) If it fails, use resiliparse encoding detection, then decode with that encoding.
    3) Fall back to latin-1 if detection is missing/invalid.
    """
    if html_bytes is None:
        return ""

    # Fast path: UTF-8
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Detect encoding when UTF-8 decoding fails
        enc = None
        try:
            info = detect_encoding(html_bytes)
            if isinstance(info, dict):
                enc = info.get("encoding") or info.get("charset")
            else:
                # If library returns a plain string or other object
                enc = getattr(info, "encoding", None) or str(info)
        except Exception:
            enc = None
        
        if not enc:
            enc = "latin-1"

        try:
            html_str = html_bytes.decode(enc)
        except Exception:
            # last resort: avoid crash
            html_str = html_bytes.decode("latin-1", errors="replace")
    
    # Extract plain text
    return extract_plain_text(html_str)