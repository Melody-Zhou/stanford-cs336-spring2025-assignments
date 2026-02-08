import argparse
import csv
import random
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.quality_rules import gopher_quality_filter
from cs336_data.quality_rules import _WORD_RE, _HAS_LETTER_RE, _ELLIPSIS_LINE_RE


def is_html_response(rec) -> bool:
    if rec.record_type != WarcRecordType.response:
        return False
    http = rec.http_headers
    if http is None:
        return False
    ctype = (http.get("content-type") or "").lower()
    return ("text/html" in ctype) or ("application/xhtml" in ctype)


def preview(text: str, limit: int = 350) -> str:
    s = " ".join(text.split())
    return s[:limit]


def diagnose(text: str) -> str:
    """Return a short reason string for why the doc fails (or 'pass')."""
    # Keep this in sync with your rules.
    if not text:
        return "empty"

    words = _WORD_RE.findall(text)
    n = len(words)
    if n < 50:
        return "too_few_words"
    if n > 100_000:
        return "too_many_words"

    avg_len = sum(len(w) for w in words) / max(1, n)
    if avg_len < 3.0:
        return "avg_word_len_too_short"
    if avg_len > 10.0:
        return "avg_word_len_too_long"

    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if lines:
        ell = sum(1 for ln in lines if _ELLIPSIS_LINE_RE.search(ln) is not None)
        if (ell / len(lines)) > 0.30:
            return "too_many_ellipsis_lines"

    letter_words = sum(1 for w in words if _HAS_LETTER_RE.search(w) is not None)
    if (letter_words / n) < 0.80:
        return "too_few_letter_words"

    return "pass"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warc", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan-limit", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=Path("runs/gopher_20_samples.csv"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sample = []
    seen = 0

    with args.warc.open("rb") as f:
        for rec in ArchiveIterator(f, parse_http=True):
            if not is_html_response(rec):
                continue

            url = rec.headers.get("WARC-Target-URI", "") or ""
            html_bytes = rec.reader.read()
            text = extract_text_from_html_bytes(html_bytes)

            pred = gopher_quality_filter(text)
            reason = diagnose(text)

            seen += 1
            row = {
                "idx": seen,
                "url": url,
                "gopher_pass": "pass" if pred else "fail",
                "reason": reason,
                "text_preview": preview(text),
                # You fill these:
                "human_judgment": "",  # good / bad / unclear
                "notes": "",
            }

            # Reservoir sampling
            if len(sample) < args.n:
                sample.append(row)
            else:
                j = rng.randrange(seen)
                if j < args.n:
                    sample[j] = row

            if seen >= args.scan_limit:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=list(sample[0].keys()))
        w.writeheader()
        w.writerows(sample)

    print(f"Scanned HTML responses: {seen}")
    print(f"Wrote {len(sample)} samples to: {args.out}")


if __name__ == "__main__":
    main()
