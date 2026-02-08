import argparse
import gzip
import random
import re
from pathlib import Path
from typing import Iterable, Tuple

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language
from cs336_data.quality_rules import gopher_quality_filter


def _open_maybe_gz(path: Path):
    """Open .gz as gzip, otherwise open as raw binary."""
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _to_single_line(text: str) -> str:
    """fastText expects one sample per line."""
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _iter_html_texts_from_warc(warc_path: Path, max_docs: int) -> Iterable[Tuple[str, str]]:
    """
    Yield (url, extracted_text) from HTML response records in a WARC(.gz).
    """
    seen = 0
    with _open_maybe_gz(warc_path) as f:
        for rec in ArchiveIterator(f, parse_http=True):
            if rec.record_type != WarcRecordType.response:
                continue
            http = rec.http_headers
            if http is None:
                continue
            ctype = (http.get("content-type") or "").lower()
            if ("text/html" not in ctype) and ("application/xhtml" not in ctype):
                continue

            url = rec.headers.get("WARC-Target-URI", "") or ""
            html_bytes = rec.reader.read()
            text = extract_text_from_html_bytes(html_bytes)

            seen += 1
            yield url, text
            if seen >= max_docs:
                break


def _keep_text(text: str, lang_thr: float) -> bool:
    """
    Apply a simple, shared filtering pipeline for both positives and negatives.
    """
    # Language filter (English)
    lang, score = identify_language(text)
    if lang != "en" or score < lang_thr:
        return False

    # Gopher rules
    if not gopher_quality_filter(text):
        return False

    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-warc", type=Path, required=True, help="Positive WARC (.warc or .warc.gz)")
    ap.add_argument("--neg-warc", type=Path, required=True, help="Negative WARC (.warc or .warc.gz)")
    ap.add_argument("--pos-n", type=int, default=10000, help="How many positive HTML docs to scan")
    ap.add_argument("--neg-n", type=int, default=10000, help="How many negative HTML docs to scan")
    ap.add_argument("--lang-thr", type=float, default=0.6, help="Language confidence threshold for English")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for downsampling")
    ap.add_argument("--out", type=Path, default=Path("runs/quality_train.txt"))
    ap.add_argument("--max-len", type=int, default=2000, help="Max characters per sample line")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pos_lines = []
    neg_lines = []

    # Collect positives
    for _, text in _iter_html_texts_from_warc(args.pos_warc, args.pos_n):
        if not text:
            continue
        if not _keep_text(text, args.lang_thr):
            continue
        line = _to_single_line(text)[: args.max_len]
        if line:
            pos_lines.append(f"__label__hq {line}")

    # Collect negatives
    for _, text in _iter_html_texts_from_warc(args.neg_warc, args.neg_n):
        if not text:
            continue
        if not _keep_text(text, args.lang_thr):
            continue
        line = _to_single_line(text)[: args.max_len]
        if line:
            neg_lines.append(f"__label__lq {line}")

    # Balance (optional): downsample to the smaller size
    m = min(len(pos_lines), len(neg_lines))
    if len(pos_lines) > m:
        pos_lines = rng.sample(pos_lines, m)
    if len(neg_lines) > m:
        neg_lines = rng.sample(neg_lines, m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for x in pos_lines:
            f.write(x + "\n")
        for x in neg_lines:
            f.write(x + "\n")

    print(f"pos_kept={len(pos_lines)} neg_kept={len(neg_lines)} balanced={m}")
    print(f"wrote: {args.out.resolve()}")


if __name__ == "__main__":
    main()
