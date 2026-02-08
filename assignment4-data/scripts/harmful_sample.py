import argparse
import csv
import random
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech


def is_html_response(rec) -> bool:
    if rec.record_type != WarcRecordType.response:
        return False
    http = rec.http_headers
    if http is None:
        return False
    ctype = (http.get("content-type") or "").lower()
    return ("text/html" in ctype) or ("application/xhtml" in ctype)


def preview(text: str, limit: int = 320) -> str:
    s = " ".join(text.split())
    return s[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warc", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan-limit", type=int, default=20000, help="Max HTML responses to scan")
    ap.add_argument("--out", type=Path, default=Path("tmp/harmful_20_samples.csv"))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Reservoir sampling over HTML responses
    sample = []
    seen = 0

    with args.warc.open("rb") as f:
        for rec in ArchiveIterator(f, parse_http=True):
            if not is_html_response(rec):
                continue

            url = rec.headers.get("WARC-Target-URI", "") or ""
            html_bytes = rec.reader.read()
            text = extract_text_from_html_bytes(html_bytes)

            nsfw_lab, nsfw_score = classify_nsfw(text)
            tox_lab, tox_score = classify_toxic_speech(text)

            seen += 1
            row = {
                "idx": seen,
                "url": url,
                "nsfw_pred": nsfw_lab,
                "nsfw_score": f"{nsfw_score:.4f}",
                "toxic_pred": tox_lab,
                "toxic_score": f"{tox_score:.4f}",
                "text_preview": preview(text),
                # Fill these manually:
                "true_nsfw": "",   # "nsfw" or "non-nsfw"
                "true_toxic": "",  # "toxic" or "non-toxic"
                "notes": "",
            }

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
