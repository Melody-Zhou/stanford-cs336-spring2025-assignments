import argparse
import csv
import random
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language


def is_html_response(rec) -> bool:
    if rec.record_type != WarcRecordType.response:
        return False
    http = rec.http_headers
    if http is None:
        return False
    ctype = (http.get("content-type") or "").lower()
    return ("text/html" in ctype) or ("application/xhtml" in ctype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warc", type=Path, required=True, help="Path to .warc.gz")
    ap.add_argument("--n", type=int, default=20, help="Number of samples")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--max-html", type=int, default=5000, help="Max HTML responses to scan")
    ap.add_argument("--out", type=Path, default=Path("runs/langid_20_samples.csv"))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Reservoir sampling: one pass, uniform sample without storing all documents.
    sample = []
    seen = 0

    with args.warc.open("rb") as f:
        for rec in ArchiveIterator(f, parse_http=True):
            if not is_html_response(rec):
                continue

            url = rec.headers.get("WARC-Target-URI", "") or ""
            html_bytes = rec.reader.read()
            text = extract_text_from_html_bytes(html_bytes)

            # Keep short preview for manual labeling (avoid huge dumps)
            preview = " ".join(text.split())
            preview = preview[:400]

            pred_lang, score = identify_language(text)

            seen += 1
            row = {
                "idx": seen,
                "url": url,
                "pred_lang": pred_lang,
                "score": f"{score:.4f}",
                "preview": preview,
                "true_lang": "",   # manual judgment
                "notes": "",       # optional
            }

            if len(sample) < args.n:
                sample.append(row)
            else:
                j = rng.randrange(seen)
                if j < args.n:
                    sample[j] = row

            if seen >= args.max_html:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=list(sample[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(sample)

    print(f"Scanned HTML responses: {seen}")
    print(f"Wrote {len(sample)} samples to: {args.out}")


if __name__ == "__main__":
    main()
