import argparse
import csv
import random
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.pii_masking import mask_emails, mask_phone_numbers, mask_ips


def is_html_response(rec) -> bool:
    if rec.record_type != WarcRecordType.response:
        return False
    http = rec.http_headers
    if http is None:
        return False
    ctype = (http.get("content-type") or "").lower()
    return ("text/html" in ctype) or ("application/xhtml" in ctype)


def mask_all(text: str):
    """Apply the three masking functions sequentially and return per-type counts."""
    t, ne = mask_emails(text)
    t, np = mask_phone_numbers(t)
    t, ni = mask_ips(t)
    return t, ne, np, ni


def compact_preview(text: str, limit: int = 320) -> str:
    """Collapse whitespace and truncate to make manual inspection easier."""
    s = " ".join(text.split())
    return s[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warc", type=Path, required=True, help="Path to .warc.gz")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--need", type=int, default=20, help="How many replaced samples to output")
    ap.add_argument("--scan-limit", type=int, default=20000, help="Max HTML responses to scan")
    ap.add_argument("--out", type=Path, default=Path("runs/pii_20_samples.csv"))
    args = ap.parse_args()

    rng = random.Random(args.seed)

    replaced_rows = []
    scanned = 0

    with args.warc.open("rb") as f:
        for rec in ArchiveIterator(f, parse_http=True):
            if not is_html_response(rec):
                continue

            url = rec.headers.get("WARC-Target-URI", "") or ""
            html_bytes = rec.reader.read()
            text = extract_text_from_html_bytes(html_bytes)

            masked, ne, np, ni = mask_all(text)
            total = ne + np + ni

            scanned += 1
            if total > 0:
                replaced_rows.append(
                    {
                        "url": url,
                        "emails": ne,
                        "phones": np,
                        "ips": ni,
                        "total": total,
                        "before_preview": compact_preview(text),
                        "after_preview": compact_preview(masked),
                        "notes_fp": "",
                        "notes_fn": "",
                    }
                )

            if scanned >= args.scan_limit:
                break

    # If fewer than needed samples have replacements, output all we have.
    if len(replaced_rows) <= args.need:
        sample = replaced_rows
    else:
        sample = rng.sample(replaced_rows, args.need)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=list(sample[0].keys()))
        w.writeheader()
        w.writerows(sample)

    print(f"Scanned HTML responses: {scanned}")
    print(f"Found replaced samples: {len(replaced_rows)}")
    print(f"Wrote {len(sample)} samples to: {args.out}")


if __name__ == "__main__":
    main()
