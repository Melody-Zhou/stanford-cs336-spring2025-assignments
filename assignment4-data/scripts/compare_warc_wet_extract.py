import argparse
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes


def iter_warc_html_texts(warc_gz_path: Path, limit: int):
    """
    Iterate response records in a WARC(.gz), extract text from raw HTML bytes.
    """
    out = []
    with warc_gz_path.open("rb") as f:
        for rec in ArchiveIterator(f, parse_http=True):
            if rec.record_type != WarcRecordType.response:
                continue
            # Only keep HTML-ish responses
            http = rec.http_headers
            if http is None:
                continue
            ctype = (http.get("content-type") or "").lower()
            if "text/html" not in ctype and "application/xhtml" not in ctype:
                continue

            # Raw body bytes
            body = rec.reader.read()
            text = extract_text_from_html_bytes(body)
            out.append((rec.headers.get("WARC-Target-URI", ""), text))
            if len(out) >= limit:
                break
    return out


def iter_wet_texts(wet_gz_path: Path, limit: int):
    """
    Iterate conversion records in a WET(.gz) and read extracted plain text.
    """
    out = []
    with wet_gz_path.open("rb") as f:
        for rec in ArchiveIterator(f, parse_http=False):
            if rec.record_type != WarcRecordType.conversion:
                continue
            text = rec.reader.read().decode("utf-8", errors="replace")
            out.append((rec.headers.get("WARC-Target-URI", ""), text))
            if len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warc", type=Path, required=True, help="Path to .warc.gz")
    ap.add_argument("--wet", type=Path, required=True, help="Path to .warc.wet.gz")
    ap.add_argument("--limit", type=int, default=5, help="How many records to compare")
    ap.add_argument("--outdir", type=Path, default=Path("runs/tmp_compare"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    warc_items = iter_warc_html_texts(args.warc, args.limit)
    wet_items = iter_wet_texts(args.wet, args.limit)

    n = min(len(warc_items), len(wet_items))
    print(f"Collected warc={len(warc_items)} wet={len(wet_items)} compare_n={n}")

    for i in range(n):
        warc_url, warc_txt = warc_items[i]
        wet_url, wet_txt = wet_items[i]

        # Save for manual inspection
        (args.outdir / f"{i:02d}_warc_url.txt").write_text(warc_url, encoding="utf-8")
        (args.outdir / f"{i:02d}_wet_url.txt").write_text(wet_url, encoding="utf-8")
        (args.outdir / f"{i:02d}_warc_extracted.txt").write_text(warc_txt, encoding="utf-8")
        (args.outdir / f"{i:02d}_wet_extracted.txt").write_text(wet_txt, encoding="utf-8")

        # Quick stats
        print(f"\n[{i:02d}]")
        print(f"  WARC URL: {warc_url}")
        print(f"  WET  URL: {wet_url}")
        print(f"  len(warc_extracted)={len(warc_txt)} len(wet_extracted)={len(wet_txt)}")

    print(f"\nSaved side-by-side outputs to: {args.outdir.resolve()}")
    print("Tip: use `diff -u` or open the two txt files to compare content/boilerplate/noise.")


if __name__ == "__main__":
    main()    