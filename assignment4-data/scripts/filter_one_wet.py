import argparse
import gzip
import json
import os
import re
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Any

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.language_identification import identify_language
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.quality_rules import gopher_quality_filter
from cs336_data.pii_masking import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.quality_classifier import quality_score


_WS_RE = re.compile(r"\s+")


def _to_single_line(text: str) -> str:
    """Keep docs compact for jsonl."""
    return _WS_RE.sub(" ", text).strip()


@dataclass
class FileStats:
    wet_file: str
    records_total: int = 0
    records_with_text: int = 0

    keep_lang: int = 0
    keep_harmful: int = 0
    keep_gopher: int = 0
    keep_quality: int = 0
    written: int = 0

    masked_emails: int = 0
    masked_phones: int = 0
    masked_ips: int = 0


def _extract_wet_text(rec) -> str:
    """Extract plain text from a WET 'conversion' record."""
    try:
        b = rec.reader.read()
    except Exception:
        return ""
    if not b:
        return ""
    return b.decode("utf-8", errors="replace")


class ReservoirSampler:
    """Streamingly sample up to k items uniformly without storing the full stream."""
    def __init__(self, k: int, seed: int = 0):
        self.k = k
        self.rng = random.Random(seed)
        self.n_seen = 0
        self.items: list[dict[str, Any]] = []

    def offer(self, item: dict[str, Any]) -> None:
        if self.k <= 0:
            return
        self.n_seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
            return
        j = self.rng.randrange(self.n_seen)
        if j < self.k:
            self.items[j] = item


def process_one_wet(
    input_path: str,
    output_path: str,
    stats_path: str,
    *,
    lang_thr: float = 0.8,
    toxic_thr: float = 0.8,
    nsfw_thr: float = 0.8,
    quality_thr: float = 0.5,
    min_chars: int = 200,
    dump_removed: int = 0,
    seed: int = 0,
    removed_out: str = "runs/removed_5.jsonl"
) -> Tuple[str, str]:
    """
    Filter a single WET file and write:
      - output_path: .jsonl.gz (one doc per line)
      - stats_path: .json
    """
    sampler = ReservoirSampler(k=0, seed=0)
    if dump_removed > 0:
        sampler = ReservoirSampler(k=dump_removed, seed=seed)

    st = FileStats(wet_file=os.path.basename(input_path))

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(input_path, "rb") as f_in, gzip.open(out_p, "wt", encoding="utf-8") as f_out:
        for rec in ArchiveIterator(f_in, parse_http=False):
            # WET uses conversion records for extracted text
            if rec.record_type != WarcRecordType.conversion:
                continue

            st.records_total += 1
            url = rec.headers.get("WARC-Target-URI", "") or ""

            text = _extract_wet_text(rec)
            text = text.strip()
            if len(text) < min_chars:
                continue

            st.records_with_text += 1

            # 1) Language filter
            lang, lscore = identify_language(text)
            if not (lang == "en" and lscore >= lang_thr):
                sampler.offer({
                    "action": "dropped",
                    "reason": "language",
                    "url": url,
                    "lang": lang,
                    "lang_score": float(lscore),
                    "snippet": _to_single_line(text),
                })
                continue
            st.keep_lang += 1

            # 2) Harmful content filter (drop high-confidence harmful docs)
            nsfw_lab, nsfw_score = classify_nsfw(text)
            tox_lab, tox_score = classify_toxic_speech(text)
            if (nsfw_lab == "nsft" and nsfw_score >= nsfw_thr) or (tox_lab == "toxic" and tox_score >= toxic_thr):
                sampler.offer({
                    "action": "dropped",
                    "reason": "harmful",
                    "url": url,
                    "nsfw": [nsfw_lab, float(nsfw_score)],
                    "toxic": [tox_lab, float(tox_score)],
                    "snippet": _to_single_line(text),
                })
                continue
            st.keep_harmful += 1

            # 3) Gopher heuristic rules
            if not gopher_quality_filter(text):
                sampler.offer({
                    "action": "dropped",
                    "reason": "gopher",
                    "url": url,
                    "snippet": _to_single_line(text),
                })                
                continue
            st.keep_gopher += 1

            # 4) PII masking (mask, do not drop)
            text, n_email = mask_emails(text)
            text, n_phone = mask_phone_numbers(text)
            text, n_ip = mask_ips(text)
            st.masked_emails += n_email
            st.masked_phones += n_phone
            st.masked_ips += n_ip

            # 5) Learned quality score
            q = float(quality_score(text))
            if q < quality_thr:
                sampler.offer({
                    "action": "dropped",
                    "reason": "quality",
                    "url": url,
                    "quality": float(q),
                    "snippet": _to_single_line(text),
                })                   
                continue
            st.keep_quality += 1

            # Write output
            doc = {
                "url": url,
                "text": _to_single_line(text),
                "lang": lang,
                "lang_score": float(lscore),
                "quality_score": q
            }
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            st.written += 1
    
    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(asdict(st), sf, ensure_ascii=False, indent=2)

    if dump_removed > 0:
        outp = Path(removed_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as rf:
            for it in sampler.items:
                rf.write(json.dumps(it, ensure_ascii=False) + "\n")

    return str(out_p), str(stats_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--stats", dest="stats", required=True)

    ap.add_argument("--lang-thr", type=float, default=0.8)
    ap.add_argument("--toxic-thr", type=float, default=0.8)
    ap.add_argument("--nsfw-thr", type=float, default=0.8)
    ap.add_argument("--quality-thr", type=float, default=0.5)
    ap.add_argument("--min-chars", type=int, default=200)

    ap.add_argument("--dump-removed", type=int, default=0,
                    help="Number of removed/modified samples to dump (for inspection)")
    ap.add_argument("--removed-out", type=str, default="runs/removed_5.jsonl",
                    help="Where to write dumped removed/modified samples (jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    process_one_wet(
        args.inp,
        args.out,
        args.stats,
        lang_thr=args.lang_thr,
        toxic_thr=args.toxic_thr,
        nsfw_thr=args.nsfw_thr,
        quality_thr=args.quality_thr,
        min_chars=args.min_chars,
        dump_removed=args.dump_removed,
        removed_out=args.removed_out,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
