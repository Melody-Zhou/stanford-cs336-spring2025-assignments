import argparse
import gzip
import json
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, List

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


# Global tokenizer for worker processes
_TOKENIZER = None
_EOS_ID = None


def _init_worker():
    """Initialize GPT-2 tokenizer once per worker process."""
    global _TOKENIZER, _EOS_ID
    _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    _EOS_ID = int(_TOKENIZER.eos_token_id)


def _tokenize_text(text: str) -> List[int]:
    """Tokenize a single document and append EOS."""
    ids = _TOKENIZER.encode(text)
    ids.append(_EOS_ID)
    return ids


def _iter_texts_from_jsonl_gz(path: Path) -> Iterable[str]:
    """Yield 'text' fields from a .json.gz file"""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if text:
                yield text


def _gather_texts(paths: List[Path]) -> List[str]:
    """Load all documents' text into memory."""
    texts: List[str] = []
    for p in paths:
        for t in _iter_texts_from_jsonl_gz(p):
            texts.append(t)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", type=str, required=True, help="Glob for filtered jsonl.gz")
    parser.add_argument("--output", type=str, required=True, help="Output .bin file path")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="Number of worker processes for tokenization")
    parser.add_argument("--chunksize", type=int, default=200, help="Chunksize for multiprocessing imap")
    parser.add_argument("--max-docs", type=int, default=0, help="Optional cap on number of documents")
    args = parser.parse_args()

    in_paths = sorted(Path().glob(args.input_glob))
    if not in_paths:
        raise FileNotFoundError(f"No files matched --input-glob={args.input_glob}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(in_paths)} input files.")
    print("Loading documents (extracting JSON['text'])...")

    texts = _gather_texts(in_paths)
    if args.max_docs and args.max_docs > 0:
        texts = texts[:args.max_docs]

    print(f"Total documents to tokenize: {len(texts)}")

    # Tokenize in parallel
    workers = max(1, int(args.workers))
    pool = mp.Pool(processes=workers, initializer=_init_worker)

    token_lists: List[List[int]] = []
    try:
        for ids in tqdm(
            pool.imap(_tokenize_text, texts, chunksize=args.chunksize),
            total=len(texts),
            desc="Tokenizing"
        ):
            token_lists.append(ids)
    finally:
        pool.close()
        pool.join()
    
    # Flattent and write
    total_tokens = sum(len(x) for x in token_lists)
    print(f"Total tokens (including EOS): {total_tokens}")

    all_ids = np.fromiter(
        (tok for doc in token_lists for tok in doc),
        dtype=np.uint32,
        count=total_tokens,
    )

    # GPT-2 vocab fits in uint16, so cast for compatibility with provided trainer.
    ids_u16 = all_ids.astype(np.uint16)

    print(f"Writing binary token file to: {output_path}")
    ids_u16.tofile(output_path)

    print("Done.")

if __name__ == "__main__":
    main()
