import hashlib
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def _normalize_text(text: str) -> str:
    """
    Normalize text before MinHash + Jaccard:
    - NFD unicode normalization
    - remove accents (combining marks)
    - lowercase
    - remove punctuation
    - normalize whitespace
    """
    if not text:
        return ""

    # NFD + strip accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # Lowercase
    text = text.lower()

    # Remove punctuation (map to space)
    chars = []
    for ch in text:
        if unicodedata.category(ch).startswith("P"):
            chars.append(" ")
        else:
            chars.append(ch)
    text = "".join(chars)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_ngrams(text: str, n: int) -> List[str]:
    """Generate word-level n-grams as string."""
    if n <= 0:
        return []
    words = text.split()
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) -n + 1)]


def _stable_u64(x: bytes) -> int:
    """Hash bytes -> unint64."""
    return int.from_bytes(hashlib.blake2b(x, digest_size=8).digest(), "little", signed=False)


def _make_seeds(num_hashes: int) -> List[int]:
    """Deterministically generate seeds (uint64) for hash mixing."""
    rng = random.Random(0)
    return [rng.getrandbits(64) for _ in range(num_hashes)]


def _minhash_signature(ngrams: Iterable[str], num_hashes: int, seeds: List[int]) -> Tuple[int, ...]:
    """
    Compute MinHash signature using a single base hash per n-gram + seed mixing.
    """
    # Initialize mins to max unit64
    mins = [2**64 - 1] * num_hashes
    for g in ngrams:
        base = _stable_u64(g.encode("utf-8", errors="replace"))
        # Derive k hashes cheaply from base + seed
        for i in range(num_hashes):
            v = (base ^ seeds[i]) * 0x9E3779B185EBCA87 & (2**64 - 1)
            if v < mins[i]:
                mins[i] = v
    return tuple(mins)


def _band_hash(sig_slice: Tuple[int, ...]) -> bytes:
    """Hash a band slice into a bucket key."""
    # Pack as bytes deterministically
    b = b"".join(x.to_bytes(8, "little", signed=False) for x in sig_slice)
    return hashlib.blake2b(b, digest_size=16).digest()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a) + len(b) - inter
    return inter / union if union > 0 else 0.0


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    """
    Fuzzy document deduplication via MinHash + LSH + exact n-gram Jaccard.

    Deterministic policy: with each duplicate cluster, keep the earliest document
    in input_files order and drop others.
    """
    paths = [Path(p) for p in input_files]
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert num_hashes % num_bands == 0, "num_hashed must be divisible by num_bands"
    rows_per_band = num_hashes // num_bands
    seeds = _make_seeds(num_hashes)

    # 1) Load docs, normalize, build n-gram sets, compute signatures
    raw_texts: List[str] = []
    ngram_sets: List[Set[str]] = []
    signatures: List[Tuple[int, ...]] = []

    for p in paths:
        raw = p.read_text(encoding="utf-8", errors="replace")
        raw_texts.append(raw)

        norm = _normalize_text(raw)
        grams = _word_ngrams(norm, ngrams)
        s = set(grams)
        ngram_sets.append(s)

        sig = _minhash_signature(s, num_hashes=num_hashes, seeds=seeds)
        signatures.append(sig)

    # 2) LSH bucketing -> candidate pairs
    buckets: Dict[Tuple[int, bytes], List[int]] = {}
    for doc_id, sig in enumerate(signatures):
        for b in range(num_bands):
            start = b * rows_per_band
            end = start + rows_per_band
            key = (b, _band_hash(sig[start:end]))
            buckets.setdefault(key, []).append(doc_id)

    candidate_pairs: Set[Tuple[int, int]] = set()
    for _, ids in buckets.items():
        if len(ids) < 2:
            continue
        ids_sorted = sorted(ids)
        for i in range(len(ids_sorted)):
            for j in range(i + 1, len(ids_sorted)):
                a, b = ids_sorted[i], ids_sorted[j]
                candidate_pairs.add((a, b))

    # 3) Verify candidates with exact Jaccard and build duplicate graph via union-find
    uf = _UnionFind(len(paths))
    for a, b in candidate_pairs:
        sim = _jaccard(ngram_sets[a], ngram_sets[b])
        if sim >= jaccard_threshold:
            uf.union(a, b)

    # 4) Collect connected components; keep earliest index in each component
    comps: Dict[int, List[int]] = {}
    for i in range(len(paths)):
        root = uf.find(i)
        comps.setdefault(root, []).append(i)

    keep: Set[int] = set()
    for _, members in comps.items():
        keep.add(min(members))

    # 5) Write kept docs to output directory with same filenames
    for i, p in enumerate(paths):
        if i not in keep:
            continue
        (out_dir / p.name).write_text(raw_texts[i], encoding="utf-8")