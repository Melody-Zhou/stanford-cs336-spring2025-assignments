from __future__ import annotations

import os
from typing import Any



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from cs336_data.extract_text import extract_text_from_html_bytes
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    from cs336_data.language_identification import identify_language
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    from cs336_data.pii_masking import mask_emails
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    from cs336_data.pii_masking import mask_phone_numbers
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    from cs336_data.pii_masking import mask_ips
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    from cs336_data.harmful_content import classify_nsfw
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    from cs336_data.harmful_content import classify_toxic_speech
    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    from cs336_data.quality_classifier import classify_quality
    return classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    from cs336_data.quality_rules import gopher_quality_filter
    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    from cs336_data.exact_deduplication import exact_line_deduplication
    return exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    from cs336_data.minhash_deduplication import minhash_deduplication
    return minhash_deduplication(
        input_files=input_files,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
        output_directory=output_directory
    )
