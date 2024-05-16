#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any
from cs336_data import cleaning


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    
    return cleaning.extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    
    return cleaning.identify_language(text)


def run_mask_emails(text: str) -> list[tuple[int, int]]:
    
    return cleaning.mask_emails(text)


def run_mask_phone_numbers(text: str) -> list[tuple[int, int]]:
    
    return cleaning.mask_phone(text)


def run_mask_ips(text: str) -> list[tuple[int, int]]:
    
    return cleaning.mask_ip(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    raise NotImplementedError


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
