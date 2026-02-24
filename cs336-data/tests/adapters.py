#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

from cs336_data.implementation import extract_text, language_identification
from cs336_data.implementation import mask_emails, mask_phone_numbers, mask_IP_addresses
from cs336_data.implementation import classify_NSFW, classify_toxic_speech
from cs336_data.implementation import gopher_quality_filter, quality_classifier


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text(html_bytes)
    
def run_identify_language(text: str) -> tuple[Any, float]:
    return language_identification(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_IP_addresses(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return classify_NSFW(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return quality_classifier(text)


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)


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
