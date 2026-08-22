"""Module-level workers for Phase 1 multiprocessing.

multiprocessing.Pool pickles the callable by qualified name, so these must live in
an importable module (not in the Modal entrypoint file).
"""

from __future__ import annotations

from cleaning import CleanResult, clean_document


def clean_plain(text: str) -> CleanResult:
    return clean_document(text, strict_ocr=False)


def clean_strict(text: str) -> CleanResult:
    return clean_document(text, strict_ocr=True)


def worker_for(strict_ocr: bool):
    return clean_strict if strict_ocr else clean_plain
