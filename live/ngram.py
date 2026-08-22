"""Seed-independent word n-gram hashing for decontamination.

`dedup.word_ngrams` uses Python's builtin `hash()`, which is randomized per process
by PYTHONHASHSEED. That is fine when the eval set and the documents are hashed in the
SAME process (as the guide does), but we build the contamination set once in its own
container and reuse it in ~20 workers, so the hashes must be stable across processes.

Words are mapped to stable 64-bit ids with blake2b (cached, since words repeat
heavily), then n-gram hashes are a polynomial roll over those ids, computed with
vectorised uint64 numpy (wraparound mod 2**64 is the modulus).
"""

from __future__ import annotations

import hashlib

import numpy as np

NGRAM_PRIME = np.uint64(1_099_511_628_211)  # FNV-1a 64-bit prime
_WORD_IDS: dict[str, int] = {}


def word_id(word: str) -> int:
    v = _WORD_IDS.get(word)
    if v is None:
        v = int.from_bytes(
            hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "little")
        _WORD_IDS[word] = v
    return v


def _powers(n: int) -> np.ndarray:
    pw = np.ones(n, dtype=np.uint64)
    for j in range(1, n):
        pw[j] = pw[j - 1] * NGRAM_PRIME   # uint64 multiply wraps mod 2**64
    return pw


def gram_hashes(tokens: list[str], n: int) -> np.ndarray:
    """Stable uint64 hash of every n-gram in `tokens` (may contain duplicates)."""
    if len(tokens) < n:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter((word_id(w) for w in tokens), dtype=np.uint64, count=len(tokens))
    win = np.lib.stride_tricks.sliding_window_view(ids, n)
    with np.errstate(over="ignore"):
        return (win * _powers(n)).sum(axis=1)


def probe() -> int:
    """Constant that must agree between the contam builder and every consumer."""
    return int(gram_hashes(["__slm125m_probe__", "alpha", "beta"], 3)[0])
