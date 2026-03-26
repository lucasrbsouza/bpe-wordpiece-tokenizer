"""Merge step for the BPE algorithm."""

import re


def merge_vocab(pair: tuple[str, str], v_in: dict[str, int]) -> dict[str, int]:
    """Replace all isolated occurrences of *pair* in the vocab with the merged token."""
    merged = "".join(pair)
    pattern = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
    return {pattern.sub(merged, word): freq for word, freq in v_in.items()}
