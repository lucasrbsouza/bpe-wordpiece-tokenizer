"""Frequency engine for the BPE algorithm."""

from collections import defaultdict


def get_stats(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    """Count the frequency of every adjacent symbol pair across the corpus."""
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs
