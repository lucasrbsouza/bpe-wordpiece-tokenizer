"""BPE training loop."""

from bpe.stats import get_stats
from bpe.merger import merge_vocab


def train(vocab: dict[str, int], num_merges: int) -> list[tuple[tuple[str, str], dict[str, int]]]:
    """Run *num_merges* BPE merge iterations and return the history of each step."""
    history: list[tuple[tuple[str, str], dict[str, int]]] = []
    current = dict(vocab)
    for _ in range(num_merges):
        pairs = get_stats(current)
        best = max(pairs, key=pairs.get)
        current = merge_vocab(best, current)
        history.append((best, dict(current)))
    return history
