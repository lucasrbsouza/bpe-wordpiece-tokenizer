"""Entry point: runs the BPE training demo and the WordPiece tokenization."""

from bpe.corpus import INITIAL_VOCAB
from bpe.trainer import train
from wordpiece.tokenizer import load_tokenizer, tokenize, TEST_SENTENCE

NUM_MERGES = 5


def run_bpe() -> None:
    """Execute the BPE training loop and print each merge step."""
    print("=" * 60)
    print("BPE Training")
    print("=" * 60)
    history = train(INITIAL_VOCAB, NUM_MERGES)
    for iteration, (pair, vocab) in enumerate(history, start=1):
        print(f"\n[Iter {iteration}] Merged pair: {pair}")
        print(f"  vocab = {vocab}")
    print()


def run_wordpiece() -> None:
    """Load the BERT tokenizer and print WordPiece tokens for the test sentence."""
    print("=" * 60)
    print("WordPiece Tokenization (bert-base-multilingual-cased)")
    print("=" * 60)
    print(f"\nInput: {TEST_SENTENCE}\n")
    tokenizer = load_tokenizer()
    tokens = tokenize(tokenizer, TEST_SENTENCE)
    print(f"Tokens: {tokens}\n")


if __name__ == "__main__":
    run_bpe()
    run_wordpiece()
