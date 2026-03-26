"""WordPiece tokenization using the multilingual BERT model from Hugging Face."""

from transformers import AutoTokenizer, PreTrainedTokenizerBase

_MODEL_NAME = "bert-base-multilingual-cased"

TEST_SENTENCE = (
    "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
)


def load_tokenizer() -> PreTrainedTokenizerBase:
    """Load the multilingual BERT tokenizer (WordPiece)."""
    return AutoTokenizer.from_pretrained(_MODEL_NAME)


def tokenize(tokenizer: PreTrainedTokenizerBase, text: str) -> list[str]:
    """Segment *text* into WordPiece subword tokens."""
    return tokenizer.tokenize(text)
