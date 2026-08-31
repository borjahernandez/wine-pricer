"""Turning a Wine into the text a language model sees.

Kept apart from `items.py` because the interesting choices live here: which text (the full tasting
note or the LLM summary), how hard to truncate it, and whether the answer is attached.

A fine-tune learns the *format* as well as the task, so the training prompt and the inference prompt
must be identical up to the final `Price is $`. `for_training` appends the answer, `for_inference`
stops at the prefix; both share `render`, which is the only place the layout is defined.
"""

from collections.abc import Iterable, Sequence
from typing import Protocol

from pricer.items import PREFIX, QUESTION, Wine

CUTOFF = 160  # tokens of wine text; the 99th percentile note fits comfortably


class Tokenizer(Protocol):
    """The slice of a HuggingFace tokenizer used here, so transformers stays an optional import."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> Sequence[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


def text_for(wine: Wine, use_summary: bool = False) -> str:
    """The model-visible description: the composed tasting note, or its LLM summary."""
    if use_summary:
        if not wine.summary:
            raise ValueError(f"Wine {wine.id} has no summary -- run scripts/tasting.py first")
        return wine.summary
    if not wine.full:
        raise ValueError(f"Wine {wine.id} has no composed text -- it did not come from the parser")
    return wine.full


def truncate(text: str, tokenizer: Tokenizer, cutoff: int = CUTOFF) -> str:
    """Cut the text to `cutoff` tokens, on a word boundary, so no prompt runs long.

    Truncating in token space and decoding back can split a word in half, which teaches the model
    nothing useful, so drop the final partial word.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= cutoff:
        return text
    decoded = tokenizer.decode(tokens[:cutoff])
    return decoded.rsplit(" ", 1)[0]


def render(text: str, price: float | None = None) -> str:
    """The one canonical layout. With a price it is a training example, without it a question."""
    answer = f"{round(price)}.00" if price is not None else ""
    return f"{QUESTION}\n\n{text}\n\n{PREFIX}{answer}"


def for_training(
    wine: Wine, tokenizer: Tokenizer | None = None, cutoff: int = CUTOFF, use_summary: bool = False
) -> str:
    text = text_for(wine, use_summary)
    if tokenizer:
        text = truncate(text, tokenizer, cutoff)
    return render(text, wine.price)


def for_inference(
    wine: Wine, tokenizer: Tokenizer | None = None, cutoff: int = CUTOFF, use_summary: bool = False
) -> str:
    text = text_for(wine, use_summary)
    if tokenizer:
        text = truncate(text, tokenizer, cutoff)
    return render(text)


def prepare(
    wines: Iterable[Wine], tokenizer: Tokenizer | None = None, cutoff: int = CUTOFF, use_summary: bool = False
) -> None:
    """Fill in `wine.prompt` for a whole split, ready to push to the Hub for fine-tuning."""
    for wine in wines:
        wine.prompt = for_training(wine, tokenizer, cutoff, use_summary)
