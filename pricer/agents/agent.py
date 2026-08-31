"""The base every agent shares: a name, a colour, and a log line you can follow in the terminal."""

import logging
from collections.abc import Sequence

from pricer.llm import DailyLimitReached


class Agent:
    """An agent is anything with a `price(text) -> float`, plus logging so a run reads like a story."""

    name: str = "Agent"
    colour: str = "\033[37m"
    RESET = "\033[0m"

    def log(self, message: str) -> None:
        logging.info(f"{self.colour}[{self.name}] {message}{self.RESET}")

    def price(self, text: str) -> float:
        raise NotImplementedError


def price_all(agent: Agent, texts: Sequence[str]) -> list[float]:
    """Price as many as the provider's daily allowance permits, then stop and return those.

    A free tier runs out mid-evaluation, and losing ninety finished estimates to the ninety-first
    call is worse than scoring ninety. Callers align their labels with `len()` of the result.
    """
    guesses: list[float] = []
    for text in texts:
        try:
            guesses.append(agent.price(text))
        except DailyLimitReached as spent:
            agent.log(f"Stopping at {len(guesses)} of {len(texts)}: {spent}")
            break
    return guesses


def setup_logging(level: int = logging.INFO) -> None:
    """Agents log rather than print, so a planning run can be followed or piped to a file."""
    logging.basicConfig(level=level, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
