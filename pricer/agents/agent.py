"""The base every agent shares: a name, a colour, and a log line you can follow in the terminal."""

import logging


class Agent:
    """An agent is anything with a `price(text) -> float`, plus logging so a run reads like a story."""

    name: str = "Agent"
    colour: str = "\033[37m"
    RESET = "\033[0m"

    def log(self, message: str) -> None:
        logging.info(f"{self.colour}[{self.name}] {message}{self.RESET}")

    def price(self, text: str) -> float:
        raise NotImplementedError


def setup_logging(level: int = logging.INFO) -> None:
    """Agents log rather than print, so a planning run can be followed or piped to a file."""
    logging.basicConfig(level=level, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
