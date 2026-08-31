"""One place to get an OpenAI-compatible client, so every LLM step in the project shares it.

Groq is the default because it is fast and free; any OpenAI-compatible endpoint works by overriding
`PROVIDERS`. The free tier's tokens-per-minute cap is the real constraint on this project, so calls
go through `chat`, which paces them against a shared token bucket and retries what the provider
rejects. Every part of the project that talks to a model uses it, which means a preprocessing pass
and an agent run in the same process share one budget instead of starving each other.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from threading import Lock

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

TPM = 7_000  # Groq's free tier allows 8,000 tokens per minute; leave headroom for retries
ATTEMPTS = 8
BACKOFF = 5.0  # seconds, when the provider does not say how long to wait
RETRY_AFTER = re.compile(r"try again in ([\d.]+)s")


@dataclass(frozen=True)
class Provider:
    key_name: str
    base_url: str | None
    default_model: str


PROVIDERS = {
    # Groq rotates its catalogue; `client_for("groq")[0].models.list()` shows what your key can reach.
    "groq": Provider("GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-20b"),
    "openai": Provider("OPENAI_API_KEY", None, "gpt-4o-mini"),
    "ollama": Provider("OLLAMA_API_KEY", "http://localhost:11434/v1", "llama3.2"),
}
DEFAULT_PROVIDER = "groq"


def client_for(provider: str = DEFAULT_PROVIDER) -> tuple[OpenAI, str]:
    """Return a client and that provider's default model.

    Raises with a clear message rather than letting an empty key produce a confusing 401.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}, expected one of {sorted(PROVIDERS)}")
    config = PROVIDERS[provider]
    key = os.getenv(config.key_name) or ("ollama" if provider == "ollama" else None)
    if not key:
        raise RuntimeError(f"{config.key_name} is not set -- put it in .env (see .env.example)")
    return OpenAI(api_key=key, base_url=config.base_url), config.default_model


class Limiter:
    """A token bucket over tokens-per-minute, because the free tier is the real bottleneck.

    Callers ask for their estimated cost before calling; when the bucket is dry they sleep. This is
    what turns "eight workers, all rate-limited, run dies" into a slow but finishing job.
    """

    def __init__(self, tokens_per_minute: int = TPM):
        self.rate = tokens_per_minute / 60.0
        self.capacity = float(tokens_per_minute)
        self.available = float(tokens_per_minute)
        self.updated = time.monotonic()
        self.lock = Lock()

    def acquire(self, cost: int) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.available = min(self.capacity, self.available + (now - self.updated) * self.rate)
                self.updated = now
                if self.available >= cost:
                    self.available -= cost
                    return
                wait = (cost - self.available) / self.rate
            time.sleep(wait)


SHARED_LIMITER = Limiter()


def chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 400,
    limiter: Limiter | None = None,
    attempts: int = ATTEMPTS,
    **kwargs,
) -> str:
    """One completion, paced and retried. Returns the reply text.

    A 429 usually names the wait in its message, so honour that rather than guessing. Anything else
    -- a timeout, a hiccup -- is worth a retry too, since these calls are always idempotent here.
    """
    limiter = limiter or SHARED_LIMITER
    cost = sum(len(m["content"]) for m in messages) // 4 + max_tokens
    last_error: Exception | None = None
    for attempt in range(attempts):
        limiter.acquire(cost)
        try:
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, **kwargs)
            return response.choices[0].message.content or ""
        except Exception as error:  # noqa: BLE001 -- retry anything: rate limits, timeouts, hiccups
            last_error = error
            match = RETRY_AFTER.search(str(error))
            delay = float(match.group(1)) + 0.5 if match else BACKOFF * (attempt + 1)
            logging.getLogger(__name__).debug(f"retrying in {delay:.1f}s after {error.__class__.__name__}")
            time.sleep(delay)
    raise RuntimeError(f"{model} failed after {attempts} attempts: {last_error}")
