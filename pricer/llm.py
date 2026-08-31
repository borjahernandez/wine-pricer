"""One place to get an OpenAI-compatible client, so every LLM step in the project shares it.

Groq is the default because it is fast and its free tier is generous enough for a 50k-row
preprocessing pass; any OpenAI-compatible endpoint works by overriding `PROVIDERS`.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


@dataclass(frozen=True)
class Provider:
    key_name: str
    base_url: str | None
    default_model: str


PROVIDERS = {
    "groq": Provider("GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
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
