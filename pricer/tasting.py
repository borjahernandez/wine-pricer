"""LLM preprocessing: turn a flowery tasting note into structured sommelier features.

Week 6 in the course uses an LLM to clean up messy product text. Wine tasting notes are already clean
prose, so the interesting pass here is the opposite direction -- *extraction*: read the note and score
the handful of dimensions a sommelier would actually use to place a bottle in a price bracket, plus a
one-line summary short enough to fine-tune on.

The result is threefold:

- six numeric features the classical baselines can use (`pricer.baselines.sommelier`),
- a short `summary` to compare against the full note as fine-tuning input,
- a concrete, cheap use of a hosted LLM over ~50k rows, which is the point of the exercise.

Runs are resumable: every completed row is appended to a JSONL cache keyed by wine id, so an
interrupted pass (or an exhausted rate limit) picks up where it stopped.
"""

import json
import re
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tqdm.auto import tqdm

from pricer.items import ROOT, Wine
from pricer.llm import client_for

CACHE_DIR = ROOT / "data" / "tasting"
WORKERS = 4
ATTEMPTS = 8
BACKOFF = 5.0  # seconds, when the provider does not say how long to wait
RETRY_AFTER = re.compile(r"try again in ([\d.]+)s")
DIMENSIONS = ("fruit", "oak", "tannin", "acidity", "body", "finish")

SYSTEM = (
    "You are a sommelier turning tasting notes into structured data. Reply with JSON only, no prose, no code fences."
)

INSTRUCTION = """Read this wine tasting note and reply with JSON:

{{
  "fruit": 0-3,      // fruit intensity
  "oak": 0-3,        // oak, vanilla, toast, spice from barrel
  "tannin": 0-3,     // grip and structure (0 for a white with none)
  "acidity": 0-3,    // freshness
  "body": 0-3,       // light to full
  "finish": 0-3,     // length of the finish
  "age_worthy": true or false,  // does the note suggest cellaring?
  "summary": "at most 20 words, the note's essence, no price or score"
}}

Score 0 when the note gives no evidence for a dimension. Never mention or guess the price.

Tasting note:
{note}"""


class Tasting(BaseModel):
    """The structured reading of one tasting note."""

    fruit: int = Field(ge=0, le=3)
    oak: int = Field(ge=0, le=3)
    tannin: int = Field(ge=0, le=3)
    acidity: int = Field(ge=0, le=3)
    body: int = Field(ge=0, le=3)
    finish: int = Field(ge=0, le=3)
    age_worthy: bool
    summary: str

    def as_text(self) -> str:
        """A compact rendering, used as the model-visible text in the `summary` ablation."""
        scores = " ".join(f"{name}={value}" for name, value in self.scores().items())
        return f"{self.summary}\n{scores} age_worthy={self.age_worthy}"

    def scores(self) -> dict[str, int]:
        return {
            "fruit": self.fruit,
            "oak": self.oak,
            "tannin": self.tannin,
            "acidity": self.acidity,
            "body": self.body,
            "finish": self.finish,
        }


def cache_path(split: str) -> Path:
    return CACHE_DIR / f"{split}.jsonl"


def load_cache(split: str) -> dict[int, Tasting]:
    """Read the rows finished by earlier runs. Malformed lines are skipped, not fatal."""
    path = cache_path(split)
    if not path.exists():
        return {}
    done: dict[int, Tasting] = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            done[record["id"]] = Tasting.model_validate(record["tasting"])
        except (json.JSONDecodeError, KeyError, ValidationError):
            continue
    return done


def extract(note: str, client: OpenAI, model: str) -> Tasting:
    """One call, retried: free tiers rate-limit hard and small models sometimes wrap JSON in prose.

    A 429 usually names the wait in its message, so honour that rather than guessing.
    """
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": INSTRUCTION.format(note=note)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=400,
                # gpt-oss models think before answering; this is pure extraction, so keep it brief
                extra_body={"reasoning_effort": "low"},
            )
            return Tasting.model_validate_json(response.choices[0].message.content)
        except Exception as error:  # noqa: BLE001 -- retry anything: rate limits, bad JSON, timeouts
            last_error = error
            match = RETRY_AFTER.search(str(error))
            time.sleep(float(match.group(1)) + 0.5 if match else BACKOFF * (attempt + 1))
    raise RuntimeError(f"Extraction failed after {ATTEMPTS} attempts: {last_error}")


def run(
    wines: Sequence[Wine],
    split: str,
    provider: str = "groq",
    model: str | None = None,
    workers: int = WORKERS,
    limit: int | None = None,
) -> dict[int, Tasting]:
    """Extract features for every wine in `wines`, skipping ids already in the cache."""
    client, default_model = client_for(provider)
    model = model or default_model
    done = load_cache(split)
    todo = [wine for wine in wines if wine.id not in done][:limit]
    print(f"{split}: {len(done):,} cached, {len(todo):,} to do with {model}")
    if not todo:
        return done

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    with open(cache_path(split), "a") as handle, ThreadPoolExecutor(max_workers=workers) as pool:

        def process(wine: Wine) -> None:
            tasting = extract(wine.description, client, model)
            with lock:
                handle.write(json.dumps({"id": wine.id, "tasting": tasting.model_dump()}) + "\n")
                handle.flush()
            done[wine.id] = tasting

        list(tqdm(pool.map(process, todo), total=len(todo), desc=split))
    return done


def attach(wines: Iterable[Wine], split: str) -> int:
    """Copy cached summaries onto the wines. Returns how many were matched."""
    done = load_cache(split)
    matched = 0
    for wine in wines:
        tasting = done.get(wine.id)
        if tasting:
            wine.summary = tasting.as_text()
            matched += 1
    return matched
