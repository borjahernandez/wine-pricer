"""The scanner: read the wine press and return wines that carry both a tasting note and a price.

Article text is messy prose, so the structuring is done by an LLM constrained to JSON and validated
by Pydantic. Anything without a price is discarded -- without one there is nothing to compare an
estimate against.
"""

import json

from pydantic import BaseModel, Field, ValidationError

from pricer import deals
from pricer.agents.agent import Agent
from pricer.llm import chat, client_for

SYSTEM = "You extract wine mentions from articles as JSON. Reply with JSON only, no prose, no code fences."

INSTRUCTION = """Find every specific wine in this article that is quoted with a price.

Reply with {{"wines": [...]}}, one object per wine:
  "name": producer, vintage and bottling as printed
  "note": the article's tasting description, in its own words, at least 15 words; if the article is
          terse, describe the wine from what it does say -- never invent flavours it does not mention
  "price": the listed price in US dollars, as a number
  "url": the article url

Skip wines with no price, and skip the article entirely if it names none. Return {{"wines": []}} then.

Article url: {url}

{text}"""


class Listing(BaseModel):
    """A wine found in the wild, with the price someone is asking for it."""

    name: str
    note: str = Field(min_length=40)
    price: float = Field(gt=0)
    url: str = ""

    def __repr__(self) -> str:
        return f"<{self.name} listed at ${self.price:.0f}>"


class Listings(BaseModel):
    wines: list[Listing]


class ScannerAgent(Agent):
    name = "Scanner Agent"
    colour = "\033[31m"

    def __init__(self, provider: str = "groq", model: str | None = None):
        self.log(f"Setting up with {provider}")
        self.client, default_model = client_for(provider)
        self.model = model or default_model

    def read(self, article: deals.Article) -> list[Listing]:
        """Structure one article. A model that returns nonsense costs us this article, not the run."""
        reply = chat(
            self.client,
            self.model,
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": INSTRUCTION.format(url=article.url, text=article.as_text())},
            ],
            max_tokens=1_500,
            response_format={"type": "json_object"},
            temperature=0,
            extra_body={"reasoning_effort": "low"},
        )
        try:
            found = Listings.model_validate_json(reply)
        except (ValidationError, json.JSONDecodeError) as error:
            self.log(f"Could not structure '{article.title[:50]}': {error.__class__.__name__}")
            return []
        for listing in found.wines:
            listing.url = listing.url or article.url
        return found.wines

    def scan(self, per_feed: int = 3) -> list[Listing]:
        articles = deals.articles(per_feed=per_feed)
        self.log(f"Fetched {len(articles)} articles from the wine press")
        listings = [listing for article in articles for listing in self.read(article)]
        self.log(f"Found {len(listings)} wines quoted with a price")
        return listings
