"""RAG: retrieve comparable tasting notes with their prices, then ask a frontier model for a number.

The retrieved notes come from the training split only, so nothing from test leaks into the context.
"""

import re

from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from pricer import vectors
from pricer.agents.agent import Agent
from pricer.llm import chat, client_for

SYSTEM = "You estimate wine prices. Reply with a dollar amount and nothing else, e.g. 'Price is $34.00'."
NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


class FrontierAgent(Agent):
    name = "Frontier Agent"
    colour = "\033[34m"

    def __init__(
        self,
        collection: Collection | None = None,
        encoder: SentenceTransformer | None = None,
        provider: str = "groq",
        model: str | None = None,
        reasoning_effort: str = "low",
        k: int = 5,
    ):
        self.log(f"Setting up with {provider}")
        self.client, default_model = client_for(provider)
        self.model = model or default_model
        self.reasoning_effort = reasoning_effort
        self.collection = collection or vectors.load()
        self.encoder = encoder or vectors.encoder()
        self.k = k
        self.log("Ready")

    def context(self, notes: list[str], prices: list[float]) -> str:
        lines = ["Here are tasting notes for wines whose prices are known:\n"]
        lines += [f"{note}\nPrice is ${price:.2f}\n" for note, price in zip(notes, prices, strict=True)]
        return "\n".join(lines)

    def messages(self, text: str) -> list[dict[str, str]]:
        notes, prices = vectors.similar(text, self.collection, self.encoder, k=self.k)
        self.log(f"Retrieved {len(notes)} comparable wines priced ${min(prices):.0f}-${max(prices):.0f}")
        user = f"{self.context(notes, prices)}\nNow estimate the price of this wine:\n{text}\n\nPrice is $"
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def price(self, text: str) -> float:
        reply = chat(
            self.client,
            self.model,
            self.messages(text),
            max_tokens=200,
            temperature=0,
            extra_body={"reasoning_effort": self.reasoning_effort},
        )
        match = NUMBER.findall(reply.replace("$", "").replace(",", ""))
        guess = float(match[-1]) if match else 0.0
        self.log(f"Estimated ${guess:.2f}")
        return guess
