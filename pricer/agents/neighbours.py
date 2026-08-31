"""Retrieval only, no language model: price a wine as the geometric mean of its nearest neighbours.

Worth building before the RAG agent, because it separates two questions that are easy to confuse:
how much does retrieval know, and how much does the language model add on top of it?
"""

import numpy as np
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from pricer import vectors
from pricer.agents.agent import Agent


class NeighboursAgent(Agent):
    name = "Neighbours Agent"
    colour = "\033[35m"

    def __init__(self, collection: Collection | None = None, model: SentenceTransformer | None = None, k: int = 8):
        self.log("Connecting to the Chroma store")
        self.collection = collection or vectors.load()
        self.model = model or vectors.encoder()
        self.k = k
        self.log(f"Ready with {self.collection.count():,} notes")

    def price(self, text: str) -> float:
        _, prices = vectors.similar(text, self.collection, self.model, k=self.k)
        # Geometric mean: prices are log-normal, so the arithmetic mean would drift upwards.
        return float(np.expm1(np.log1p(prices).mean()))
