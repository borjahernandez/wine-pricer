"""A Chroma vector store over the training tasting notes, so agents can look up comparable wines.

The embedded text is the tasting note only -- no price, no critic score, no winery. Anything the
model should not know at prediction time must not be in the vector store either, or RAG becomes a
leak with extra steps.
"""

from collections.abc import Sequence

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from pricer.items import ROOT, Wine

CHROMA_DIR = ROOT / "chroma"
COLLECTION = "wines"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 1_000


def encoder(name: str = ENCODER) -> SentenceTransformer:
    return SentenceTransformer(name)


def store(path=CHROMA_DIR) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(path))


def build(wines: Sequence[Wine], name: str = COLLECTION, model: SentenceTransformer | None = None) -> Collection:
    """(Re)build the collection from scratch. ~50k notes takes a few minutes on CPU."""
    client = store()
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    collection = client.create_collection(name)
    model = model or encoder()
    for start in tqdm(range(0, len(wines), BATCH), desc="embedding"):
        batch = wines[start : start + BATCH]
        documents = [wine.description for wine in batch]
        collection.add(
            ids=[str(wine.id) for wine in batch],
            documents=documents,
            embeddings=model.encode(documents).astype(float).tolist(),
            metadatas=[
                {
                    "price": wine.price,
                    "variety": wine.variety or "",
                    "country": wine.country or "",
                    "vintage": wine.vintage or 0,
                }
                for wine in batch
            ],
        )
    return collection


def load(name: str = COLLECTION) -> Collection:
    """Open an existing collection, with a message that says what to run if it is missing."""
    client = store()
    if name not in [c.name for c in client.list_collections()]:
        raise FileNotFoundError(f"No Chroma collection '{name}' in {CHROMA_DIR} -- run scripts/vectors.py first")
    return client.get_collection(name)


def similar(
    note: str,
    collection: Collection,
    model: SentenceTransformer,
    k: int = 5,
) -> tuple[list[str], list[float]]:
    """The k nearest tasting notes and their prices."""
    vector = model.encode([note]).astype(float).tolist()
    found = collection.query(query_embeddings=vector, n_results=k)
    return found["documents"][0], [float(m["price"]) for m in found["metadatas"][0]]


def sample_coordinates(collection: Collection, limit: int = 2_000) -> tuple[np.ndarray, list[float], list[str]]:
    """Embeddings, prices and varieties for a random-ish slice, for the 2D/3D visualisations."""
    found = collection.get(include=["embeddings", "metadatas"], limit=limit)
    return (
        np.array(found["embeddings"]),
        [float(m["price"]) for m in found["metadatas"]],
        [str(m["variety"]) for m in found["metadatas"]],
    )
