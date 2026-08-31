"""Load the raw wine reviews from HuggingFace and parse them into Wine objects."""

from datasets import load_dataset
from tqdm.auto import tqdm

from pricer.items import Wine
from pricer.parser import parse

RAW_DATASET = "spawn99/wine-reviews"


def load(dataset_name: str = RAW_DATASET, split: str = "train") -> list[Wine]:
    """Load every usable review. Roughly a third of the raw rows are rejected by the parser."""
    dataset = load_dataset(dataset_name, split=split)
    wines = [parse(row) for row in tqdm(dataset, desc="parsing")]
    wines = [wine for wine in wines if wine is not None]
    print(f"Parsed {len(wines):,} wines from {len(dataset):,} raw rows")
    return wines
