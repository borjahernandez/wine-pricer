"""Load the raw wine reviews from HuggingFace and parse them into Wine objects."""

from datasets import concatenate_datasets, load_dataset
from tqdm.auto import tqdm

from pricer.items import Wine
from pricer.parser import parse

RAW_DATASET = "spawn99/wine-reviews"


def load(dataset_name: str = RAW_DATASET) -> list[Wine]:
    """Load every usable review. Roughly a third of the raw rows are rejected by the parser.

    The upstream train/validation/test split is one arbitrary partition of a single scrape, so it is
    concatenated and re-split here. Taking only `train` would throw away 84,271 reviews and hide the
    duplicate notes that span the upstream splits from `deduplicate`.
    """
    dataset = concatenate_datasets(list(load_dataset(dataset_name).values()))
    wines = [parse(row) for row in tqdm(dataset, desc="parsing")]
    wines = [wine for wine in wines if wine is not None]
    print(f"Parsed {len(wines):,} wines from {len(dataset):,} raw rows")
    return wines
