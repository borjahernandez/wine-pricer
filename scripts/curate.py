"""Curation as a script: load the raw reviews, clean them, balance the prices, split, cache.

    uv run python scripts/curate.py --cap 10000

The notebook `notebooks/1_curate_and_explore.ipynb` walks through the same steps with the charts.
"""

import argparse

from pricer.curate import CAP, balance, deduplicate, holdout, price_histogram
from pricer.items import DATA_DIR, Wine
from pricer.loaders import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=CAP, help="max wines per log-price bin")
    parser.add_argument("--out", default=DATA_DIR, help="where to cache the curated splits")
    parser.add_argument("--push", help="HuggingFace dataset name to push to, e.g. you/wine_lite")
    args = parser.parse_args()

    wines = deduplicate(load())
    print("\nRaw price distribution:")
    price_histogram(wines)

    pool, val, test = holdout(wines)
    train = balance(pool, cap=args.cap)
    print("\nBalanced training price distribution:")
    price_histogram(train)
    Wine.save_local(train, val, test, path=args.out)
    print(f"\nCached to {args.out}")

    if args.push:
        Wine.push_to_hub(args.push, train, val, test)
        print(f"Pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
