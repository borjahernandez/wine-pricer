"""Run the LLM extraction pass over the curated splits, and optionally publish the result.

    uv run python scripts/tasting.py --splits test validation      # cheap, do these first
    uv run python scripts/tasting.py --splits train --limit 5000   # chip away at the big one
    uv run python scripts/tasting.py --push you/wine-pricer        # curated splits + summaries

Resumable: rows already in data/tasting/<split>.jsonl are skipped, so rerun after a rate limit.
"""

import argparse

from pricer.items import DATA_DIR, Wine
from pricer.tasting import attach, run

SPLITS = ("train", "validation", "test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--splits", nargs="*", default=["test", "validation"], choices=list(SPLITS))
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", help="override the provider's default model")
    parser.add_argument("--limit", type=int, help="stop after this many new rows per split")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--push", help="HuggingFace dataset name to push the enriched splits to")
    args = parser.parse_args()

    splits = dict(zip(SPLITS, Wine.load_local(args.data), strict=True))
    for name in args.splits:
        run(
            splits[name],
            split=name,
            provider=args.provider,
            model=args.model,
            workers=args.workers,
            limit=args.limit,
        )

    for name, wines in splits.items():
        matched = attach(wines, name)
        print(f"{name}: {matched:,}/{len(wines):,} wines carry a summary")

    Wine.save_local(splits["train"], splits["validation"], splits["test"], path=args.data)
    if args.push:
        Wine.push_to_hub(args.push, splits["train"], splits["validation"], splits["test"])
        print(f"Pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
