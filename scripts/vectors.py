"""Build the Chroma vector store from the curated training split.

uv run python scripts/vectors.py            # all 82,786 training notes
uv run python scripts/vectors.py --limit 5000
"""

import argparse

from pricer import vectors
from pricer.items import Wine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="embed only the first N training wines")
    parser.add_argument("--name", default=vectors.COLLECTION)
    args = parser.parse_args()

    train, _, _ = Wine.load_local()
    wines = train[: args.limit] if args.limit else train
    collection = vectors.build(wines, name=args.name)
    print(f"{collection.name}: {collection.count():,} tasting notes in {vectors.CHROMA_DIR}")


if __name__ == "__main__":
    main()
