"""Week 6 days 3-4, as a script: fit the baseline ladder and score every rung the same way.

    uv run python scripts/baselines.py

Scores land in results.json, which the LLM notebooks append to, so the whole ladder stays comparable.
"""

import argparse

from pricer.baselines import constant, lsa_forest, metadata_only, tfidf
from pricer.evaluator import Report, leaderboard
from pricer.items import DATA_DIR, Wine

RUNGS = {
    "constant": constant,
    "metadata": metadata_only,
    "tfidf": tfidf,
    "lsa_forest": lsa_forest,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--rungs", nargs="*", default=list(RUNGS), choices=list(RUNGS))
    parser.add_argument("--train-size", type=int, help="subsample the training set, for a quick pass")
    args = parser.parse_args()

    train, _, test = Wine.load_local(args.data)
    if args.train_size:
        train = train[: args.train_size]
    print(f"Fitting on {len(train):,} wines, scoring on the full {len(test):,} test wines\n")

    for name in args.rungs:
        model = RUNGS[name](train)
        guesses = model.predict_all(test)
        report = Report(model.__name__, [wine.label for wine in test], list(guesses), [w.price for w in test])
        print(report.summary())
        report.save()

    print()
    leaderboard()


if __name__ == "__main__":
    main()
