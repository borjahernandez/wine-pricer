"""Deduplicate, rebalance and split the parsed wines.

Two problems to fix before training:

1. The raw dataset is a merge of two scrapes, so ~17% of rows are exact duplicates and many more
   share a tasting note. Left in, they leak between train and test.
2. Prices are log-normal -- half the wines cost under $25, and the tail runs to $500. A model trained
   on that distribution learns to always guess $25, which scores well on MSE and is useless. We
   cap how many wines each log-price bin may contribute, which flattens the target distribution.

Capping trades volume for balance and there is no free lunch: the most expensive bin holds only ~80
wines, so a perfectly flat set would be tiny. `CAP = 10_000` over 16 bins keeps a bit over half the
data while pulling the distribution closer to flat. Both knobs are worth an experiment -- train at
`cap=20_000` (nearly the raw distribution) and compare RMSLE on the expensive end of the test set.

Order matters, hence `holdout` before `balance`: val and test come out of the deduplicated pool, so
the cap only ever changes the training set. Balancing first would rebalance the test set too, and two
caps would be scored on two different exam papers -- an unbalanced test set is also the honest one,
since the wines you meet in a shop are not uniform in price.
"""

import random
from collections import Counter, defaultdict

import numpy as np

from pricer.items import Wine
from pricer.parser import MAX_PRICE, MIN_PRICE

SEED = 42
BINS = 16
CAP = 10_000


def deduplicate(wines: list[Wine]) -> list[Wine]:
    """Drop wines sharing a tasting note, keeping the first occurrence."""
    seen: set[str] = set()
    unique = [wine for wine in wines if not (wine.description in seen or seen.add(wine.description))]
    print(f"{len(unique):,} wines after dropping {len(wines) - len(unique):,} duplicate tasting notes")
    return unique


def log_price_bins(wines: list[Wine], bins: int = BINS) -> np.ndarray:
    """Assign each wine to one of `bins` equal-width bins in log price."""
    log_prices = np.log1p(np.array([wine.price for wine in wines]))
    edges = np.linspace(np.log1p(MIN_PRICE), np.log1p(MAX_PRICE) + 1e-9, bins + 1)
    return np.clip(np.digitize(log_prices, edges) - 1, 0, bins - 1)


def price_histogram(wines: list[Wine], bins: int = BINS) -> None:
    """Print the log-price bin occupancy, the thing `balance` is trying to flatten."""
    counts = Counter(log_price_bins(wines, bins).tolist())
    edges = np.expm1(np.linspace(np.log1p(MIN_PRICE), np.log1p(MAX_PRICE), bins + 1))
    for bin_index in range(bins):
        count = counts.get(bin_index, 0)
        bar = "#" * round(40 * count / max(counts.values()))
        print(f"${edges[bin_index]:>6,.0f}-{edges[bin_index + 1]:>6,.0f} {count:>7,} {bar}")


def balance(wines: list[Wine], cap: int = CAP, bins: int = BINS, seed: int = SEED) -> list[Wine]:
    """Keep at most `cap` wines from each log-price bin, chosen at random.

    Cheap wines are plentiful so most are dropped; the expensive tail is kept in full.
    """
    assignments = log_price_bins(wines, bins)
    by_bin: dict[int, list[Wine]] = defaultdict(list)
    for wine, bin_index in zip(wines, assignments.tolist(), strict=True):
        by_bin[bin_index].append(wine)

    rng = random.Random(seed)
    sample = []
    for bin_index in sorted(by_bin):
        group = by_bin[bin_index]
        rng.shuffle(group)
        sample.extend(group[:cap])

    before, after = [w.price for w in wines], [w.price for w in sample]
    print(
        f"Capped at {cap:,} per bin: {len(sample):,} of {len(wines):,} wines, "
        f"median price ${np.median(after):,.0f} (was ${np.median(before):,.0f})"
    )
    return sample


def holdout(
    wines: list[Wine], val_size: int = 2_000, test_size: int = 2_000, seed: int = SEED
) -> tuple[list[Wine], list[Wine], list[Wine]]:
    """Set val and test aside, returning the pool that `balance` then turns into a training set.

    Ids are assigned here, over the whole deduplicated set, so a wine keeps the same id whatever the
    cap does to the pool around it.
    """
    shuffled = list(wines)
    random.Random(seed).shuffle(shuffled)
    for index, wine in enumerate(shuffled):
        wine.id = index
    test = shuffled[-test_size:]
    val = shuffled[-(test_size + val_size) : -test_size]
    pool = shuffled[: -(test_size + val_size)]
    print(f"pool={len(pool):,} val={len(val):,} test={len(test):,}")
    return pool, val, test
