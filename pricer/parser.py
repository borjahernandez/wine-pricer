"""Turn a raw `spawn99/wine-reviews` row into a Wine, or reject it.

The raw dataset is a merge of two Kaggle scrapes of Wine Enthusiast, so it carries exact duplicate
rows, missing prices, and a `title` column that is null for roughly half the rows. Deduplication is
handled in `curate.py`; this module handles per-row cleaning and filtering.
"""

import math
import re

from pricer.items import Wine

MIN_CHARS = 100
MIN_PRICE = 4
MAX_PRICE = 500

# Fields included in the text a model sees. `points` is deliberately absent: the critic's score is
# highly predictive of price and is not something you know when smelling a glass. `winery` is absent
# because brand prestige lets a model recall the label instead of reading the note. Both make good
# ablations -- pass them explicitly to `compose` to measure how much they are worth.
DEFAULT_FIELDS = ("vintage", "variety", "country", "province", "region", "note")

VINTAGE_PATTERN = re.compile(r"\b(19[3-9]\d|20[0-2]\d)\b")


def clean(text: str | None) -> str | None:
    if not text:
        return None
    collapsed = re.sub(r"\s+", " ", str(text)).strip()
    return collapsed or None


def get_vintage(title: str | None, description: str | None) -> int | None:
    """Pull the vintage year out of the title, e.g. 'Antichi Vinai 1877 2013 Pietralava Red (Etna)'.

    Titles embed both a founding year in the winery name and the vintage, so take the last match.
    Falls back to the tasting note, which often mentions the vintage for older wines.
    """
    for source in (title, description):
        if source:
            matches = VINTAGE_PATTERN.findall(source)
            if matches:
                return int(matches[-1])
    return None


def compose(wine: Wine, fields: tuple[str, ...] = DEFAULT_FIELDS) -> str:
    """Render a Wine as the text a model reads, one `Label: value` line per requested field."""
    rendered = {
        "note": f"Tasting note: {wine.description}",
        "region": f"Region: {wine.region or wine.province}" if (wine.region or wine.province) else None,
        "points": f"Critic score: {wine.points}/100",
        "vintage": f"Vintage: {wine.vintage}" if wine.vintage else None,
        "variety": f"Variety: {wine.variety}" if wine.variety else None,
        "country": f"Country: {wine.country}" if wine.country else None,
        "province": f"Province: {wine.province}" if wine.province else None,
        "winery": f"Winery: {wine.winery}" if wine.winery else None,
        "designation": f"Designation: {wine.designation}" if wine.designation else None,
    }
    return "\n".join(line for field in fields if (line := rendered[field]))


def parse(row: dict) -> Wine | None:
    """Build a Wine from a raw row, or return None if the row is not usable."""
    price = row.get("price")
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return None
    price = float(price)
    if not MIN_PRICE <= price <= MAX_PRICE:
        return None

    description = clean(row.get("description"))
    if not description or len(description) < MIN_CHARS:
        return None

    title = clean(row.get("title"))
    wine = Wine(
        description=description,
        price=price,
        points=int(row["points"]),
        variety=clean(row.get("variety")),
        country=clean(row.get("country")),
        province=clean(row.get("province")),
        region=clean(row.get("region_1")),
        winery=clean(row.get("winery")),
        designation=clean(row.get("designation")),
        vintage=get_vintage(title, description),
        taster=clean(row.get("taster_name")),
        title=title,
    )
    wine.full = compose(wine)
    return wine
