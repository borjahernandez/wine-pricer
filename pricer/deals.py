"""Where wines to price actually come from: real RSS feeds from the wine press.

There is no free live wine-price API, and the deal-aggregator feeds (dealnews and friends) carry
almost no wine. What does exist is the wine press: Wine Enthusiast, Decanter and similar publish
recommendation round-ups whose articles quote a tasting note and a shelf price. That is exactly the
shape this project needs -- a note to price, and a listed price to compare our estimate against.

Verified reachable at the time of writing (200, parseable, items present):
  wineenthusiast.com/feed/     ~10 items, recommendation round-ups
  decanter.com/feed/           ~50 items, reviews and news
Both are editorial rather than commercial, so a given article may quote no price at all; the scanner
agent throws those away.
"""

from dataclasses import dataclass

import feedparser
import requests
from bs4 import BeautifulSoup

FEEDS = (
    "https://www.wineenthusiast.com/feed/",
    "https://www.decanter.com/feed/",
)
AGENT = "Mozilla/5.0 (compatible; wine-pricer/0.1)"
TIMEOUT = 20
MAX_CHARS = 12_000  # a wine article is a few thousand characters; the rest is navigation and ads


@dataclass
class Article:
    """One fetched article: the text an LLM can mine for wines and prices."""

    title: str
    url: str
    summary: str
    body: str

    def as_text(self) -> str:
        return f"{self.title}\n\n{self.summary}\n\n{self.body}"[:MAX_CHARS]


def fetch(url: str) -> str:
    """The readable text of a page. Scripts and styles removed, whitespace collapsed."""
    response = requests.get(url, headers={"User-Agent": AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())


def articles(feeds: tuple[str, ...] = FEEDS, per_feed: int = 5, fetch_bodies: bool = True) -> list[Article]:
    """Recent articles from the wine press. Feeds that are down are skipped, not fatal."""
    found: list[Article] = []
    for url in feeds:
        parsed = feedparser.parse(url, agent=AGENT)
        for entry in parsed.entries[:per_feed]:
            link = entry.get("link", "")
            summary = " ".join(BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(" ").split())
            body = ""
            if fetch_bodies and link:
                try:
                    body = fetch(link)
                except requests.RequestException:
                    body = ""
            found.append(Article(title=entry.get("title", ""), url=link, summary=summary, body=body))
    return found
