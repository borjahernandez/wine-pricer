"""The planner: scan the wine press, estimate what each wine should cost, and surface the bargains.

`opportunity = estimate - listed price`. Positive means the press is quoting less than our models
think the wine is worth. Treat it as a way to see the whole pipeline work end to end rather than
investment advice: the estimate carries an RMSLE of roughly 0.5, so it is a fuzzy number.

Findings are remembered in `memory.json`, so a repeated run does not re-report the same wine.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from pricer.agents.agent import Agent
from pricer.agents.messaging import MessagingAgent
from pricer.agents.scanner import Listing, ScannerAgent
from pricer.items import ROOT
from pricer.parser import MAX_PRICE, MIN_PRICE

MEMORY_FILE = ROOT / "memory.json"
THRESHOLD = 15.0  # dollars of estimated underpricing before it is worth a notification


class Opportunity(BaseModel):
    """A scanned wine, what we think it is worth, and the gap."""

    listing: Listing
    estimate: float

    @property
    def discount(self) -> float:
        return self.estimate - self.listing.price

    def summary(self) -> str:
        return (
            f"{self.listing.name}: listed ${self.listing.price:.0f}, "
            f"we estimate ${self.estimate:.0f} (gap ${self.discount:+.0f})\n{self.listing.url}"
        )


class PlanningAgent(Agent):
    name = "Planning Agent"
    colour = "\033[94m"

    def __init__(self, pricer: Agent, scanner: ScannerAgent | None = None, messenger: MessagingAgent | None = None):
        self.pricer = pricer
        self.scanner = scanner or ScannerAgent()
        self.messenger = messenger or MessagingAgent()
        self.log(f"Ready, pricing with {self.pricer.name}")

    def memory(self, path: Path = MEMORY_FILE) -> list[str]:
        return json.loads(path.read_text()) if path.exists() else []

    def remember(self, urls: list[str], path: Path = MEMORY_FILE) -> None:
        path.write_text(json.dumps(sorted(set(self.memory(path)) | set(urls)), indent=2))

    def in_range(self, listing: Listing) -> bool:
        """Our models only ever saw $4-$500 bottles; a $22,500 auction lot is not a bargain, it is
        out of distribution, and the estimate for it is meaningless."""
        if MIN_PRICE <= listing.price <= MAX_PRICE:
            return True
        self.log(f"Skipping {listing.name} at ${listing.price:,.0f}: outside the trained price range")
        return False

    def plan(self, per_feed: int = 3, threshold: float = THRESHOLD) -> list[Opportunity]:
        seen = set(self.memory())
        scanned = [wine for wine in self.scanner.scan(per_feed=per_feed) if self.in_range(wine)]
        listings = [wine for wine in scanned if f"{wine.url}#{wine.name}" not in seen]
        self.log(f"{len(listings)} wines are new since the last run")
        opportunities = [Opportunity(listing=wine, estimate=self.pricer.price(wine.note)) for wine in listings]
        opportunities.sort(key=lambda o: o.discount, reverse=True)
        for opportunity in opportunities:
            if opportunity.discount >= threshold:
                self.messenger.notify("Wine worth a look", opportunity.summary())
        self.remember([f"{o.listing.url}#{o.listing.name}" for o in opportunities])
        return opportunities
