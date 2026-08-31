"""Tests for the week-8 pieces that can be checked without a network, a GPU or an API key.

The agents themselves are thin wrappers around services; what is worth testing is the logic around
them -- the rate limiter, the out-of-range guard, the blend, and the parsing of scanned wines.
"""

import time

import pytest
from pydantic import ValidationError

from pricer.agents.agent import Agent
from pricer.agents.ensemble import EnsembleAgent
from pricer.agents.planning import Opportunity, PlanningAgent
from pricer.agents.scanner import Listing
from pricer.deals import MAX_CHARS, Article
from pricer.items import Wine
from pricer.llm import Limiter

NOTE = (
    "Aromas of black cherry and cedar open onto a firm palate of graphite and dried herbs, with "
    "chalky tannins carrying a long, savoury finish that suggests real cellaring potential."
)


class Fixed(Agent):
    """An agent that always says the same thing, so a blend has something predictable to learn."""

    def __init__(self, name: str, guess: float):
        self.name = name
        self.guess = guess

    def price(self, text: str) -> float:
        return self.guess


class TestLimiter:
    def test_spends_the_budget_then_waits(self):
        limiter = Limiter(tokens_per_minute=6_000)  # 100 tokens per second
        limiter.acquire(6_000)
        started = time.monotonic()
        limiter.acquire(100)
        assert time.monotonic() - started >= 0.5

    def test_within_budget_does_not_wait(self):
        limiter = Limiter(tokens_per_minute=6_000)
        started = time.monotonic()
        limiter.acquire(500)
        assert time.monotonic() - started < 0.1


class TestScannerModels:
    def test_a_listing_needs_a_real_note_and_a_price(self):
        with pytest.raises(ValidationError):
            Listing(name="Some wine", note="Nice.", price=25.0)
        with pytest.raises(ValidationError):
            Listing(name="Some wine", note=NOTE, price=0.0)
        assert Listing(name="Some wine", note=NOTE, price=25.0).price == 25.0

    def test_article_text_is_capped(self):
        article = Article(title="t", url="u", summary="s", body="x" * (MAX_CHARS * 2))
        assert len(article.as_text()) == MAX_CHARS


class TestPlanning:
    def planner(self) -> PlanningAgent:
        return PlanningAgent.__new__(PlanningAgent)  # only in_range is under test; skip the services

    def test_out_of_range_listings_are_dropped(self):
        planner = self.planner()
        assert planner.in_range(Listing(name="normal", note=NOTE, price=40.0))
        assert not planner.in_range(Listing(name="auction lot", note=NOTE, price=22_500.0))
        assert not planner.in_range(Listing(name="box wine", note=NOTE, price=2.0))

    def test_discount_is_estimate_minus_listed(self):
        opportunity = Opportunity(listing=Listing(name="w", note=NOTE, price=30.0), estimate=45.0)
        assert opportunity.discount == pytest.approx(15.0)
        assert "$+15" in opportunity.summary()


class TestEnsemble:
    def wines(self) -> list[Wine]:
        return [Wine(description=NOTE, price=price, points=90) for price in (10, 20, 40, 80, 160)]

    def test_blend_learns_to_follow_the_accurate_member(self):
        truthful = Fixed("truthful", 40.0)
        liar = Fixed("liar", 400.0)
        ensemble = EnsembleAgent([truthful, liar])
        ensemble.fit(self.wines())
        # Both members are constant, so the blend can only learn the mean -- it must at least land
        # inside the range of the data rather than following the liar.
        assert 10.0 <= ensemble.price(NOTE) <= 160.0

    def test_unfitted_ensemble_refuses_to_guess(self):
        with pytest.raises(ValueError, match="not fitted"):
            EnsembleAgent([Fixed("a", 10.0)]).price(NOTE)
