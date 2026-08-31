"""The ensemble: a linear regression over what the members say, fitted on the validation split.

Members disagree in useful ways -- the classical model reads vocabulary, the neighbours agent reads
the market, the frontier agent reads meaning -- so a weighted blend usually beats all of them. The
blend is fitted in log space, like every other model here, and includes min/max of the members so it
can learn to distrust an outlier.
"""

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from pricer.agents.agent import Agent
from pricer.items import ROOT, Wine

WEIGHTS_FILE = ROOT / "data" / "ensemble.pkl"


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    colour = "\033[33m"

    def __init__(self, members: Sequence[Agent], weights: LinearRegression | None = None):
        self.members = list(members)
        self.log(f"Blending {', '.join(agent.name for agent in self.members)}")
        self.weights = weights

    def features(self, text: str) -> np.ndarray:
        guesses = np.log1p([max(agent.price(text), 0.0) for agent in self.members])
        return np.concatenate([guesses, [guesses.min(), guesses.max()]])

    def fit(self, wines: Sequence[Wine]) -> LinearRegression:
        """Fit on validation wines -- never on train, where the members have already seen the answer."""
        self.log(f"Fitting blend weights on {len(wines):,} wines")
        rows = np.array([self.features(wine.description) for wine in wines])
        target = np.log1p([wine.price for wine in wines])
        self.weights = LinearRegression().fit(rows, target)
        for agent, weight in zip(self.members, self.weights.coef_, strict=False):
            self.log(f"  {agent.name}: {weight:+.3f}")
        return self.weights

    def price(self, text: str) -> float:
        if self.weights is None:
            raise ValueError("Ensemble is not fitted -- call fit(val) or load() first")
        guess = float(np.expm1(self.weights.predict([self.features(text)])[0]))
        self.log(f"Estimated ${guess:.2f}")
        return guess

    def save(self, path: Path = WEIGHTS_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self.weights, handle)

    def load(self, path: Path = WEIGHTS_FILE) -> None:
        with open(path, "rb") as handle:
            self.weights = pickle.load(handle)
