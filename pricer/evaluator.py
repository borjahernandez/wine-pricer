"""The single scoring harness. Every model in this project is judged here and nowhere else.

Four numbers are reported, because on a log-normal target no single one is honest:

- MAE      average absolute error in dollars. Intuitive, but dominated by the expensive tail.
- RMSLE    root mean squared log error. The primary metric: being $20 out on a $15 wine is a real
           mistake, being $20 out on a $400 wine is not, and only a log-space metric agrees.
- R2       fraction of price variance explained, for comparison with the classical baselines.
- Hit rate fraction of guesses within 20% (or $10) of the truth -- the "would I trust this?" number.

Results are appended to results.json so the ladder accumulates across notebooks instead of
having to be re-run from scratch every session.
"""

import json
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from pricer.items import ROOT, Wine

DEFAULT_SIZE = 2000
WORKERS = 5
RESULTS_FILE = ROOT / "results.json"


def post_process(value: float | str) -> float:
    """Pull a number out of whatever a model gave us. LLMs answer '$42.00' or 'about 42 dollars'."""
    if isinstance(value, int | float):
        return float(value)
    cleaned = str(value).replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
    return float(match.group()) if match else 0.0


class Report:
    def __init__(self, title: str, labels: list[str], guesses: list[float], truths: list[float]):
        self.title = title
        self.labels = labels
        self.guesses = np.array(guesses, dtype=float)
        self.truths = np.array(truths, dtype=float)

    @property
    def errors(self) -> np.ndarray:
        return np.abs(self.guesses - self.truths)

    @property
    def mae(self) -> float:
        return float(self.errors.mean())

    @property
    def rmsle(self) -> float:
        clipped = np.clip(self.guesses, 0, None)
        return float(np.sqrt(np.mean((np.log1p(clipped) - np.log1p(self.truths)) ** 2)))

    @property
    def r2(self) -> float:
        residual = float(((self.truths - self.guesses) ** 2).sum())
        total = float(((self.truths - self.truths.mean()) ** 2).sum())
        return 1.0 - residual / total if total else 0.0

    @property
    def hit_rate(self) -> float:
        within = (self.errors < 10) | (self.errors / self.truths < 0.2)
        return float(within.mean())

    def colors(self) -> list[str]:
        result = []
        for error, truth in zip(self.errors, self.truths, strict=True):
            if error < 10 or error / truth < 0.2:
                result.append("green")
            elif error < 25 or error / truth < 0.4:
                result.append("orange")
            else:
                result.append("red")
        return result

    def summary(self) -> str:
        return (
            f"{self.title}: MAE ${self.mae:,.2f} | RMSLE {self.rmsle:.3f} | "
            f"R2 {self.r2 * 100:.1f}% | hits {self.hit_rate * 100:.1f}%"
        )

    def chart(self) -> None:
        """Guess against truth on log axes, with the y=x line a perfect model would sit on."""
        limit = max(self.truths.max(), self.guesses.max()) * 1.1
        plt.figure(figsize=(11, 9))
        plt.scatter(self.truths, np.clip(self.guesses, 0.5, None), s=12, c=self.colors(), alpha=0.7)
        plt.plot([1, limit], [1, limit], color="deepskyblue", lw=1.5, ls="--")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(3, limit)
        plt.ylim(3, limit)
        plt.xlabel("Actual price ($)")
        plt.ylabel("Predicted price ($)")
        plt.title(self.summary())
        plt.grid(alpha=0.2)
        plt.show()

    def save(self, path: Path = RESULTS_FILE) -> None:
        results = json.loads(path.read_text()) if path.exists() else {}
        results[self.title] = {
            "mae": self.mae,
            "rmsle": self.rmsle,
            "r2": self.r2,
            "hit_rate": self.hit_rate,
            "size": len(self.truths),
        }
        path.write_text(json.dumps(results, indent=2, sort_keys=True))


class Tester:
    def __init__(
        self,
        predictor: Callable[[Wine], float | str],
        data: Sequence[Wine],
        title: str | None = None,
        size: int = DEFAULT_SIZE,
        workers: int = WORKERS,
        verbose: bool = True,
    ):
        self.predictor = predictor
        self.data = data
        self.title = title or self.make_title(predictor)
        self.size = min(size, len(data))
        self.workers = workers
        self.verbose = verbose

    @staticmethod
    def make_title(predictor: Callable) -> str:
        return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    def run(self) -> Report:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            guesses = list(
                tqdm(
                    pool.map(lambda wine: post_process(self.predictor(wine)), self.data[: self.size]),
                    total=self.size,
                    desc=self.title,
                )
            )
        truths = [wine.price for wine in self.data[: self.size]]
        labels = [wine.label for wine in self.data[: self.size]]
        report = Report(self.title, labels, guesses, truths)
        if self.verbose:
            print(report.summary())
        return report


def evaluate(
    predictor: Callable[[Wine], float | str],
    data: Sequence[Wine],
    title: str | None = None,
    size: int = DEFAULT_SIZE,
    workers: int = WORKERS,
    chart: bool = True,
    save: bool = True,
) -> Report:
    report = Tester(predictor, data, title=title, size=size, workers=workers).run()
    if save:
        report.save()
    if chart:
        report.chart()
    return report


def leaderboard(path: Path = RESULTS_FILE) -> None:
    """Print every model scored so far, best RMSLE first."""
    if not path.exists():
        print("No results yet")
        return
    results = json.loads(path.read_text())
    ordered = sorted(results.items(), key=lambda kv: kv[1]["rmsle"])
    print(f"{'model':<34}{'MAE':>10}{'RMSLE':>9}{'R2':>8}{'hits':>8}{'n':>7}")
    for name, scores in ordered:
        print(
            f"{name[:33]:<34}{scores['mae']:>9,.2f}{scores['rmsle']:>9.3f}"
            f"{scores['r2'] * 100:>7.1f}%{scores['hit_rate'] * 100:>7.1f}%{scores['size']:>7,}"
        )


def rmsle_of(guesses: Sequence[float], truths: Sequence[float]) -> float:
    """RMSLE for models scored outside the Tester, e.g. scikit-learn on the full validation set."""
    guessed = np.clip(np.array(guesses, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(guessed) - np.log1p(np.array(truths, dtype=float))) ** 2)))
