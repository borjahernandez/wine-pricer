"""Tests for the parts of the pipeline where a silent bug would quietly ruin every experiment:
parsing rules, leakage, target balance, split reproducibility, and the metrics themselves.
"""

import numpy as np
import pytest

from pricer import prompts
from pricer.baselines import constant, tfidf
from pricer.curate import balance, deduplicate, log_price_bins, split
from pricer.evaluator import Report
from pricer.items import PREFIX, QUESTION, Wine
from pricer.parser import MIN_CHARS, clean, compose, get_vintage, parse

NOTE = (
    "Aromas of black cherry and cedar open onto a firm palate of graphite and dried herbs, with "
    "chalky tannins carrying a long, savoury finish that suggests real cellaring potential."
)


def row(**overrides) -> dict:
    base = {
        "description": NOTE,
        "price": 42.0,
        "points": 91,
        "variety": "Nebbiolo",
        "country": "Italy",
        "province": "Piedmont",
        "region_1": "Barolo",
        "winery": "Vietti",
        "designation": "Rocche",
        "title": "Vietti 2016 Rocche Nebbiolo (Barolo)",
        "taster_name": "Kerin O'Keefe",
    }
    return base | overrides


def wine(price: float, description: str = NOTE, **overrides) -> Wine:
    return Wine(description=description, price=price, points=90, **overrides)


class WordTokenizer:
    """A stand-in for a HuggingFace tokenizer: one token per word, so truncation is checkable."""

    def __init__(self):
        self.vocabulary: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        for word in text.split():
            if word not in self.vocabulary:
                self.vocabulary.append(word)
        return [self.vocabulary.index(word) for word in text.split()]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self.vocabulary[token] for token in tokens)


class TestParser:
    def test_parses_a_good_row(self):
        parsed = parse(row())
        assert parsed is not None
        assert parsed.price == 42.0
        assert parsed.vintage == 2016
        assert parsed.region == "Barolo"
        assert NOTE in parsed.full

    @pytest.mark.parametrize(
        "overrides",
        [
            {"price": None},
            {"price": float("nan")},
            {"price": 2.0},
            {"price": 900.0},
            {"description": None},
            {"description": "Nice wine."},
        ],
    )
    def test_rejects_unusable_rows(self, overrides):
        assert parse(row(**overrides)) is None

    def test_min_chars_is_the_boundary(self):
        assert parse(row(description="a" * (MIN_CHARS - 1))) is None
        assert parse(row(description="a" * MIN_CHARS)) is not None

    def test_clean_collapses_whitespace(self):
        assert clean("  Aromas\n\tof   cedar  ") == "Aromas of cedar"
        assert clean("   ") is None
        assert clean(None) is None

    def test_vintage_prefers_the_title_and_takes_the_last_year(self):
        assert get_vintage("Vietti 2016 Rocche", "released in 2019") == 2016
        assert get_vintage(None, "a blend of 2014 and 2015 fruit") == 2015
        assert get_vintage("Vietti Rocche", "no year here") is None

    def test_compose_hides_price_leaking_fields(self):
        text = compose(parse(row()))
        assert "91" not in text  # points would proxy the price
        assert "Vietti" not in text  # winery would let the model memorise brand prestige
        assert "42" not in text


class TestCuration:
    def test_deduplicate_keeps_first_occurrence_of_a_note(self):
        first, second, other = wine(10), wine(99), wine(20, description=NOTE + " Extra.")
        unique = deduplicate([first, second, other])
        assert unique == [first, other]

    def test_balance_flattens_the_target(self):
        wines = [wine(8) for _ in range(500)] + [wine(300) for _ in range(20)]
        balanced = balance(wines, cap=20)
        counts = np.bincount(log_price_bins(balanced), minlength=1)
        assert counts.max() == 20
        assert len(balanced) == 40

    def test_balance_is_deterministic(self):
        wines = [wine(5 + index) for index in range(200)]
        ids = [id(w) for w in balance(wines, cap=5)]
        assert ids == [id(w) for w in balance(wines, cap=5)]

    def test_split_sizes_and_unique_ids(self):
        wines = [wine(5 + index) for index in range(100)]
        train, val, test = split(wines, val_size=10, test_size=10)
        assert (len(train), len(val), len(test)) == (80, 10, 10)
        assert len({w.id for w in train + val + test}) == 100

    def test_split_test_set_is_stable_when_the_train_set_grows(self):
        wines = [wine(5 + index) for index in range(100)]
        _, _, test = split(wines, val_size=10, test_size=10)
        prices = sorted(w.price for w in test)
        _, _, again = split(wines, val_size=10, test_size=10)
        assert sorted(w.price for w in again) == prices


class TestPrompts:
    def test_training_prompt_carries_the_answer_and_inference_prompt_does_not(self):
        item = parse(row())
        prompts.prepare([item])
        assert QUESTION in item.prompt
        assert item.prompt.endswith(f"{PREFIX}42.00")
        assert item.test_prompt() == prompts.for_inference(item)
        assert "42.00" not in item.test_prompt()

    def test_summary_input_needs_a_summary(self):
        item = parse(row())
        with pytest.raises(ValueError):
            prompts.for_inference(item, use_summary=True)
        item.summary = "Dense, savoury Barolo built for the cellar."
        assert item.summary in prompts.for_inference(item, use_summary=True)

    def test_truncate_keeps_whole_words(self):
        tokenizer = WordTokenizer()
        text = "one two three four five six"
        assert prompts.truncate(text, tokenizer, cutoff=99) == text
        assert prompts.truncate(text, tokenizer, cutoff=3) == "one two"


class TestEvaluator:
    def test_perfect_predictions_score_perfectly(self):
        truths = [10.0, 50.0, 300.0]
        report = Report("oracle", ["a", "b", "c"], truths, truths)
        assert report.mae == 0
        assert report.rmsle == 0
        assert report.r2 == pytest.approx(1.0)
        assert report.hit_rate == 1.0

    def test_negative_guesses_do_not_break_rmsle(self):
        report = Report("bad", ["a"], [-5.0], [20.0])
        assert np.isfinite(report.rmsle)

    def test_hit_rate_allows_ten_dollars_or_twenty_percent(self):
        report = Report("close", ["a", "b", "c"], [14.0, 230.0, 400.0], [5.0, 200.0, 200.0])
        assert report.hit_rate == pytest.approx(2 / 3)


class TestBaselines:
    def test_constant_guesses_the_geometric_mean(self):
        train = [wine(10), wine(90, description=NOTE + " Two."), wine(30, description=NOTE + " Three.")]
        model = constant(train)
        assert model(train[0]) == pytest.approx(np.expm1(np.log1p([10, 90, 30]).mean()), rel=1e-6)

    def test_tfidf_learns_something_from_the_note(self):
        cheap = [wine(6, description=f"Simple juicy jammy everyday quaffer number {i}. " + NOTE) for i in range(30)]
        pricey = [
            wine(300, description=f"Profound structured cellar-worthy monument number {i}. " + NOTE) for i in range(30)
        ]
        model = tfidf(cheap + pricey, max_features=200)
        assert model(cheap[0]) < model(pricey[0])
        assert model.predict_all(cheap + pricey).shape == (60,)
