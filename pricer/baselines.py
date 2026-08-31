"""The baseline ladder. Nothing clever here, and that is the point.

Every rung returns a plain `Wine -> float` function so it can be dropped straight into
`evaluator.evaluate`, exactly like the LLM predictors later on. Climb the ladder in order and never
accept a model that fails to beat the rung below it.

All the regressors fit on `log1p(price)` and exponentiate back. Fitting on raw dollars lets a handful
of $400 bottles dominate the loss and produces a model that is worse everywhere else.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pricer.items import Wine

METADATA = ["variety", "country", "province"]
NUMERIC = ["vintage", "note_length", "word_count"]


def to_frame(wines: Sequence[Wine]) -> pd.DataFrame:
    """The feature table. Deliberately excludes `points` and `winery` -- see parser.DEFAULT_FIELDS."""
    return pd.DataFrame(
        {
            "note": [wine.description for wine in wines],
            "variety": [wine.variety or "unknown" for wine in wines],
            "country": [wine.country or "unknown" for wine in wines],
            "province": [wine.province or "unknown" for wine in wines],
            "vintage": [wine.vintage or 0 for wine in wines],
            "note_length": [len(wine.description) for wine in wines],
            "word_count": [len(wine.description.split()) for wine in wines],
        }
    )


def log_prices(wines: Sequence[Wine]) -> np.ndarray:
    return np.log1p(np.array([wine.price for wine in wines], dtype=float))


class Model:
    """A fitted pipeline in the `Wine -> float` shape the evaluator expects.

    Callable for one wine at a time, like the LLM predictors, but also able to score a whole split in
    one vectorised call.
    """

    def __init__(self, name: str, pipeline: Pipeline, text_only: bool = False):
        self.__name__ = name
        self.pipeline = pipeline
        self.text_only = text_only  # pipelines that take raw notes rather than the feature table

    def __call__(self, wine: Wine) -> float:
        return float(self.predict_all([wine])[0])

    def predict_all(self, wines: Sequence[Wine]) -> np.ndarray:
        if self.text_only:
            return np.expm1(self.pipeline.predict([wine.description for wine in wines]))
        return np.expm1(self.pipeline.predict(to_frame(wines)))


class ConstantModel:
    """Rung 0. Always guess the geometric mean. Any model that cannot beat this is broken."""

    def __init__(self, train: Sequence[Wine]):
        self.guess = float(np.expm1(log_prices(train).mean()))
        self.__name__ = f"Constant ${self.guess:.0f}"

    def __call__(self, wine: Wine) -> float:
        return self.guess

    def predict_all(self, wines: Sequence[Wine]) -> np.ndarray:
        return np.full(len(wines), self.guess)


def constant(train: Sequence[Wine]) -> ConstantModel:
    return ConstantModel(train)


def metadata_only(train: Sequence[Wine]) -> Model:
    """Rung 1. Grape, country, vintage and note length -- no reading of the note at all.

    This is the number to beat: whatever the language models add, they have to add it on top of
    'Napa Cabernet from 2013' being expensive for reasons that have nothing to do with the prose.
    """
    pipeline = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), METADATA),
                        ("num", StandardScaler(), NUMERIC),
                    ]
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(to_frame(train), log_prices(train))
    return Model("Metadata + Linear Regression", pipeline)


def tfidf(train: Sequence[Wine], max_features: int = 40_000) -> Model:
    """Rung 2. Bag of words over the tasting note, plus the metadata. Ridge on log price.

    Surprisingly strong, because critics telegraph price with vocabulary: 'quaffable' and 'simple'
    are cheap, 'brooding', 'structured' and 'decades' are expensive.
    """
    pipeline = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "note",
                            TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3, sublinear_tf=True),
                            "note",
                        ),
                        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), METADATA),
                        ("num", StandardScaler(), NUMERIC),
                    ]
                ),
            ),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(to_frame(train), log_prices(train))
    return Model("TF-IDF + Ridge", pipeline)


def tfidf_text(train: Sequence[Wine], max_features: int = 40_000) -> Model:
    """The same rung with the note as its only input, for callers who only ever have prose.

    The agents are handed a tasting note and nothing else. Serving them the metadata-aware
    model means fitting on real varieties and regions and then predicting with 'unknown' for all of
    them -- a train/serve skew that cost about 0.2 RMSLE when measured. Better to fit the model that
    matches what the caller can actually supply.
    """
    pipeline = Pipeline(
        [
            (
                "note",
                TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3, sublinear_tf=True),
            ),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit([wine.description for wine in train], log_prices(train))
    return Model("TF-IDF + Ridge (note only)", pipeline, text_only=True)


def lsa_forest(train: Sequence[Wine], components: int = 200, trees: int = 200) -> Model:
    """Rung 3. Compress the note to a dense vector with LSA, then a random forest.

    The non-linear rung. Slower to fit and usually a little worse than Ridge on this data, which is
    a useful result in itself: a linear model over words is hard to beat on short, formulaic text.
    """
    pipeline = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "note",
                            Pipeline(
                                [
                                    ("tfidf", TfidfVectorizer(max_features=40_000, min_df=3, sublinear_tf=True)),
                                    ("svd", TruncatedSVD(n_components=components, random_state=42)),
                                ]
                            ),
                            "note",
                        ),
                        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), METADATA),
                        ("num", StandardScaler(), NUMERIC),
                    ]
                ),
            ),
            ("model", RandomForestRegressor(n_estimators=trees, n_jobs=-1, random_state=42, min_samples_leaf=2)),
        ]
    )
    pipeline.fit(to_frame(train), log_prices(train))
    return Model("LSA + Random Forest", pipeline)
