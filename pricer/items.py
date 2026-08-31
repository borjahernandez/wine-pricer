"""The Wine datapoint: a tasting note plus the metadata a sommelier would see, and a price."""

from pathlib import Path
from typing import Self

from datasets import Dataset, DatasetDict, Features, Value, load_dataset, load_from_disk
from pydantic import BaseModel

QUESTION = "How much does this bottle of wine cost, to the nearest dollar?"
PREFIX = "Price is $"
# Anchored to the repo, not the working directory, so notebooks and scripts read the same cache.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "curated"

# Declared rather than inferred: a split where every `summary` is still None infers as null, and then
# the splits disagree and the Hub push refuses them.
FEATURES = Features(
    {
        "description": Value("string"),
        "price": Value("float64"),
        "points": Value("int64"),
        "variety": Value("string"),
        "country": Value("string"),
        "province": Value("string"),
        "region": Value("string"),
        "winery": Value("string"),
        "designation": Value("string"),
        "vintage": Value("int64"),
        "taster": Value("string"),
        "title": Value("string"),
        "full": Value("string"),
        "summary": Value("string"),
        "prompt": Value("string"),
        "id": Value("int64"),
    }
)


class Wine(BaseModel):
    """A wine review with a known price.

    `full` is the cleaned text built from the raw review by the parser. `summary` is the shorter,
    structured rewrite produced by an LLM in the preprocessing pass. Whichever of the two is used
    to build `prompt` is the input the models actually see.
    """

    description: str
    price: float
    points: int
    variety: str | None = None
    country: str | None = None
    province: str | None = None
    region: str | None = None
    winery: str | None = None
    designation: str | None = None
    vintage: int | None = None
    taster: str | None = None
    title: str | None = None
    full: str | None = None
    summary: str | None = None
    prompt: str | None = None
    id: int | None = None

    @property
    def label(self) -> str:
        """A short human-readable name, used in evaluation charts."""
        if self.title:
            return self.title
        parts = [str(self.vintage) if self.vintage else None, self.winery, self.variety]
        return " ".join(p for p in parts if p) or self.description[:40]

    def test_prompt(self) -> str:
        """The stored training prompt with the answer removed, for inference."""
        if not self.prompt:
            raise ValueError(f"Wine {self.id} has no prompt -- run pricer.prompts.prepare first")
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.label} = ${self.price}>"

    @staticmethod
    def to_dataset_dict(train: list[Self], val: list[Self], test: list[Self]) -> DatasetDict:
        return DatasetDict(
            {
                "train": Dataset.from_list([w.model_dump() for w in train], features=FEATURES),
                "validation": Dataset.from_list([w.model_dump() for w in val], features=FEATURES),
                "test": Dataset.from_list([w.model_dump() for w in test], features=FEATURES),
            }
        )

    @classmethod
    def from_dataset_dict(cls, ds: DatasetDict) -> tuple[list[Self], list[Self], list[Self]]:
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )

    @staticmethod
    def push_to_hub(
        dataset_name: str,
        train: list[Self],
        val: list[Self],
        test: list[Self],
        private: bool = True,
    ) -> None:
        """Private by default: the tasting notes are Wine Enthusiast's, not ours to republish."""
        Wine.to_dataset_dict(train, val, test).push_to_hub(dataset_name, private=private)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        return cls.from_dataset_dict(load_dataset(dataset_name))

    @staticmethod
    def save_local(train: list[Self], val: list[Self], test: list[Self], path: str | Path = DATA_DIR) -> None:
        """Cache the curated splits on disk, so the baselines can run without a Hub round-trip."""
        Wine.to_dataset_dict(train, val, test).save_to_disk(str(path))

    @classmethod
    def load_local(cls, path: str | Path = DATA_DIR) -> tuple[list[Self], list[Self], list[Self]]:
        return cls.from_dataset_dict(load_from_disk(str(path)))
