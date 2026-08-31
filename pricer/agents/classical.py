"""The classical agent: the baseline TF-IDF + Ridge model, cached to disk so the app starts fast.

The note-only variant, because an agent is handed prose and nothing else -- see `baselines.tfidf_text`
for why serving the metadata-aware model here would quietly cost accuracy. It is the cheapest member
of the ensemble and the bar the fancier agents must clear.
"""

import pickle
from pathlib import Path

from pricer.agents.agent import Agent
from pricer.baselines import Model, tfidf_text
from pricer.items import ROOT, Wine

MODEL_FILE = ROOT / "data" / "tfidf_ridge_note_only.pkl"


class ClassicalAgent(Agent):
    name = "Classical Agent"
    colour = "\033[36m"

    def __init__(self, path: Path = MODEL_FILE):
        self.log("Loading the TF-IDF + Ridge model")
        if path.exists():
            with open(path, "rb") as handle:
                self.model: Model = pickle.load(handle)
        else:
            self.log(f"No model at {path.name}, fitting one from the training split")
            train, _, _ = Wine.load_local()
            self.model = tfidf_text(train)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as handle:
                pickle.dump(self.model, handle)
        self.log("Ready")

    def price(self, text: str) -> float:
        wine = Wine(description=text, price=0.0, points=0)
        return float(self.model.predict_all([wine])[0])
