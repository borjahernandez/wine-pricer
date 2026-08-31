# wine-pricer

*How much is that bottle?* Given a tasting note and a little geography, predict the price.

A playground built along the lines of weeks 6-8 of Ed Donner's LLM engineering course, but on wine
reviews instead of Amazon products: curate a big messy dataset, establish classical baselines,
fine-tune an open model against them, then wrap the whole thing in retrieval and agents.

The point is the experiments, not the leaderboard. Every stage has knobs worth turning, and every
model is scored by the same `Report`, so anything you try is directly comparable to everything else.

## The data

[`spawn99/wine-reviews`](https://huggingface.co/datasets/spawn99/wine-reviews) — Wine Enthusiast
tasting notes with a critic score, a price, and geography. It is a merge of two Kaggle scrapes, which
is what makes it a good exercise: 196,630 raw rows become 179,200 parseable wines, of which only
**125,322 have a distinct tasting note**. Leave the duplicates in and the same wine lands in train and
test, and every model looks better than it is.

Prices are log-normal: median $25, tail to $500. Trained as-is, a model learns to always guess $25.
`pricer/curate.py` caps how many wines each of 16 log-price bins may contribute, which flattens the
target at the cost of volume — the default `cap=6000` keeps 53,895 wines (49,895 train / 2,000
validation / 2,000 test).

Two fields are deliberately hidden from the models by default:

| field | why it is hidden |
| --- | --- |
| `points` | the critic's score is a strong price proxy, and you don't have it when smelling a glass |
| `winery` | brand prestige lets a model recall the label instead of reading the note |

Both make good ablations — pass them to `pricer.parser.compose` and measure what they are worth.

## Where things are

```
pricer/
  items.py      the Wine datapoint, prompt building, Hub and local persistence
  parser.py     raw row -> Wine, plus the filtering rules and the composed model input
  curate.py     deduplicate, balance the log-price distribution, split
  evaluator.py  Report / Tester: MAE, RMSLE, R2, hit rate, charts, results.json leaderboard
  baselines.py  the classical ladder
  llm.py        one OpenAI-compatible client for Groq / OpenAI / Ollama
  tasting.py    LLM extraction pass: tasting note -> structured sommelier features + summary
notebooks/      the same material with charts, one notebook per stage
scripts/        the same material as CLIs, for long runs
tests/          the pipeline invariants: parsing rules, leakage, balance, split, metrics
```

## Running it

```bash
uv sync
uv run python scripts/curate.py --cap 6000     # ~5 min, downloads and caches the splits
uv run python scripts/baselines.py             # fits the ladder, writes results.json
uv run pytest
```

Then open `notebooks/week6_curate.ipynb` and `notebooks/week6_baselines.ipynb`.

The LLM pass needs a key in `.env` (copy `.env.example`):

```bash
uv run python scripts/tasting.py --splits test validation   # cheap; the train split is the long one
```

## Metrics

RMSLE is the headline number: the target is log-normal, so what matters is *relative* error — being
$20 out on a $25 bottle is a disaster, on a $400 bottle it is noise. Alongside it, MAE in dollars for
intuition, R² for reference (it will be ugly and negative for weak models, because squared dollar
error is dominated by the expensive tail), and a hit rate: within $10 or 20% of the true price.

Where the classical ladder lands today, on the 2,000-wine test split:

| model | MAE | RMSLE | R² | hits |
| --- | --- | --- | --- | --- |
| Constant $31 (geometric mean) | $26.72 | 0.790 | -9.0% | 27.9% |
| Metadata + linear regression | $19.31 | 0.554 | 30.0% | 50.1% |
| TF-IDF + Ridge | $17.09 | **0.479** | 44.4% | 55.9% |
| LSA + random forest | $18.36 | 0.527 | 34.7% | 52.9% |

That TF-IDF row is the number to beat.

## Experiments to try

**Data (week 6)**
- Re-curate at `cap=20_000` (closer to the raw distribution) and compare RMSLE on the expensive half
  of the test set. Does the extra volume help, or just re-teach the prior?
- Add `points` to the composed text and measure the jump. That gap is the value of the critic score.
- Predict `points` instead of `price` — same harness, far less skew.
- Dedup harder: near-duplicate notes (same winery, one word changed) still leak.

**Models (week 7)**
- Fine-tune an open 7-8B model with QLoRA on the tasting-note prompts and put it on the same
  leaderboard.
- Full note vs. the LLM-extracted summary as the input — is the flowery prose worth its tokens?
- Frontier models zero-shot, few-shot, and with retrieved neighbours, for the price of a few cents.

**Agents (week 8)**
- RAG over the tasting notes: retrieve the 5 nearest notes and their prices, and let a frontier model
  reason from the comparables.
- Ensemble the specialist, the frontier model, and the retrieval model with a linear blend.
- A "sommelier" agent that goes the other way: given a budget and a mood, recommend a bottle.

## Status

Week 6 is done and tested. Weeks 7 and 8 are next; see the repo's task list in the PR description.
