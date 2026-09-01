# wine-pricer

*How much is that bottle?* Given a tasting note and a little geography, predict the price.

A playground for LLM engineering on real, messy data: curate a big scraped dataset, establish
classical baselines, fine-tune an open model against them, then wrap the whole thing in retrieval and
agents.

The point is the experiments, not the leaderboard. Every stage has knobs worth turning, and every
model is scored by the same `Report`, so anything you try is directly comparable to everything else.

## The data

[`spawn99/wine-reviews`](https://huggingface.co/datasets/spawn99/wine-reviews) — Wine Enthusiast
tasting notes with a critic score, a price, and geography. It is a merge of two Kaggle scrapes, which
is what makes it a good exercise: the three upstream splits are one arbitrary partition of a single
scrape, so they are concatenated and re-split here — 280,901 raw rows become 255,856 parseable wines,
of which only **155,262 have a distinct tasting note**. Leave the duplicates in and the same wine
lands in train and test, and every model looks better than it is (concatenating first also exposes the
100,594 duplicates that span the upstream splits).

Prices are log-normal: median $25, tail to $500. Trained as-is, a model learns to always guess $25.
`pricer/curate.py` caps how many wines each of 16 log-price bins may contribute, which flattens the
target at the cost of volume — the default `cap=10_000` keeps 82,786 training wines out of a 151,262
pool.

Order matters: `holdout` takes the 2,000 validation and 2,000 test wines out of the deduplicated pool
*before* `balance` touches it. So the cap is purely a training-pool knob, and every run is scored on
the same fixed, unbalanced test set — which is also the honest one, since the wines in a shop are not
uniform in price. Balance first and each cap gets its own exam paper, mixing "more training data" with
"easier test set" in every comparison.

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
  curate.py     deduplicate, hold out val/test, balance the training pool's log-price distribution
  evaluator.py  Report / Tester: MAE, RMSLE, R2, hit rate, charts, results.json leaderboard
  baselines.py  the classical ladder
  llm.py        one OpenAI-compatible client, plus the rate limiter every LLM call goes through
  tasting.py    LLM extraction pass: tasting note -> structured sommelier features + summary
  prompts.py    the one canonical prompt layout, shared by training and inference
  vectors.py    Chroma store over the tasting notes, for retrieval
  deals.py      the wine-press RSS feeds the scanner reads
  agents/       classical, neighbours, frontier (RAG), specialist (QLoRA), ensemble,
                scanner, messaging, planning
app.py          the Gradio front end
notebooks/      1_curate_and_explore, 2_baseline_ladder, 3_prompts_and_tokens,
                4_qlora_finetune_colab, 5_retrieval_and_rag, 6_agent_framework
scripts/        the same material as CLIs, for long runs
tests/          the pipeline invariants: parsing rules, leakage, balance, holdout, metrics, agents
```

## Running it

```bash
uv sync
uv run python scripts/curate.py --cap 10000    # ~10 min, downloads and caches the splits
uv run python scripts/baselines.py             # fits the ladder, writes results.json
uv run pytest
```

Then open `notebooks/1_curate_and_explore.ipynb` and `notebooks/2_baseline_ladder.ipynb`.

Anything that talks to a model needs a key in `.env` (copy `.env.example`):

```bash
uv run python scripts/tasting.py --splits test          # structured features from each note
uv run python scripts/vectors.py                       # embeds 82,786 notes into Chroma, ~5 min
uv run python scripts/plan.py --pricer classical       # scan the wine press and price what it finds
uv run python app.py                                   # the Gradio app on :7860
```

The fine-tune itself runs in Colab: `notebooks/4_qlora_finetune_colab.ipynb` (4-bit Qwen2.5-3B + LoRA),
streaming loss, learning rate and gradient norms to Weights & Biases — add `WANDB_API_KEY` as a Colab
secret alongside `HF_TOKEN`.
`notebooks/3_prompts_and_tokens.ipynb` builds the prompts and pushes the dataset to the Hub first.

**Budget the free Groq tier before planning any LLM experiment.** It allows 8,000 tokens a minute and
**200,000 a day**, which works out at roughly six wines a minute and a few hundred wines a day. So:
the tasting-note pass over all 82,786 training wines is months on the free tier, and any evaluation
of the frontier agent is a sample of a hundred-odd wines, not the full test split. `pricer/llm.py`
paces calls against the per-minute budget instead of failing, raises `DailyLimitReached` when the
daily one is gone (no amount of retrying fixes that), and `scripts/tasting.py` resumes, so long runs
can be interrupted and restarted the next day. A paid tier removes all of this.

## Metrics

RMSLE is the headline number: the target is log-normal, so what matters is *relative* error — being
$20 out on a $25 bottle is a disaster, on a $400 bottle it is noise. Alongside it, MAE in dollars for
intuition, R² for reference (it will be ugly and negative for weak models, because squared dollar
error is dominated by the expensive tail), and a hit rate: within $10 or 20% of the true price.

Where the classical ladder lands today, on the 2,000-wine test split:

| model | MAE | RMSLE | R² | hits |
| --- | --- | --- | --- | --- |
| Constant $30 (geometric mean) | $18.48 | 0.634 | -2.4% | 39.1% |
| Metadata + linear regression | $14.48 | 0.489 | 23.9% | 57.6% |
| TF-IDF + Ridge | $12.82 | **0.423** | 38.3% | 63.6% |
| LSA + random forest | $13.42 | 0.454 | 31.9% | 60.0% |

That TF-IDF row is the number to beat. Don't read these against numbers from a balanced test set: an
unbalanced test set is dominated by cheap wines, where the models are strongest, so every row here
looks better than the same model scored on a flattened split. The ranking is what transfers.

The retrieval and LLM agents are scored on a 100-wine sample of the same split (the frontier agent costs a network
call per wine), so read them against each other rather than against the rows above:

| agent | MAE | RMSLE | hits | n |
| --- | --- | --- | --- | --- |
| Classical, note only | $14.40 | **0.454** | 57.0% | 100 |
| Neighbours, retrieval only (k=8) | $17.55 | 0.588 | 53.0% | 100 |
| Frontier (RAG + LLM) | — | — | — | daily token budget spent; rerun tomorrow |

Two lessons already: retrieval on its own beats guessing the mean but loses to bag-of-words, and the
note-only model matters — serving the metadata-aware pipeline a note with `variety='unknown'` cost
about 0.16 RMSLE when measured. Fitting a model on features you cannot supply at inference costs more
than the features are worth.

## Experiments to try

**Data and curation**
- Re-curate at `cap=20_000` (closer to the raw distribution) and compare RMSLE on the expensive half
  of the test set. Does the extra volume help, or just re-teach the prior?
- Add `points` to the composed text and measure the jump. That gap is the value of the critic score.
- Predict `points` instead of `price` — same harness, far less skew.
- Dedup harder: near-duplicate notes (same winery, one word changed) still leak.

**Models and fine-tuning**
- Fine-tune an open 7-8B model with QLoRA on the tasting-note prompts and put it on the same
  leaderboard.
- Full note vs. the LLM-extracted summary as the input — is the flowery prose worth its tokens?
- Frontier models zero-shot, few-shot, and with retrieved neighbours, for the price of a few cents.

**Retrieval and agents**
- Sweep `k` in `NeighboursAgent`, and weight the neighbours by similarity rather than flat.
- Retrieve on the LLM summary instead of the full note and see which finds better comparables.
- Add the fine-tuned specialist to the ensemble and refit the blend — does it dominate the others?
- Have the frontier agent return a range and feed its width to the blend as an uncertainty feature.
- A "sommelier" agent that goes the other way: given a budget and a mood, recommend a bottle.

### Where the scanner's wines come from

There is no free live wine-price API, and the deal aggregators carry almost no wine (checked:
dealnews' grocery feeds are 25 items of camping gear). What does exist is the wine press —
[Wine Enthusiast](https://www.wineenthusiast.com/feed/) and [Decanter](https://www.decanter.com/feed/)
publish round-ups that quote a tasting note *and* a shelf price, which is exactly the pair this
project needs. `pricer/deals.py` fetches those, and the scanner agent structures them with an LLM.

They are editorial, so expect one or two priced wines per handful of articles, and expect the
occasional $22,500 auction lot — the planner drops anything outside the $4-$500 range the models were
trained on, because an estimate for it would be meaningless.

## Status

Every stage above is built and tested end to end, apart from the QLoRA fine-tune itself, which needs a GPU:
run `notebooks/4_qlora_finetune_colab.ipynb` in Colab, push the adapter, and `SpecialistAgent` picks it up
(set `WINE_ADAPTER` if you name it something else).
