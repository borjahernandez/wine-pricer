"""Generate the project notebooks from plain-Python cell definitions.

Notebooks are painful to write and review as JSON, so the source of truth lives here: each notebook
is a list of (kind, source) cells, and this script writes clean, output-free .ipynb files.

    uv run python scripts/build_notebooks.py
"""

from pathlib import Path

import nbformat

NOTEBOOKS = Path("notebooks")

CURATION = [
    (
        "md",
        """# Meet the wines: curation and exploration

Goal: understand the raw data well enough to know what a good price prediction would even mean.

The dataset is [`spawn99/wine-reviews`](https://huggingface.co/datasets/spawn99/wine-reviews):
Wine Enthusiast tasting notes with a critic score (`points`), a price, and geography. Two Kaggle
scrapes merged together, so expect duplicates and missing prices.""",
    ),
    (
        "code",
        """from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from pricer.curate import balance, deduplicate, price_histogram, split
from pricer.items import Wine
from pricer.loaders import RAW_DATASET
from pricer.parser import compose, parse

raw = load_dataset(RAW_DATASET, split="train")
raw""",
    ),
    ("md", "### One row, in full"),
    (
        "code",
        """for key, value in raw[0].items():
    print(f"{key:>14}: {value}")""",
    ),
    (
        "md",
        """### How much of the raw data is usable?

`parse` rejects a row when the price is missing, the price is outside \\$4-\\$500, or the tasting note
is under 100 characters. Everything else becomes a `Wine`.""",
    ),
    (
        "code",
        """parsed = [parse(row) for row in raw]
wines = [wine for wine in parsed if wine]
print(f"{len(raw):,} raw rows -> {len(wines):,} usable wines ({len(wines) / len(raw):.0%})")""",
    ),
    (
        "md",
        """### Duplicates

The merge left the same tasting note in the file many times over. Left alone, the identical wine
appears in both train and test, and every model looks better than it is.""",
    ),
    (
        "code",
        """notes = Counter(wine.description for wine in wines)
print(f"{len(notes):,} distinct notes among {len(wines):,} wines")
for note, count in notes.most_common(3):
    print(f"\\n{count} copies: {note[:110]}...")""",
    ),
    ("code", "wines = deduplicate(wines)"),
    (
        "md",
        """### The target

Price is log-normal. The median bottle is cheap and the tail is long, which is why the metric that
matters here is RMSLE (error in *relative* terms) rather than raw-dollar MSE -- being \\$20 out on a
\\$25 bottle is a disaster, on a \\$400 bottle it is noise.""",
    ),
    (
        "code",
        """prices = np.array([wine.price for wine in wines])
print(f"median ${np.median(prices):,.0f}  mean ${prices.mean():,.0f}  p95 ${np.percentile(prices, 95):,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(prices, bins=80, color="#7f1d3f")
axes[0].set_title("price ($)")
axes[1].hist(np.log1p(prices), bins=80, color="#7f1d3f")
axes[1].set_title("log1p(price)")
plt.show()""",
    ),
    (
        "md",
        """### Does the critic score explain the price?

Partly -- and that is exactly why `points` is kept out of the text the models read by default. It is
a strong, easy proxy that would let a model skip reading the tasting note. Turn it back on with
`compose(wine, fields=(..., "points"))` when you want to measure how much of the signal it carries.""",
    ),
    (
        "code",
        """points = np.array([wine.points for wine in wines])
correlation = np.corrcoef(points, np.log1p(prices))[0, 1]
plt.figure(figsize=(7, 4))
plt.scatter(points + np.random.uniform(-0.4, 0.4, len(points)), prices, s=2, alpha=0.05, color="#7f1d3f")
plt.yscale("log")
plt.title(f"points vs price (r = {correlation:.2f} on log price)")
plt.show()""",
    ),
    ("md", "### Where the wines come from, and what they are"),
    (
        "code",
        """facets = {
    "country": lambda wine: wine.country,
    "variety": lambda wine: wine.variety,
    "taster": lambda wine: wine.taster,
}
for label, pick in facets.items():
    values = [pick(wine) or "unknown" for wine in wines]
    counts = Counter(values)
    print(f"\\n{label}: {len(counts):,} distinct")
    for name, count in counts.most_common(8):
        median = np.median([wine.price for wine, value in zip(wines, values, strict=True) if value == name])
        print(f"  {name[:28]:<30}{count:>7,}   median ${median:>6,.0f}")""",
    ),
    (
        "md",
        """### Tasting notes are short

A few hundred characters each, so a fine-tune sees the whole note comfortably inside a small context.""",
    ),
    (
        "code",
        """lengths = np.array([len(wine.description) for wine in wines])
print(f"median {np.median(lengths):.0f} chars, p99 {np.percentile(lengths, 99):.0f} chars")
plt.figure(figsize=(7, 3))
plt.hist(lengths, bins=80, color="#7f1d3f")
plt.title("tasting note length (chars)")
plt.show()""",
    ),
    (
        "md",
        """### Balancing

Trained on the raw distribution a model learns that guessing \\$25 is nearly always safe. Capping how
many wines each log-price bin may contribute flattens the target and forces the model to read.

The trade is volume: the top bins hold only a few hundred wines each, so a *perfectly* flat set would
be tiny. `cap=6000` keeps a bit over half the data. **Experiment:** rerun with `cap=20_000` and
compare RMSLE on the expensive half of the test set.""",
    ),
    ("code", "price_histogram(wines)"),
    ("code", "balanced = balance(wines, cap=6_000)\nprice_histogram(balanced)"),
    ("md", "### What a model actually reads"),
    ("code", 'print(balanced[0].full)\nprint("\\n--- price:", balanced[0].price)'),
    (
        "code",
        """# The same wine with the leaky fields switched on, for comparison
print(compose(balanced[0], fields=("vintage", "variety", "country", "region", "winery", "points", "note")))""",
    ),
    (
        "md",
        """### Split and cache

Validation and test come off the end of one deterministic shuffle, so shrinking the train set for a
quick experiment does not move the test set.""",
    ),
    (
        "code",
        """train, val, test = split(balanced)
Wine.save_local(train=train, val=val, test=test)""",
    ),
    (
        "md",
        """Next: `scripts/baselines.py` fits the classical ladder on this cache, and
`notebooks/2_baseline_ladder.ipynb` walks through what each rung is worth.""",
    ),
]

BASELINES = [
    (
        "md",
        """# The baseline ladder

Before any LLM, establish what cheap models achieve. Every rung uses the same `Report`, so the
fine-tuned model later is directly comparable.

Rungs: always-guess-the-average, metadata-only linear regression, TF-IDF + Ridge, and LSA + random
forest.""",
    ),
    (
        "code",
        """from pricer.baselines import constant, lsa_forest, metadata_only, tfidf
from pricer.evaluator import evaluate, leaderboard
from pricer.items import Wine

train, val, test = Wine.load_local()
print(f"train={len(train):,} val={len(val):,} test={len(test):,}")""",
    ),
    (
        "md",
        """### Rung 0: the geometric mean

Any model that cannot beat this is broken. Note the negative R2 -- squared error in dollars is
dominated by the expensive tail, which is exactly why RMSLE is the metric to watch.""",
    ),
    ("code", "evaluate(constant(train), test)"),
    ("md", "### Rung 1: metadata only, no tasting note\n\nCountry, province, variety, vintage, note length."),
    ("code", "evaluate(metadata_only(train), test)"),
    ("md", "### Rung 2: the tasting note as bag-of-words"),
    ("code", "evaluate(tfidf(train), test)"),
    ("md", "### Rung 3: dense text features into a random forest"),
    ("code", "evaluate(lsa_forest(train), test)"),
    ("code", "leaderboard()"),
    (
        "md",
        """### Which words cost money?

Ridge coefficients on the word features, which double as a sanity check that the model is reading
tasting vocabulary rather than picking up on a scrape artefact.""",
    ),
    (
        "code",
        """import numpy as np

model = tfidf(train)
pipeline = model.pipeline
features = pipeline.named_steps["features"]
names = features.get_feature_names_out()
weights = pipeline.named_steps["model"].coef_
order = np.argsort(weights)
print("cheap:")
for index in order[:15]:
    print(f"  {names[index].split('__')[-1]:<24}{weights[index]:+.3f}")
print("expensive:")
for index in order[-15:][::-1]:
    print(f"  {names[index].split('__')[-1]:<24}{weights[index]:+.3f}")""",
    ),
    (
        "md",
        """### Experiments worth running here

- Re-curate with `cap=20_000` and see whether the extra cheap wines help or just re-teach the prior.
- Add `points` to the composed text (`pricer.parser.compose`) and measure the jump. That gap is the
  value of the critic score, and a lower bound on how much a model can learn from the note alone.
- Predict `points` instead of price -- same harness, a much less skewed target.
- Swap Ridge for gradient boosting on the LSA features.""",
    ),
]


PROMPTS = [
    (
        "md",
        """# Prompts and token budgets for the fine-tune

The fine-tune eats text, so before any GPU time: decide what the model reads, how long it may be,
and push the result to the Hub where Colab can reach it.

`pricer/prompts.py` holds the layout. The rule that matters: the training prompt and the inference
prompt must be identical up to the final `Price is $`, or the model learns a format it will never see
again.""",
    ),
    (
        "code",
        """import os

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

from pricer import prompts
from pricer.items import Wine

load_dotenv(override=True)
login(os.environ["HF_TOKEN"])

BASE_MODEL = "Qwen/Qwen2.5-3B"  # open weights, no gate to accept; Llama-3.2-3B works the same way
DATASET = "borjahernandez/wine-pricer"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
train, val, test = Wine.load_local()
print(f"train={len(train):,} val={len(val):,} test={len(test):,}")""",
    ),
    ("md", "### How long is a wine, in tokens?"),
    (
        "code",
        """counts = np.array([len(tokenizer.encode(wine.full, add_special_tokens=False)) for wine in train[:5_000]])
print(f"median {np.median(counts):.0f}, p99 {np.percentile(counts, 99):.0f}, max {counts.max()}")
plt.figure(figsize=(8, 3))
plt.hist(counts, bins=60, color="#7f1d3f")
plt.axvline(prompts.CUTOFF, color="black", linestyle="--", label=f"CUTOFF={prompts.CUTOFF}")
plt.legend()
plt.title("tokens per wine")
plt.show()
cut = (counts > prompts.CUTOFF).mean()
print(f"CUTOFF truncates {cut:.1%} of wines")""",
    ),
    (
        "md",
        """### Build the prompts

Truncation happens in token space and then backs off to a word boundary, so a wine never ends
mid-word. Every price is rendered as `$42.00` -- one consistent shape, two tokens, easy to parse
back.""",
    ),
    (
        "code",
        """prompts.prepare(train, tokenizer)
prompts.prepare(val, tokenizer)
prompts.prepare(test, tokenizer)
print(train[0].prompt)
print("\\n--- at inference the model sees:\\n")
print(train[0].test_prompt())""",
    ),
    (
        "md",
        """### Push to the Hub

Colab pulls this dataset for training. Nothing here is secret, but the tasting notes are Wine
Enthusiast's, so keep the dataset private if you plan to leave it up.""",
    ),
    ("code", 'Wine.push_to_hub(DATASET, train, val, test)\nprint(f"https://huggingface.co/datasets/{DATASET}")'),
    (
        "md",
        """### Experiment: is the flowery prose worth its tokens?

Rerun this notebook with `use_summary=True` (after `scripts/tasting.py` has filled in the LLM
summaries) and push to a second dataset. Fine-tune on both. The summary is roughly a fifth of the
tokens, so if it scores within noise of the full note, the note is mostly decoration.""",
    ),
    (
        "code",
        """# prompts.prepare(train, tokenizer, use_summary=True)   # needs scripts/tasting.py to have run
# Wine.push_to_hub(f"{DATASET}-summaries", train, val, test)""",
    ),
]

QLORA = [
    (
        "md",
        """# QLoRA fine-tune

**Run this in Colab on a T4 (free) or an A100.** Nothing here works on a laptop: it needs a CUDA GPU
for 4-bit quantisation.

The plan: load a 3B base model in 4-bit, attach LoRA adapters to the attention projections, and train
on the tasting-note prompts so the model completes `Price is $` with a number. Only the adapters
train -- about 0.5% of the parameters -- which is what makes this fit in 16GB.""",
    ),
    (
        "code",
        """!pip install -q "transformers>=4.44" "peft>=0.13" "trl>=0.11" "bitsandbytes>=0.44" \\
    "datasets>=3.0" "accelerate>=1.0"
!git clone -q https://github.com/borjahernandez/wine-pricer.git
%cd wine-pricer""",
    ),
    (
        "code",
        """import torch
from datasets import load_dataset
from google.colab import userdata
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

login(userdata.get("HF_TOKEN"))

BASE_MODEL = "Qwen/Qwen2.5-3B"
DATASET = "borjahernandez/wine-pricer"
RUN = "wine-pricer-qwen3b"
PREFIX = "Price is $"  # the response template: loss is computed on what follows it""",
    ),
    (
        "md",
        """### Hyperparameters

Sensible starting points, all worth a sweep:

| knob | value | why |
| --- | --- | --- |
| `r` | 32 | adapter rank. 8 underfits here, 64 costs memory for little gain |
| `alpha` | 64 | conventionally 2r |
| target modules | attention projections | where the task-specific reasoning lives |
| `lr` | 1e-4 | LoRA tolerates rates ~10x a full fine-tune |
| epochs | 1 | 50k examples is plenty; a second epoch mostly memorises |
| 4-bit nf4, double quant | on | the whole reason this fits on a T4 |""",
    ),
    (
        "code",
        """LORA = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

QUANT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

CONFIG = SFTConfig(
    output_dir=RUN,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch 16
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_32bit",
    max_seq_length=256,
    dataset_text_field="prompt",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    report_to="none",
    push_to_hub=True,
    hub_model_id=f"borjahernandez/{RUN}",
    hub_private_repo=True,
)""",
    ),
    (
        "code",
        """data = load_dataset(DATASET)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=QUANT, device_map="auto")
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Train on the answer only: without this the model spends its capacity learning to recite tasting notes.
collator = DataCollatorForCompletionOnlyLM(response_template=PREFIX, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    peft_config=LORA,
    args=CONFIG,
    data_collator=collator,
)
trainer.train()
trainer.push_to_hub(f"Fine-tuned on {DATASET}")""",
    ),
    (
        "md",
        """### Score it on the same test split as everything else

Two ways to read the answer out:

1. **Generate** a few tokens and parse the number.
2. **Weighted average over the logits** of the first answer token -- the model's whole distribution
   instead of its argmax, which is measurably better calibrated for a numeric target.

Both go through `pricer.evaluator`, so the result drops straight onto the same leaderboard as the
classical baselines.""",
    ),
    (
        "code",
        """import re

from pricer.evaluator import evaluate
from pricer.items import Wine

_, _, test = Wine.from_hub(DATASET)
model.eval()


def parse_price(text: str) -> float:
    match = re.search(r"[-+]?\\d[\\d,]*\\.?\\d*", text.replace("$", ""))
    return float(match.group().replace(",", "")) if match else 0.0


def specialist(wine: Wine) -> float:
    inputs = tokenizer(wine.test_prompt(), return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :])
    return parse_price(completion)


specialist.__name__ = "Fine-tuned Qwen2.5-3B"
evaluate(specialist, test, size=250)""",
    ),
    (
        "code",
        """def weighted(wine: Wine, top: int = 8) -> float:
    \"\"\"Expected price under the model's own distribution over the first answer token.\"\"\"
    inputs = tokenizer(wine.test_prompt(), return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    values, indices = probabilities.topk(top)
    prices, weights = [], []
    for probability, index in zip(values.tolist(), indices.tolist(), strict=True):
        price = parse_price(tokenizer.decode(index))
        if price:
            prices.append(price)
            weights.append(probability)
    if not prices:
        return 0.0
    total = sum(weights)
    return sum(price * weight for price, weight in zip(prices, weights, strict=True)) / total


weighted.__name__ = "Fine-tuned Qwen2.5-3B (weighted)"
evaluate(weighted, test, size=250)""",
    ),
    (
        "md",
        """### Experiments

- **Base model, untrained** on the same prompts: the gap is what the fine-tune actually bought.
- **Rank sweep**: r = 8 / 32 / 64 at matched steps.
- **Summaries vs full notes** (`-summaries` dataset from the previous notebook).
- **Add `points` to the prompt** and watch the fine-tune coast -- the same leakage the baselines see.
- **Bigger base**: an 8B model in 4-bit still fits an A100. Does scale beat data curation here?""",
    ),
]


def build(name: str, cells: list[tuple[str, str]]) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(source) if kind == "md" else nbformat.v4.new_code_cell(source)
            for kind, source in cells
        ]
    )
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    NOTEBOOKS.mkdir(exist_ok=True)
    path = NOTEBOOKS / name
    nbformat.write(notebook, path)
    print(f"wrote {path} ({len(cells)} cells)")


RAG = [
    (
        "md",
        """# Retrieval and RAG over tasting notes

Retrieval asks a different question from the models so far: not "what does this prose imply about
price" but "what did wines that taste like this actually cost".

The store holds tasting notes only. No price, no critic score, no winery in the embedded text -- if
those went in, similarity search would find the answer instead of a comparable wine, and the whole
evaluation would be a leak with extra steps. Prices live in the metadata, retrieved *after* the
match, which is how a comparable is supposed to work.""",
    ),
    (
        "code",
        """from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from pricer import vectors
from pricer.agents import ClassicalAgent, FrontierAgent, NeighboursAgent, price_all, setup_logging
from pricer.evaluator import Report, leaderboard
from pricer.items import Wine

setup_logging()
train, val, test = Wine.load_local()
collection = vectors.load()  # built by scripts/vectors.py
encoder = vectors.encoder()
print(f"{collection.count():,} tasting notes embedded")""",
    ),
    (
        "md",
        """### Does the embedding space know about price?

If wines that taste alike also cost alike, the neighbourhood structure carries price information and
retrieval will help. Colour a t-SNE projection by price and look for gradient rather than noise.""",
    ),
    (
        "code",
        """embeddings, prices, varieties = vectors.sample_coordinates(collection, limit=2_000)
flat = TSNE(n_components=2, random_state=42, init="pca", perplexity=30).fit_transform(embeddings)
plt.figure(figsize=(9, 7))
points = plt.scatter(flat[:, 0], flat[:, 1], c=np.log1p(prices), cmap="RdYlGn_r", s=8)
plt.colorbar(points, label="log1p(price)")
plt.title("2,000 tasting notes, coloured by price")
plt.show()
print("Most common varieties in the sample:", Counter(varieties).most_common(5))""",
    ),
    ("md", "### What does retrieval return for one wine?"),
    (
        "code",
        """wine = test[7]
notes, found = vectors.similar(wine.description, collection, encoder, k=5)
print(f"{wine.label} -- actually ${wine.price:.0f}\\n")
for note, price in zip(notes, found, strict=True):
    print(f"${price:>6.0f}  {note[:110]}...")
print(f"\\ngeometric mean of the neighbours: ${np.expm1(np.log1p(found).mean()):.2f}")""",
    ),
    (
        "md",
        """### Retrieval alone, then retrieval plus a language model

Two agents, one question each. `NeighboursAgent` is pure retrieval: the geometric mean of the k
nearest prices, no LLM. `FrontierAgent` puts the same neighbours in a prompt and asks a model for a
number. The gap between them is what the language model contributes over the lookup; if it is small,
the expensive part is not earning its keep.

100 test wines, because the frontier agent goes over the network for each one. Note the sample size in
the leaderboard: these rows are not comparable to the 2,000-wine baseline rows, only to each other.

The classical agent here is the **note-only** model, not the metadata-aware baseline. An agent receives prose and
nothing else, so feeding the metadata-aware pipeline `variety='unknown', vintage=0` at inference --
after fitting it on the real values -- cost about 0.2 RMSLE. Train on what you can actually serve.""",
    ),
    (
        "code",
        """sample = test[:100]
neighbours = NeighboursAgent(collection, encoder)
classical = ClassicalAgent()

for name, agent in [("Neighbours (k=8, retrieval only)", neighbours), ("Classical (note only)", classical)]:
    guesses = [agent.price(w.description) for w in sample]
    Report(name, [w.label for w in sample], guesses, [w.price for w in sample]).save()

frontier = FrontierAgent(collection, encoder)
guesses = price_all(frontier, [w.description for w in sample])  # stops early if the daily quota runs out
scored = sample[: len(guesses)]
if scored:
    Report("Frontier (RAG + LLM)", [w.label for w in scored], guesses, [w.price for w in scored]).save()
leaderboard()""",
    ),
    (
        "md",
        """### Experiments worth running here

- Sweep `k` in `NeighboursAgent`. Too few neighbours is noisy, too many regresses to the mean.
- Weight the neighbours by similarity instead of averaging them flat.
- Retrieve on the LLM summary instead of the full note (`scripts/tasting.py` first) and see whether a
  tighter, more structured text retrieves better comparables.
- Embed with a bigger encoder (`all-mpnet-base-v2`) and measure whether the extra dimensions pay.
- Give the frontier agent the neighbours' varieties and regions too, and see if context helps or
  just distracts it.""",
    ),
]

AGENTS = [
    (
        "md",
        """# The agent framework

Five agents, each with one job, wired into a pipeline that goes from an RSS feed to a notification:

| agent | what it does |
| --- | --- |
| `ClassicalAgent` | the baseline TF-IDF + Ridge model, cheap and offline |
| `NeighboursAgent` | retrieval only: the geometric mean of comparable prices |
| `FrontierAgent` | RAG plus a language model |
| `SpecialistAgent` | our own QLoRA fine-tune (needs a GPU, so not run here) |
| `EnsembleAgent` | a linear blend of the above, fitted on validation |
| `ScannerAgent` | reads the wine press and structures every wine quoted with a price |
| `PlanningAgent` | scan, price, rank by gap, notify |""",
    ),
    (
        "code",
        """from pricer import vectors
from pricer.agents import (
    ClassicalAgent,
    EnsembleAgent,
    FrontierAgent,
    NeighboursAgent,
    PlanningAgent,
    ScannerAgent,
    price_all,
    setup_logging,
)
from pricer.evaluator import Report, leaderboard
from pricer.items import Wine

setup_logging()
train, val, test = Wine.load_local()
collection, encoder = vectors.load(), vectors.encoder()
members = [ClassicalAgent(), NeighboursAgent(collection, encoder), FrontierAgent(collection, encoder)]""",
    ),
    (
        "md",
        """### Fit the blend

On **validation**, never on train: the classical member was fitted on train and the retrieval members
can find train wines verbatim, so their training-set accuracy is fantasy. 150 wines is enough for
five coefficients and keeps the frontier agent's bill small -- and 150 frontier calls is already most
of a free tier's day, so drop it further if you are counting tokens.""",
    ),
    (
        "code",
        """ensemble = EnsembleAgent(members)
ensemble.fit(val[:150])
ensemble.save()
ensemble.price(test[0].description), test[0].price""",
    ),
    ("md", "### Does the blend beat its members?"),
    (
        "code",
        """sample = test[:100]
guesses = price_all(ensemble, [w.description for w in sample])
scored = sample[: len(guesses)]
if scored:
    Report("Ensemble", [w.label for w in scored], guesses, [w.price for w in scored]).save()
leaderboard()""",
    ),
    (
        "md",
        """### The scanner: real wines, in the wild

There is no free live wine-price API, and the deal aggregators carry almost no wine, so the source
here is the wine press: Wine Enthusiast and Decanter RSS. Their articles quote a tasting note and a
shelf price, which is exactly the pair this project needs. See `pricer/deals.py` for what was
verified reachable.

Editorial feeds are noisy: many articles name no price at all, and the scanner throws those away.""",
    ),
    (
        "code",
        """scanner = ScannerAgent()
listings = scanner.scan(per_feed=3)
for listing in listings:
    print(f"${listing.price:>7.0f}  {listing.name}\\n          {listing.note[:110]}...")""",
    ),
    (
        "md",
        """### The planner, end to end

Scan, drop anything outside the $4-$500 range the models were trained on, price the rest, rank by the
gap, notify on anything big. `memory.json` stops a second run re-reporting the same wine.

A word on the gap: our best model carries an RMSLE near 0.5, so a "$20 bargain" is inside the noise.
The interesting output is the pipeline working, not the trade.""",
    ),
    (
        "code",
        """planner = PlanningAgent(ensemble, scanner=scanner)
opportunities = planner.plan(per_feed=3, threshold=15.0)
for opportunity in opportunities[:10]:
    print(opportunity.summary(), "\\n")""",
    ),
    (
        "md",
        """### Experiments worth running here

- Add the fine-tuned `SpecialistAgent` to the members (on a GPU box) and refit the blend. Does the
  specialist dominate, or does the ensemble still want the retrieval members?
- Replace the linear blend with gradient boosting over the members' guesses.
- Have the frontier agent output a *range* and use its width as an uncertainty feature for the blend.
- Point the scanner at a retailer's feed instead of the press and see how much of the pipeline still
  works when the prose is marketing copy rather than criticism.
- Run `python app.py` for the Gradio front end, and `scripts/plan.py` for the pipeline on a cron.""",
    ),
]


if __name__ == "__main__":
    build("1_curate_and_explore.ipynb", CURATION)
    build("2_baseline_ladder.ipynb", BASELINES)
    build("3_prompts_and_tokens.ipynb", PROMPTS)
    build("4_qlora_finetune_colab.ipynb", QLORA)
    build("5_retrieval_and_rag.ipynb", RAG)
    build("6_agent_framework.ipynb", AGENTS)
