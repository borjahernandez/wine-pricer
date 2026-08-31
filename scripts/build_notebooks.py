"""Generate the project notebooks from plain-Python cell definitions.

Notebooks are painful to write and review as JSON, so the source of truth lives here: each notebook
is a list of (kind, source) cells, and this script writes clean, output-free .ipynb files.

    uv run python scripts/build_notebooks.py
"""

from pathlib import Path

import nbformat

NOTEBOOKS = Path("notebooks")

WEEK6 = [
    (
        "md",
        """# Week 6, day 1 -- meet the wines

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
`notebooks/week6_baselines.ipynb` walks through what each rung is worth.""",
    ),
]

WEEK6_BASELINES = [
    (
        "md",
        """# Week 6, day 2 -- the baseline ladder

Before any LLM, establish what cheap models achieve. Every rung uses the same `Report`, so the
fine-tuned model in week 7 is directly comparable.

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


WEEK7_PROMPTS = [
    (
        "md",
        """# Week 7, day 1 -- prompts for the fine-tune

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

WEEK7_QLORA = [
    (
        "md",
        """# Week 7, days 2-4 -- QLoRA fine-tune

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

Both go through `pricer.evaluator`, so the result drops straight onto the week-6 leaderboard.""",
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


if __name__ == "__main__":
    build("week6_curate.ipynb", WEEK6)
    build("week6_baselines.ipynb", WEEK6_BASELINES)
    build("week7_prompts.ipynb", WEEK7_PROMPTS)
    build("week7_qlora_colab.ipynb", WEEK7_QLORA)
