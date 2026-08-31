---
name: testing-gradio-app
description: How to run and end-to-end test the "Vintage Is Right" Gradio app (app.py) in wine-pricer, including how to interpret Groq free-tier quota exhaustion and how to feed very long notes through the UI.
---

# Testing the Vintage Is Right Gradio app

## Start it
- `uv sync --extra agents`, then `uv run python app.py` → serves on `0.0.0.0:7860`.
- First request builds the agents lazily (TF-IDF pickle, Chroma store, sentence-transformers encoder);
  startup to first response can take ~30-60 s. `curl -s -o /dev/null -w "%{http_code}" localhost:7860`
  returning 200 only means Gradio is up, not that the agents are warm.
- Prerequisites that must already exist on disk (they are NOT committed): `data/curated`,
  `data/tfidf_ridge_note_only.pkl`, `chroma/`. Rebuild with
  `uv run python scripts/curate.py --cap 6000` then `uv run python scripts/vectors.py` (~7 min total).
- Redirect stdout/stderr to a log file (e.g. `> /tmp/app.log 2>&1`) — agent logs there
  (`[Frontier Agent] Retrieved 5 comparable wines priced $X-$Y`, `[Scanner Agent] Fetched N articles`)
  are the only way to see which upstream error actually occurred behind a UI toast.

## UI paths
- Tab "Price a wine": Textbox "Tasting note" (prefilled example), Checkbox "Also run the retrieval
  agents" (on by default), Button "Estimate", Dataframes "What the agents think" (agent/estimate) and
  "Comparable wines retrieved from the training set" (comparable wine/price, 5 rows when retrieval on).
- Tab "Scan the press": Slider "Articles per feed" (1-10, default 3), Button "Scan".
- Gotcha: the note Textbox grows with content, which pushes the Estimate button down the page. After
  pasting a long note, re-screenshot and click the button at its new position — a stale coordinate
  click lands in the textbox and silently does nothing (the old results stay on screen and look like
  a fresh run).
- The comparables table's `price` column is horizontally scrolled off; `scroll right` over the table
  to make prices visible for screenshots.

## Feeding very long input
No clipboard tooling (xclip/xsel/tkinter) is installed. To paste ~10k characters, write the text to a
file and use `DISPLAY=:0 xdotool type --delay 2 --clearmodifiers --file /tmp/long.txt` after clicking
the textbox. Budget ~1 ms/char plus overhead (10.5k chars took ~9 minutes); run it backgrounded and
poll, do not use a short timeout.

## Groq quota: expected, handled paths (not bugs)
Free tier is 200,000 tokens/day (`GROQ_API_KEY` from `.env`/environment). When it is spent:
- "Price a wine" shows the Frontier row as `unavailable: daily token budget spent` while Classical and
  Neighbours still return numbers and comparables still populate.
- "Scan the press" shows the toast `The provider's daily token allowance is spent -- try again tomorrow`.
Confirm the cause in the log: look for `DailyLimitReached ... on tokens per day (TPD): Limit 200000`.
Note `pricer/llm.py` only classifies errors matching `tokens per day|TPD` as daily; other 429s are
retried 8 times, so a per-minute limit shows up as a slow response, not this message.
Note also that gr.Error is written to the server log as a traceback — that is normal and does not mean
the UI showed a traceback. Judge "no traceback" by the browser screenshot only.
- Direct probing of `api.groq.com` with bare `urllib` from this box returns HTTP 403 `error code: 1010`
  (edge proxy blocks it); use the app itself to check whether the key/quota works.

## Scan tab caveats
- `memory.json` at the repo root records already-reported wines and there is no UI reset, so a second
  scan can legitimately return "Nothing new with a price in the feeds right now". Delete/rename
  `memory.json` (repo root, not `data/`) if you need a fresh scan.
- Editorial RSS often quotes no prices, so an empty findings table is a valid pass. RSS fetching itself
  is verifiable independently of the LLM via the log line `[Scanner Agent] Fetched N articles`.

## Devin Secrets Needed
- `GROQ_API_KEY` (frontier agent and press scanner). Everything except the Frontier row and the scan
  tab works without it.
