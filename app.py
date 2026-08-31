"""Vintage Is Right -- the Gradio front end.

    uv run python app.py

Two tabs. **Price a wine** takes a tasting note and shows what each agent thinks, next to the
comparable wines RAG retrieved -- the retrieved bottles are the explanation for the number. **Scan
the press** runs the planning agent over live RSS and lists the gaps it found.

Agents are built lazily: the classical one costs a second, the RAG ones need the Chroma store and a
Groq key, so nothing is loaded until a tab asks for it.
"""

import gradio as gr
import pandas as pd

from pricer.agents import ClassicalAgent, FrontierAgent, NeighboursAgent, PlanningAgent, setup_logging
from pricer.llm import DailyLimitReached
from pricer.vectors import encoder, load, similar

EXAMPLE = (
    "Aromas of dried cherry, tobacco leaf and warm earth lead into a firm, savoury palate framed by "
    "fine-grained tannins and bright acidity. The finish is long and dusty, and it should reward a "
    "decade in the cellar."
)
built: dict[str, object] = {}


def agents() -> dict[str, object]:
    """Build once, reuse for the life of the process."""
    if not built:
        built["classical"] = ClassicalAgent()
        collection, model = load(), encoder()
        built["neighbours"] = NeighboursAgent(collection, model)
        built["frontier"] = FrontierAgent(collection, model)
        built["collection"], built["encoder"] = collection, model
    return built


def estimate(note: str, use_rag: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not note.strip():
        raise gr.Error("Paste a tasting note first")
    ready = agents()
    rows = [("Classical (note only)", f"${ready['classical'].price(note):.2f}")]
    comparables = pd.DataFrame(columns=["comparable wine", "price"])
    if use_rag:
        rows.append(("Neighbours (retrieval only)", f"${ready['neighbours'].price(note):.2f}"))
        try:
            rows.append(("Frontier (RAG + LLM)", f"${ready['frontier'].price(note):.2f}"))
        except DailyLimitReached:
            # The free tier runs out long before curiosity does; the other agents still work.
            rows.append(("Frontier (RAG + LLM)", "unavailable: daily token budget spent"))
        notes, found = similar(note, ready["collection"], ready["encoder"], k=5)
        comparables = pd.DataFrame(
            {"comparable wine": [text[:200] + "..." for text in notes], "price": [f"${p:.0f}" for p in found]}
        )
    return pd.DataFrame(rows, columns=["agent", "estimate"]), comparables


def scan(per_feed: int) -> pd.DataFrame:
    planner = PlanningAgent(agents()["classical"])
    try:
        opportunities = planner.plan(per_feed=int(per_feed))
    except DailyLimitReached as error:
        raise gr.Error("The provider's daily token allowance is spent -- try again tomorrow") from error
    if not opportunities:
        return pd.DataFrame([{"wine": "Nothing new with a price in the feeds right now", "listed": "", "gap": ""}])
    return pd.DataFrame(
        [
            {
                "wine": o.listing.name,
                "listed": f"${o.listing.price:.0f}",
                "we estimate": f"${o.estimate:.0f}",
                "gap": f"${o.discount:+.0f}",
                "url": o.listing.url,
            }
            for o in opportunities
        ]
    )


with gr.Blocks(title="Vintage Is Right") as ui:
    gr.Markdown("# Vintage Is Right\nHow much is that bottle worth, judging only by how it tastes?")
    with gr.Tab("Price a wine"):
        note = gr.Textbox(label="Tasting note", lines=6, value=EXAMPLE)
        rag = gr.Checkbox(label="Also run the retrieval agents (needs the Chroma store and a Groq key)", value=True)
        go = gr.Button("Estimate", variant="primary")
        estimates = gr.Dataframe(label="What the agents think", interactive=False)
        comparables = gr.Dataframe(label="Comparable wines retrieved from the training set", interactive=False)
        go.click(estimate, inputs=[note, rag], outputs=[estimates, comparables])
    with gr.Tab("Scan the press"):
        gr.Markdown("Reads Wine Enthusiast and Decanter RSS, structures every wine quoted with a price, prices it.")
        per_feed = gr.Slider(1, 10, value=3, step=1, label="Articles per feed")
        scan_button = gr.Button("Scan", variant="primary")
        findings = gr.Dataframe(label="Gaps between the listed price and our estimate", interactive=False)
        scan_button.click(scan, inputs=[per_feed], outputs=[findings])

if __name__ == "__main__":
    setup_logging()
    ui.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(primary_hue="rose"))
