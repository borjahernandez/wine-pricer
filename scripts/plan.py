"""Run one full agent pass: scan the wine press, price what it finds, report the gaps.

uv run python scripts/plan.py                    # classical pricer, cheap and offline
uv run python scripts/plan.py --pricer ensemble  # needs the Chroma store and a Groq key
"""

import argparse

from pricer.agents import (
    ClassicalAgent,
    EnsembleAgent,
    FrontierAgent,
    NeighboursAgent,
    PlanningAgent,
    setup_logging,
)


def build_pricer(kind: str):
    classical = ClassicalAgent()
    if kind == "classical":
        return classical
    if kind == "neighbours":
        return NeighboursAgent()
    if kind == "frontier":
        return FrontierAgent()
    ensemble = EnsembleAgent([classical, NeighboursAgent(), FrontierAgent()])
    ensemble.load()  # written by notebooks/6_agent_framework.ipynb
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricer", default="classical", choices=["classical", "neighbours", "frontier", "ensemble"])
    parser.add_argument("--per-feed", type=int, default=3, help="articles to read per feed")
    parser.add_argument("--threshold", type=float, default=15.0, help="dollars of gap before notifying")
    args = parser.parse_args()

    setup_logging()
    planner = PlanningAgent(build_pricer(args.pricer))
    opportunities = planner.plan(per_feed=args.per_feed, threshold=args.threshold)
    print(f"\n{len(opportunities)} wines priced\n")
    for opportunity in opportunities[:10]:
        print(opportunity.summary(), "\n")


if __name__ == "__main__":
    main()
