"""
Entry point for the hierarchical multi-agent trading system.

Enforces the strict pipeline:
    Team C (Data/Signals) -> Team B (Quants) -> Team A (Strategy/Risk) -> Team S (Execution)

Lower teams never execute trades; only Team S's execution agent is
permitted to call the Alpaca API, and only on a report Team A's Risker
has approved.
"""

import argparse
import json
import os

from dotenv import load_dotenv

from agents.team_a_strategy import build_team_a_report
from agents.team_b_quants import analyze_team_b
from agents.team_c_data import gather_team_c_signals
from agents.team_s_execution import execute_team_a_report

load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"))


def run_pipeline(symbol: str, allocated_capital: float, dry_run: bool = True) -> dict:
    print(f"[Team C] Gathering data & signals for {symbol}...")
    team_c_output = gather_team_c_signals(symbol)

    print("[Team B] Running quantitative analysis...")
    team_b_output = analyze_team_b(team_c_output)

    print("[Team A] Forming trade thesis and applying risk management...")
    team_a_report = build_team_a_report(team_b_output, team_c_output, allocated_capital)
    print(json.dumps({k: v for k, v in team_a_report.items() if k != "price_history"}, indent=2, default=str))

    if dry_run:
        print("[Team S] Dry run enabled -- skipping execution.")
        return {"dry_run": True, "team_a_report": team_a_report}

    print("[Team S] Reviewing Team A's report for execution...")
    execution_result = execute_team_a_report(team_a_report)
    print(json.dumps(execution_result, indent=2, default=str))

    return {"team_a_report": team_a_report, "execution_result": execution_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hierarchical multi-agent trading pipeline.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to analyze.")
    parser.add_argument("--capital", type=float, default=10000.0, help="Capital allocated to this thesis (USD).")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit the paper trade via Alpaca (otherwise runs as a dry run).",
    )
    args = parser.parse_args()

    run_pipeline(args.symbol, args.capital, dry_run=not args.execute)
