"""
Entry point for the hierarchical multi-agent trading system.

Enforces the strict pipeline:
    Team C (Data/Signals) -> Team B (Quants) -> Team A (Strategy/Risk) -> Team S (Execution)

Lower teams never execute trades; only Team S's execution agent is
permitted to call the Alpaca API, and only on a report Team A's Risker
has approved.

Each run scans a watchlist of symbols (not just one) -- every symbol
goes through the full pipeline independently, and each gets its own
entry in logs/history.json regardless of whether Team A approved it.
"""

import argparse
import json
import os

from dotenv import load_dotenv

from agents.team_a_strategy import build_team_a_report
from agents.team_b_quants import analyze_team_b
from agents.team_c_data import gather_team_c_signals
from agents.team_s_execution import execute_team_a_report
from tools.alpaca_tools import get_account_info
from tools.run_logger import append_run_record, build_run_record

load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"))

# A broad, liquid cross-section of the market rather than one symbol --
# large caps across tech, finance, energy, healthcare, and consumer so
# Team B's regime detection isn't just reading one sector's mood. Scanning
# the literal entire market isn't practical (thousands of symbols, each
# running a full ARIMA/GBM/Random-Forest pass) so this stands in as "the
# market" -- extend it freely via --symbols.
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "XOM", "JNJ",
]


def _try_get_account_info() -> dict | None:
    """Best-effort account snapshot for the dashboard; None if creds are missing/invalid."""
    try:
        return get_account_info()
    except Exception as exc:
        print(f"[Team S] Could not fetch Alpaca account info: {exc}")
        return None


def run_pipeline_for_symbol(symbol: str, allocated_capital: float, dry_run: bool) -> dict:
    print(f"\n=== {symbol} ===")
    print(f"[Team C] Gathering data & signals for {symbol}...")
    team_c_output = gather_team_c_signals(symbol)

    print("[Team B] Running quantitative analysis...")
    team_b_output = analyze_team_b(team_c_output)

    print("[Team A] Forming trade thesis and applying risk management...")
    team_a_report = build_team_a_report(team_b_output, team_c_output, allocated_capital)
    print(json.dumps({k: v for k, v in team_a_report.items() if k != "price_history"}, indent=2, default=str))

    execution_result = None
    if dry_run:
        print("[Team S] Dry run enabled -- skipping execution.")
    else:
        print("[Team S] Reviewing Team A's report for execution...")
        execution_result = execute_team_a_report(team_a_report)
        print(json.dumps(execution_result, indent=2, default=str))

    # Fetched after any execution above so the dashboard's equity curve
    # reflects this symbol's trade too, not a stale pre-batch snapshot.
    account_info = _try_get_account_info()
    record = build_run_record(
        symbol=symbol,
        capital=allocated_capital,
        dry_run=dry_run,
        team_c_output=team_c_output,
        team_b_output=team_b_output,
        team_a_report=team_a_report,
        execution_result=execution_result,
        account_info=account_info,
    )
    append_run_record(record)

    return {"symbol": symbol, "dry_run": dry_run, "team_a_report": team_a_report, "execution_result": execution_result}


def run_pipeline(symbols: list[str], allocated_capital: float, dry_run: bool = True) -> list[dict]:
    """Scan every symbol in `symbols` through the full pipeline, independently."""
    results = []
    for symbol in symbols:
        try:
            results.append(run_pipeline_for_symbol(symbol, allocated_capital, dry_run))
        except Exception as exc:
            print(f"[{symbol}] Pipeline failed: {exc}")
            results.append({"symbol": symbol, "error": str(exc)})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hierarchical multi-agent trading pipeline across a watchlist.")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_WATCHLIST),
        help="Comma-separated ticker symbols to scan, e.g. 'AAPL,MSFT,TSLA'.",
    )
    parser.add_argument("--capital", type=float, default=10000.0, help="Capital allocated per approved thesis (USD).")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit paper trades via Alpaca (otherwise runs as a dry run).",
    )
    args = parser.parse_args()

    symbol_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run_pipeline(symbol_list, args.capital, dry_run=not args.execute)
