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
# large/mid caps spanning every major sector so Team B's regime detection
# isn't just reading one sector's mood. Scanning the literal entire market
# isn't practical (thousands of symbols, each running a full
# ARIMA/GBM/Random-Forest pass) so this ~75-name list stands in for "the
# market" -- extend it freely via --symbols.
DEFAULT_WATCHLIST = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CRM",
    "ADBE", "CSCO", "AMD", "INTC", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "BMY", "CVS",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX", "TGT", "HD", "LOW",
    # Industrials
    "BA", "CAT", "GE", "HON", "UPS", "LMT", "MMM", "DE",
    # Communication services
    "DIS", "CMCSA", "VZ", "T", "NFLX",
    # Utilities
    "NEE", "DUK", "SO",
    # Materials
    "LIN", "FCX",
    # Real estate
    "AMT", "PLD",
]

# Position sizing is capped two ways at once -- whichever is smaller wins:
#   1. a hard dollar ceiling (MAX_POSITION_USD), so no single trade can ever
#      eat a large slice of a small account regardless of its equity, and
#   2. a percentage of current equity (MAX_POSITION_PCT), so sizing scales
#      down automatically if the account shrinks.
# On a $200 account with the defaults below, that's min($50, $50) = $50 per
# position, and Team A's own half-Kelly cap (capped at 25% of that -- see
# agents/team_a_strategy.py's MAX_POSITION_FRACTION) brings the realistic
# ceiling on any single trade down further to about $12.50. A $20/10% pair
# was tried first and turned out too conservative -- on a $200 account it
# capped every trade at $5, which can't buy a whole share of nearly
# anything in a large/mid-cap watchlist, so every approved thesis got
# skipped. This still leaves Team S's live buying-power check
# (agents/team_s_execution.py) as the backstop against overrunning what's
# actually left to spend.
DEFAULT_MAX_POSITION_USD = 50.0
DEFAULT_MAX_POSITION_PCT = 0.25


def _try_get_account_info() -> dict | None:
    """Best-effort account snapshot for the dashboard; None if creds are missing/invalid."""
    try:
        return get_account_info()
    except Exception as exc:
        print(f"[Team S] Could not fetch Alpaca account info: {exc}")
        return None


def compute_position_cap(account_info: dict | None, max_position_usd: float, max_position_pct: float) -> float:
    """The dollar ceiling for a single position this run -- see the sizing note above."""
    if account_info is None:
        return max_position_usd  # no live account (e.g. a dry run without credentials) -- fall back to the flat ceiling
    return min(max_position_usd, account_info["equity"] * max_position_pct)


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


def run_pipeline(
    symbols: list[str],
    dry_run: bool = True,
    max_position_usd: float = DEFAULT_MAX_POSITION_USD,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
) -> list[dict]:
    """Scan every symbol in `symbols` through the full pipeline, independently."""
    # Priced once per batch from real equity, not a flag disconnected from the
    # account -- see compute_position_cap()'s docstring for the reasoning.
    account_info = _try_get_account_info()
    position_cap = compute_position_cap(account_info, max_position_usd, max_position_pct)
    equity_note = f"${account_info['equity']:.2f} equity" if account_info else "no live account"
    print(f"Position cap this run: ${position_cap:.2f} per symbol ({equity_note}, {len(symbols)} symbols)")

    results = []
    for symbol in symbols:
        try:
            results.append(run_pipeline_for_symbol(symbol, position_cap, dry_run))
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
    parser.add_argument(
        "--max-position-usd",
        type=float,
        default=DEFAULT_MAX_POSITION_USD,
        help="Hard dollar ceiling for a single position, regardless of equity.",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=DEFAULT_MAX_POSITION_PCT,
        help="Ceiling for a single position as a fraction of current equity (0.10 = 10%%).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit paper trades via Alpaca (otherwise runs as a dry run).",
    )
    args = parser.parse_args()

    symbol_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run_pipeline(
        symbol_list,
        dry_run=not args.execute,
        max_position_usd=args.max_position_usd,
        max_position_pct=args.max_position_pct,
    )
