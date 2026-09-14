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
import datetime
import json
import os
import random

from dotenv import load_dotenv

from agents.team_a_strategy import build_team_a_report
from agents.team_b_quants import analyze_team_b
from agents.team_c_data import gather_team_c_signals
from agents.team_s_execution import execute_team_a_report
from symbol_universe import SP500_CANDIDATES
from tools.alpaca_tools import get_account_info, get_latest_prices
from tools.run_logger import append_run_record, build_run_record

load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"))

# Fallback watchlist, used only if --symbols is passed explicitly (bypassing
# price filtering entirely) or if the price-based selection below can't run
# (e.g. no live Alpaca credentials to price the candidate pool with). The
# default path scans symbol_universe.SP500_CANDIDATES (~560 symbols) and
# price-filters it down to whatever's actually affordable -- see
# select_affordable_symbols().
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


DEFAULT_PRICE_BRACKET_PCT = 0.125  # -> a $200 account brackets to ~$25/share, scaling proportionally with equity
DEFAULT_MAX_CANDIDATES = 76  # matches the compute/Reddit-rate-limit budget the watchlist was originally sized for


def select_affordable_symbols(
    candidates: list[str],
    account_equity: float | None,
    price_bracket_pct: float = DEFAULT_PRICE_BRACKET_PCT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[str]:
    """
    Cheaply price-filter a large candidate pool down to whatever's actually
    affordable at the account's current equity, before running the full
    (expensive: ~6 network calls + ARIMA/GBM/Random-Forest per symbol)
    pipeline on any of them.

    The price ceiling scales with equity (`max($1, equity * price_bracket_pct)`)
    so a $200 account and a $10,000 account each get a bracket proportional
    to what they can actually deploy, rather than a fixed dollar range that
    only makes sense for one account size. The floor is always $1 to keep
    penny stocks out.

    If more than `max_candidates` symbols qualify, a day-seeded random
    sample is taken -- stable within a run (and across the same UTC day, so
    a scheduled run and a same-day manual re-run see the same picks), but
    not a permanently frozen prefix of the list either.
    """
    if account_equity is None:
        return []  # no live account to price a bracket against -- caller falls back to DEFAULT_WATCHLIST

    price_ceiling = max(1.0, account_equity * price_bracket_pct)
    prices = get_latest_prices(candidates)
    affordable = [s for s in candidates if s in prices and 1.0 <= prices[s] <= price_ceiling]

    print(
        f"Price bracket this run: $1.00-${price_ceiling:.2f} "
        f"(equity ${account_equity:.2f} x {price_bracket_pct:.1%}) "
        f"-> {len(affordable)}/{len(candidates)} candidates affordable"
    )

    if len(affordable) > max_candidates:
        rng = random.Random(datetime.date.today().isoformat())
        affordable = rng.sample(affordable, max_candidates)
        affordable.sort()

    return affordable


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
    symbols: list[str] | None = None,
    dry_run: bool = True,
    max_position_usd: float = DEFAULT_MAX_POSITION_USD,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    price_bracket_pct: float = DEFAULT_PRICE_BRACKET_PCT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[dict]:
    """
    Scan `symbols` through the full pipeline, independently -- or, if
    `symbols` is None, price-filter symbol_universe.SP500_CANDIDATES down to
    whatever's affordable at current equity and scan that instead.
    """
    # Priced once per batch from real equity, not a flag disconnected from the
    # account -- see compute_position_cap()'s docstring for the reasoning.
    account_info = _try_get_account_info()
    position_cap = compute_position_cap(account_info, max_position_usd, max_position_pct)
    equity_note = f"${account_info['equity']:.2f} equity" if account_info else "no live account"

    if symbols is None:
        equity = account_info["equity"] if account_info else None
        symbols = select_affordable_symbols(SP500_CANDIDATES, equity, price_bracket_pct, max_candidates)
        if not symbols:
            print("No affordable candidates found (or no live account to price against) -- falling back to DEFAULT_WATCHLIST.")
            symbols = DEFAULT_WATCHLIST

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
        default=None,
        help=(
            "Comma-separated ticker symbols to scan, e.g. 'AAPL,MSFT,TSLA'. "
            "Bypasses price-bracket filtering entirely -- runs exactly what's given. "
            "Omit to auto-select affordable symbols from the ~560-symbol candidate pool instead."
        ),
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
        "--price-bracket-pct",
        type=float,
        default=DEFAULT_PRICE_BRACKET_PCT,
        help="Per-share price ceiling as a fraction of equity (0.125 = 12.5%%), floor $1. Ignored if --symbols is set.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="Cap on how many price-affordable candidates get the full pipeline in one run. Ignored if --symbols is set.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit paper trades via Alpaca (otherwise runs as a dry run).",
    )
    args = parser.parse_args()

    symbol_list = None
    if args.symbols is not None:
        symbol_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    run_pipeline(
        symbol_list,
        dry_run=not args.execute,
        max_position_usd=args.max_position_usd,
        max_position_pct=args.max_position_pct,
        price_bracket_pct=args.price_bracket_pct,
        max_candidates=args.max_candidates,
    )
