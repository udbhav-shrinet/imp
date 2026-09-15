"""30-minute hierarchical bulk-trading pipeline.

Market Intelligence -> Quant Research -> Portfolio Council -> Execution Commander

The first three teams only produce evidence and a portfolio plan. The final
team is the only code path allowed to submit paper orders.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random

from dotenv import load_dotenv

from agents.execution_commander import execute_exit_plan, execute_portfolio_plan
from agents.market_intelligence import gather_market_intelligence
from agents.portfolio_council import build_portfolio_plan
from agents.position_manager import build_exit_plan
from agents.quant_research import research_symbol
from symbol_universe import SP500_CANDIDATES
from tools.alpaca_tools import (
    get_account_info,
    get_latest_prices,
    get_open_orders,
    get_open_positions,
    get_order_status,
)
from tools.fred_tools import get_macro_snapshot
from tools.run_logger import append_run_record
from tools.trade_ledger import append_fill_snapshots, order_ids_from_execution

load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"))

DEFAULT_MAX_CANDIDATES = 100
DEFAULT_MAX_POSITIONS = 15
DEFAULT_MAX_NEW_POSITIONS = 5
DEFAULT_MAX_POSITION_PCT = 0.03
DEFAULT_MAX_NEW_EXPOSURE_PCT = 0.10


def safe_account() -> dict:
    try:
        return get_account_info()
    except Exception as exc:
        print(f"[Execution Commander] Account unavailable: {exc}")
        return {"equity": 0.0, "cash": 0.0, "buying_power": 0.0}


def affordable_symbols(candidates: list[str], equity: float | None, limit: int) -> list[str]:
    if equity is None or equity <= 0:
        return candidates[:limit]
    prices = get_latest_prices(candidates)
    eligible = [s for s in candidates if 1 <= prices.get(s, 0) <= max(5, equity * 0.25)]
    if len(eligible) > limit:
        eligible = random.Random(datetime.date.today().isoformat()).sample(eligible, limit)
    return sorted(eligible)


def run_pipeline(
    symbols: list[str] | None,
    dry_run: bool,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    max_new_positions: int = DEFAULT_MAX_NEW_POSITIONS,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    max_new_exposure_pct: float = DEFAULT_MAX_NEW_EXPOSURE_PCT,
    allow_shorts: bool = False,
    target_annual_volatility: float = 0.12,
    max_portfolio_var: float = 0.04,
    max_portfolio_cvar: float = 0.06,
    max_correlation: float = 0.85,
) -> dict:
    account = safe_account()
    if symbols is None:
        symbols = affordable_symbols(SP500_CANDIDATES, account.get("equity"), max_candidates)
    print(f"[Market Intelligence] Scanning {len(symbols)} symbols.")
    macro = get_macro_snapshot()
    research = []
    failures = []
    for symbol in symbols:
        try:
            intelligence = gather_market_intelligence(symbol)
            research.append(research_symbol(intelligence, macro))
        except Exception as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
            print(f"[{symbol}] rejected before research: {exc}")

    try:
        positions = get_open_positions()
        open_orders = get_open_orders()
    except Exception as exc:
        print(f"[Portfolio Council] Existing exposure unavailable: {exc}")
        positions, open_orders = [], []

    exit_plan = build_exit_plan(positions, research, open_orders)
    exit_execution = execute_exit_plan(exit_plan, dry_run=dry_run)
    plan = build_portfolio_plan(
        research=research,
        account=account,
        positions=positions,
        open_orders=open_orders,
        max_positions=max_positions,
        max_new_positions=max_new_positions,
        max_position_pct=max_position_pct,
        max_new_exposure_pct=max_new_exposure_pct,
        allow_shorts=allow_shorts,
        target_annual_volatility=target_annual_volatility,
        max_portfolio_var=max_portfolio_var,
        max_portfolio_cvar=max_portfolio_cvar,
        max_correlation=max_correlation,
    )
    execution = execute_portfolio_plan(plan, dry_run=dry_run)
    fill_snapshots = []
    if not dry_run:
        for order_id in order_ids_from_execution(exit_execution, execution):
            try:
                fill_snapshots.append(get_order_status(order_id))
            except Exception as exc:
                fill_snapshots.append({"id": order_id, "status": "reconciliation_error", "error": str(exc)})
        append_fill_snapshots(fill_snapshots)
    try:
        positions_after = get_open_positions()
        account_after = safe_account()
    except Exception as exc:
        print(f"[Execution Commander] Post-trade reconciliation unavailable: {exc}")
        positions_after, account_after = [], account
    record = {
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "interval": "30m",
        "dry_run": dry_run,
        "symbols_scanned": symbols,
        "research": research,
        "plan": plan,
        "exit_plan": exit_plan,
        "exit_execution": exit_execution,
        "execution": execution,
        "fill_snapshots": fill_snapshots,
        "account": account,
        "account_after": account_after,
        "positions_before": positions,
        "positions_after": positions_after,
        "open_orders_before": open_orders,
        "failures": failures,
        "macro": macro,
    }
    append_run_record(record)
    print(json.dumps({"planned": len(plan["trades"]), "executed": execution.get("executed", False),
                      "failures": len(failures)}, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a 30-minute paper bulk-trading cycle.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols; omit for the candidate universe.")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS)
    parser.add_argument("--max-new-positions", type=int, default=DEFAULT_MAX_NEW_POSITIONS)
    parser.add_argument("--max-position-pct", type=float, default=DEFAULT_MAX_POSITION_PCT)
    parser.add_argument("--max-new-exposure-pct", type=float, default=DEFAULT_MAX_NEW_EXPOSURE_PCT)
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument("--target-annual-volatility", type=float, default=0.12)
    parser.add_argument("--max-portfolio-var", type=float, default=0.04)
    parser.add_argument("--max-portfolio-cvar", type=float, default=0.06)
    parser.add_argument("--max-correlation", type=float, default=0.85)
    parser.add_argument("--execute", action="store_true", help="Submit paper orders after all portfolio checks.")
    args = parser.parse_args()
    symbols = None if args.symbols is None else [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run_pipeline(
        symbols=symbols,
        dry_run=not args.execute,
        max_candidates=args.max_candidates,
        max_positions=args.max_positions,
        max_new_positions=args.max_new_positions,
        max_position_pct=args.max_position_pct,
        max_new_exposure_pct=args.max_new_exposure_pct,
        allow_shorts=args.allow_shorts,
        target_annual_volatility=args.target_annual_volatility,
        max_portfolio_var=args.max_portfolio_var,
        max_portfolio_cvar=args.max_portfolio_cvar,
        max_correlation=args.max_correlation,
    )
