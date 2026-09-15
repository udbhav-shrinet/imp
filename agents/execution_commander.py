"""Execution Commander: the only team allowed to submit Alpaca orders."""

from __future__ import annotations

import uuid

from tools.alpaca_tools import (
    get_account_info,
    get_latest_price,
    get_open_orders,
    get_open_positions,
    submit_order,
    slice_order_twap,
)


def execute_portfolio_plan(plan: dict, dry_run: bool = True, slices: int = 1) -> dict:
    if not plan.get("approved"):
        return {"executed": False, "reason": "Portfolio Council approved no trades.", "orders": []}
    if dry_run:
        return {"executed": False, "dry_run": True, "orders": plan["trades"]}

    account = get_account_info()
    buying_power = float(account["buying_power"])
    held = {p["symbol"] for p in get_open_positions()}
    pending = {o.get("symbol") for o in get_open_orders()}
    orders = []
    for trade in plan["trades"]:
        symbol = trade["symbol"]
        if symbol in held or symbol in pending:
            orders.append({"symbol": symbol, "accepted": False, "reason": "position/order already exists"})
            continue
        notional = min(float(trade["notional_usd"]), buying_power)
        price = get_latest_price(symbol)
        qty = round(notional / price, 4)
        if qty <= 0:
            continue
        client_tag = f"bulk-{uuid.uuid4().hex[:20]}"
        try:
            fills = slice_order_twap(
                symbol, qty, trade["side"], slices=slices, whole_shares=trade["side"] == "sell",
                client_order_id=client_tag,
            )
            orders.append({"symbol": symbol, "accepted": bool(fills), "orders": fills})
            buying_power = max(0.0, buying_power - notional)
        except Exception as exc:
            orders.append({"symbol": symbol, "accepted": False, "reason": str(exc)})
    return {"executed": any(o.get("accepted") for o in orders), "orders": orders}


def execute_exit_plan(exit_plan: dict, dry_run: bool = True) -> dict:
    if not exit_plan.get("approved"):
        return {"executed": False, "orders": []}
    if dry_run:
        return {"executed": False, "dry_run": True, "orders": exit_plan["exits"]}
    orders = []
    for exit_order in exit_plan["exits"]:
        try:
            order = submit_order(
                symbol=exit_order["symbol"],
                qty=exit_order["qty"],
                side=exit_order["side"],
                client_order_id=f"exit-{uuid.uuid4().hex[:20]}",
            )
            orders.append({**exit_order, "accepted": True, "order": order})
        except Exception as exc:
            orders.append({**exit_order, "accepted": False, "reason": str(exc)})
    return {"executed": any(item["accepted"] for item in orders), "orders": orders}
