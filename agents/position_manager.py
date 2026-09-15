"""Position management for the 30-minute cycle.

Existing positions are evaluated before new entries. Exits are deliberately
mechanical: a breached ATR stop or a strong opposite/flat research signal
produces a reduce/close instruction. This module never talks to Alpaca.
"""

from __future__ import annotations


def build_exit_plan(
    positions: list[dict],
    research: list[dict],
    open_orders: list[dict],
) -> dict:
    by_symbol = {item["symbol"]: item for item in research}
    pending = {item.get("symbol") for item in open_orders}
    exits = []
    for position in positions:
        symbol = position["symbol"]
        signal = by_symbol.get(symbol)
        if not signal or symbol in pending:
            continue
        entry = float(position["avg_entry_price"])
        last = float(signal["price"])
        atr = float(signal["atr"])
        qty = abs(float(position["qty"]))
        long_position = float(position["qty"]) > 0
        stop = entry - 2 * atr if long_position else entry + 2 * atr
        stop_breached = last <= stop if long_position else last >= stop
        opposite = (
            long_position and signal["direction"] == "short"
        ) or (
            not long_position and signal["direction"] == "long"
        )
        flat = signal["direction"] == "flat"
        if stop_breached or opposite or flat:
            exits.append({
                "symbol": symbol,
                "side": "sell" if long_position else "buy",
                "qty": qty,
                "reason": "atr_stop" if stop_breached else "opposite_signal" if opposite else "flat_signal",
                "reference_price": last,
                "stop_price": stop,
                "estimated_pl": float(position.get("unrealized_pl", 0.0)),
            })
    return {"approved": bool(exits), "exits": exits}
