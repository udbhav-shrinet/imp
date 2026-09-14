"""
Team S: Execution (the "Main Guy").

The single agent authorized to place trades. It reviews Team A's
risk-approved report and, only if `approved` is True, executes a paper
trade via the Alpaca Trading API. All authentication is handled inside
`tools.alpaca_tools`, which reads credentials from environment variables
(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL) -- no keys are
ever hardcoded here.

Execution style: orders are sliced TWAP-style (time-weighted average
price) to simulate real-world slippage avoidance, even against the
paper trading endpoint.
"""

from crewai import Agent

from tools.alpaca_tools import get_account_info, get_latest_price, slice_order_twap


execution_agent = Agent(
    role="Execution Trader",
    goal=(
        "Execute Team A's risk-approved trade thesis as a paper trade via Alpaca, "
        "slicing large orders TWAP-style, and never act on a report Team A did not approve."
    ),
    backstory=(
        "The final gatekeeper before capital moves. Strictly mechanical: if the "
        "Risker vetoed a thesis, this agent refuses to trade, no exceptions."
    ),
    allow_delegation=False,
    verbose=True,
)


def execute_team_a_report(report: dict, twap_slices: int = 4) -> dict:
    """
    Execute (or reject) Team A's final report.

    Strict hierarchy enforcement: this is the only function in the entire
    system that calls `submit_order` / `slice_order_twap`, and it refuses
    to act unless Team A's Risker explicitly approved the trade.
    """
    if not report.get("approved"):
        return {
            "executed": False,
            "reason": report.get("veto_reason", "Trade not approved by Team A."),
        }

    symbol = report["symbol"]
    direction = report["direction"]
    position_size_usd = report["position_size_usd"]

    if position_size_usd <= 0:
        return {"executed": False, "reason": "Approved position size is zero."}

    price = get_latest_price(symbol)

    # Team A sizes each thesis independently, so on a small account several
    # approved theses in the same batch can collectively exceed what's
    # actually left to spend. Re-check live buying power here -- the one
    # place that actually knows the running total -- and clamp down to it
    # rather than let the broker reject the order outright.
    try:
        buying_power = get_account_info()["buying_power"]
        affordable_usd = min(position_size_usd, buying_power)
    except Exception:
        affordable_usd = position_size_usd  # can't verify live -- fall through to Alpaca's own check

    # Whole shares only -- Alpaca rejects fractional-share orders on the
    # short side outright, and fractional orders carry other API
    # restrictions besides.
    qty = int(affordable_usd // price)
    if qty < 1:
        reason = (
            f"Buying power (${affordable_usd:.2f}) buys less than one whole share of {symbol} at ${price:.2f}."
            if affordable_usd < position_size_usd
            else f"Position size (${position_size_usd:.2f}) buys less than one whole share at ${price:.2f}."
        )
        return {"executed": False, "reason": reason}

    side = "buy" if direction == "long" else "sell"
    try:
        fills = slice_order_twap(symbol=symbol, total_qty=qty, side=side, slices=twap_slices)
    except Exception as exc:
        return {"executed": False, "reason": f"Order rejected by broker: {exc}"}

    if not fills:
        return {"executed": False, "reason": "No child orders were accepted by the broker."}

    return {
        "executed": True,
        "symbol": symbol,
        "side": side,
        "total_qty": qty,
        "reference_price": price,
        "child_orders": fills,
    }
