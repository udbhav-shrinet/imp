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

from tools.alpaca_tools import get_latest_price, slice_order_twap


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
    qty = round(position_size_usd / price, 4)
    if qty <= 0:
        return {"executed": False, "reason": "Computed order quantity is zero."}

    side = "buy" if direction == "long" else "sell"
    fills = slice_order_twap(symbol=symbol, total_qty=qty, side=side, slices=twap_slices)

    return {
        "executed": True,
        "symbol": symbol,
        "side": side,
        "total_qty": qty,
        "reference_price": price,
        "child_orders": fills,
    }
