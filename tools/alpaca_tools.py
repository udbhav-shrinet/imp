"""
Alpaca trading tools used exclusively by Team S (Execution).

All credentials are loaded dynamically from environment variables via
python-dotenv. No API keys are ever hardcoded. All calls target the
paper trading endpoint (APCA_API_BASE_URL) so no real capital is at risk.
"""

import os

from alpaca_trade_api.rest import REST
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))


def get_alpaca_client() -> REST:
    """Build an authenticated Alpaca REST client from environment variables."""
    # .strip() guards against stray whitespace/tab characters that can sneak
    # in when a secret is copy-pasted into a CI provider's UI.
    base_url = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").strip()
    key_id = os.environ["APCA_API_KEY_ID"].strip()
    secret_key = os.environ["APCA_API_SECRET_KEY"].strip()

    if "paper-api" not in base_url:
        raise RuntimeError(
            "Refusing to trade: APCA_API_BASE_URL must point at the paper "
            "trading endpoint (https://paper-api.alpaca.markets)."
        )

    return REST(key_id=key_id, secret_key=secret_key, base_url=base_url)


def get_account_info() -> dict:
    """Return current paper account equity, buying power, and cash."""
    client = get_alpaca_client()
    account = client.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
    }


def get_latest_price(symbol: str) -> float:
    """Fetch the latest trade price for a symbol."""
    client = get_alpaca_client()
    trade = client.get_latest_trade(symbol)
    return float(trade.price)


def submit_order(
    symbol: str,
    qty: float,
    side: str,
    order_type: str = "market",
    time_in_force: str = "day",
    limit_price: float | None = None,
) -> dict:
    """
    Submit an order through the Alpaca paper trading API.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        qty: Number of shares (fractional allowed if account supports it).
        side: "buy" or "sell".
        order_type: "market", "limit", "stop", or "stop_limit".
        time_in_force: "day", "gtc", "opg", "cls", "ioc", or "fok".
        limit_price: Required when order_type is "limit" or "stop_limit".
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if qty <= 0:
        raise ValueError(f"qty must be positive, got {qty!r}")

    client = get_alpaca_client()
    order = client.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        time_in_force=time_in_force,
        limit_price=limit_price,
    )
    return {
        "id": order.id,
        "symbol": order.symbol,
        "qty": order.qty,
        "side": order.side,
        "type": order.type,
        "status": order.status,
        "submitted_at": str(order.submitted_at),
    }


def slice_order_twap(symbol: str, total_qty: int, side: str, slices: int = 4) -> list[dict]:
    """
    Split a large order into `slices` equal child orders (TWAP-style) to
    simulate real-world slippage avoidance, even on paper trades.

    Quantities are whole shares: Alpaca rejects fractional-share orders
    on the short side outright ("fractional orders cannot be sold
    short"), and fractional orders carry other restrictions (order
    types, bracket orders) that whole shares sidestep entirely.
    """
    if slices < 1:
        raise ValueError("slices must be >= 1")

    total_qty = int(total_qty)
    if total_qty < 1:
        return []

    per_slice_qty = total_qty // slices
    remainder = total_qty - per_slice_qty * slices

    results = []
    for i in range(slices):
        qty = per_slice_qty + (remainder if i == slices - 1 else 0)
        if qty <= 0:
            continue
        results.append(submit_order(symbol=symbol, qty=qty, side=side))
    return results


def get_open_positions() -> list[dict]:
    """Return all open paper positions."""
    client = get_alpaca_client()
    positions = client.list_positions()
    return [
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
        }
        for p in positions
    ]
