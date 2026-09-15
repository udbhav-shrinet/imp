"""Portfolio Council: turns independent research into one risk-constrained plan."""

from __future__ import annotations

import numpy as np


def _portfolio_tail_risk(selected: list[dict], notionals: list[float]) -> tuple[float, float]:
    """Historical 95% portfolio VaR/CVaR using equally dated return samples."""
    if not selected:
        return 0.0, 0.0
    length = min(len(item.get("recent_returns", [])) for item in selected)
    if length < 20:
        return 0.0, 0.0
    weights = [notional / max(sum(notionals), 1e-9) for notional in notionals]
    matrix = [
        item["recent_returns"][-length:]
        for item in selected
    ]
    portfolio_returns = [
        sum(weights[col] * matrix[col][row] for col in range(len(matrix)))
        for row in range(length)
    ]
    percentile = float(np.percentile(portfolio_returns, 5))
    losses = [value for value in portfolio_returns if value <= percentile]
    var = max(0.0, -percentile)
    cvar = max(0.0, -sum(losses) / len(losses)) if losses else var
    return var, cvar


def build_portfolio_plan(
    research: list[dict],
    account: dict,
    positions: list[dict],
    open_orders: list[dict],
    max_positions: int = 15,
    max_new_positions: int = 5,
    max_position_pct: float = 0.03,
    max_new_exposure_pct: float = 0.10,
    allow_shorts: bool = False,
    target_annual_volatility: float = 0.12,
    max_portfolio_var: float = 0.04,
    max_portfolio_cvar: float = 0.06,
    max_correlation: float = 0.85,
) -> dict:
    """Rank signals, exclude conflicts, and allocate a bounded batch.

    Allocation is equal-risk capped:
        notional_i = min(equity * max_position_pct,
                         equity * max_new_exposure_pct / selected_count)
    This is intentionally conservative until a strategy-specific backtest exists.
    """
    equity = float(account.get("equity", 0))
    held = {p["symbol"] for p in positions}
    pending = {o.get("symbol") for o in open_orders}
    candidates = [
        r for r in research
        if r["direction"] != "flat"
        and r["symbol"] not in held
        and r["symbol"] not in pending
        and (allow_shorts or r["direction"] != "short")
        and r["volatility"] > 0
        and r.get("expected_edge", 0) > r.get("estimated_round_trip_cost", 1)
    ]
    candidates.sort(
        key=lambda item: abs(item["score"])
        * min(1.0, target_annual_volatility / max(item["volatility"], 0.05))
        / max(item.get("estimated_round_trip_cost", 0.01), 0.0001),
        reverse=True,
    )
    room = max(0, min(max_new_positions, max_positions - len(held)))
    batch_cap = equity * max_new_exposure_pct
    per_trade_cap = min(equity * max_position_pct, batch_cap / max(room, 1))
    selected = []
    for candidate in candidates:
        if len(selected) >= room:
            break
        if any(
            candidate["symbol"] != other["symbol"]
            and _correlation(candidate.get("recent_returns", []), other.get("recent_returns", [])) > max_correlation
            for other in selected
        ):
            continue
        selected.append(candidate)
    trades = []
    for item in selected:
        stop_distance = max(item["atr"] * 2.0, item["price"] * 0.01)
        volatility_scale = min(1.0, target_annual_volatility / max(item["volatility"], 0.05))
        notional = per_trade_cap * volatility_scale
        trades.append({
            "symbol": item["symbol"],
            "side": "buy" if item["direction"] == "long" else "sell",
            "notional_usd": round(notional, 2),
            "reference_price": item["price"],
            "stop_distance": round(stop_distance, 4),
            "research_score": item["score"],
            "risk_fraction": round(notional / max(equity, 1), 5),
            "estimated_round_trip_cost": item["estimated_round_trip_cost"],
            "expected_edge": item["expected_edge"],
            "reason": item["reasons"],
        })
    notionals = [trade["notional_usd"] for trade in trades]
    portfolio_var, portfolio_cvar = _portfolio_tail_risk(selected, notionals)
    risk_approved = portfolio_var <= max_portfolio_var and portfolio_cvar <= max_portfolio_cvar
    if not risk_approved:
        trades = []
    return {
        "approved": bool(trades) and risk_approved,
        "trades": trades,
        "rejected_count": len(research) - len(selected),
        "portfolio_var_95": portfolio_var,
        "portfolio_cvar_95": portfolio_cvar,
        "risk_approved": risk_approved,
        "constraints": {
            "max_positions": max_positions,
            "max_new_positions": max_new_positions,
            "max_position_pct": max_position_pct,
            "max_new_exposure_pct": max_new_exposure_pct,
            "allow_shorts": allow_shorts,
            "target_annual_volatility": target_annual_volatility,
            "max_portfolio_var": max_portfolio_var,
            "max_portfolio_cvar": max_portfolio_cvar,
            "max_correlation": max_correlation,
        },
    }


def _correlation(first: list[float], second: list[float]) -> float:
    import numpy as np

    length = min(len(first), len(second))
    if length < 20:
        return 0.0
    value = np.corrcoef(first[-length:], second[-length:])[0, 1]
    return float(value) if np.isfinite(value) else 0.0
