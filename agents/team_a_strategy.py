"""
Team A: Strategy & Risk.

Two agents that receive Team B's quantitative output. The Trader
synthesizes a trade thesis; the Risker sizes it with the Kelly Criterion
and VaR/CVaR, holding veto power over any thesis that breaches risk
limits. Only a Risker-approved report is passed on to Team S for
execution -- Team A never places trades itself.

Theories applied:
  - Trader: statistical arbitrage / pairs-trading style synthesis of
    directional probability and mean-reversion signals into a thesis.
  - Risker: Kelly Criterion position sizing, Value at Risk (VaR) /
    Conditional VaR (CVaR), GARCH(1,1)-style volatility tightening.
"""

import numpy as np
from crewai import Agent


# --------------------------------------------------------------------------
# Risk math
# --------------------------------------------------------------------------

def kelly_fraction(win_probability: float, win_loss_ratio: float) -> float:
    """
    Kelly Criterion: f* = p - (1 - p) / b
    where p = win probability, b = win/loss ratio (avg win / avg loss).
    Clamped to [0, 1] and then halved ("half-Kelly") for a conservative bet size.
    """
    p = win_probability
    b = max(win_loss_ratio, 1e-6)
    f_star = p - (1 - p) / b
    f_star = float(np.clip(f_star, 0.0, 1.0))
    return f_star * 0.5  # half-Kelly for safety margin


def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical (empirical) Value at Risk at the given confidence level."""
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def historical_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall: mean loss beyond the VaR threshold."""
    var = historical_var(returns, confidence)
    tail_losses = returns[returns <= -var]
    if len(tail_losses) == 0:
        return var
    return float(-tail_losses.mean())


def estimate_garch_volatility(returns: np.ndarray, omega: float = 1e-6, alpha: float = 0.1, beta: float = 0.85) -> float:
    """
    Simplified GARCH(1,1) forward-looking volatility estimate:
        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    Seeded with the sample variance, iterated across the return series.
    """
    if len(returns) < 2:
        return float(np.std(returns)) if len(returns) else 0.0

    variance = np.var(returns)
    for r in returns:
        variance = omega + alpha * r ** 2 + beta * variance
    return float(np.sqrt(variance))


# --------------------------------------------------------------------------
# CrewAI Agents
# --------------------------------------------------------------------------

trader_agent = Agent(
    role="Trader",
    goal=(
        "Synthesize Team B's mathematical, ML, and macro output into a directional "
        "trade thesis (long, short, or no trade), grounded in statistical arbitrage "
        "and mean-reversion / co-integration reasoning."
    ),
    backstory=(
        "A discretionary trader who uses quant signals as evidence, not gospel, and "
        "always states an explicit thesis with a target holding horizon."
    ),
    allow_delegation=True,
    verbose=True,
)

risker_agent = Agent(
    role="Risker",
    goal=(
        "Size the Trader's thesis using the Kelly Criterion and validate it against "
        "VaR/CVaR limits and GARCH-implied volatility. Veto the trade outright if "
        "risk limits are breached."
    ),
    backstory=(
        "A risk manager who has read Taleb on fat tails and treats every position "
        "as a potential black swan until proven otherwise. Has final veto authority "
        "over Team A's outgoing report."
    ),
    allow_delegation=False,
    verbose=True,
)


VAR_LIMIT = 0.05   # max acceptable 1-day 95% VaR as a fraction of position value
MAX_POSITION_FRACTION = 0.25  # never risk more than 25% of allocated capital on one thesis


def form_trade_thesis(team_b_output: dict) -> dict:
    """Trader logic: synthesize a directional thesis from Team B's output."""
    probability_up = team_b_output["ml_engineer"]["probability_up_next_day"]
    regime = team_b_output["mathematician"]["market_regime"]
    ou = team_b_output["mathematician"]["ornstein_uhlenbeck"]
    rsi = team_b_output["team_c_statistics"].get("rsi_14")

    if probability_up > 0.55 and regime in ("bull", "sideways"):
        direction = "long"
    elif probability_up < 0.45 and regime in ("bear", "sideways"):
        direction = "short"
    else:
        direction = "no_trade"

    return {
        "symbol": team_b_output["symbol"],
        "direction": direction,
        "confidence": abs(probability_up - 0.5) * 2,  # 0..1
        "regime": regime,
        "mean_reversion_speed": ou["theta"],
        "rsi_14": rsi,
        "rationale": (
            f"P(up)={probability_up:.2f} in a '{regime}' regime with OU reversion "
            f"speed theta={ou['theta']:.3f}."
        ),
    }


def apply_risk_management(thesis: dict, team_c_output: dict, allocated_capital: float) -> dict:
    """
    Risker logic: size the thesis with Kelly, check VaR/CVaR, and veto if
    limits are breached. Returns the final Team A report handed to Team S.
    """
    returns = team_c_output["price_history"]["Close"].pct_change().dropna().values

    if thesis["direction"] == "no_trade":
        return {**thesis, "approved": False, "veto_reason": "No directional edge identified.", "position_size_usd": 0.0}

    win_probability = thesis["confidence"] / 2 + 0.5
    win_loss_ratio = 1.5  # assumed avg win / avg loss; refine with historical trade log in production
    kelly_size_fraction = kelly_fraction(win_probability, win_loss_ratio)

    var_95 = historical_var(returns)
    cvar_95 = historical_cvar(returns)
    garch_vol = estimate_garch_volatility(returns)

    position_fraction = min(kelly_size_fraction, MAX_POSITION_FRACTION)
    position_size_usd = allocated_capital * position_fraction

    veto = var_95 > VAR_LIMIT
    if veto:
        return {
            **thesis,
            "approved": False,
            "veto_reason": f"1-day 95% VaR ({var_95:.3f}) exceeds limit ({VAR_LIMIT}).",
            "var_95": var_95,
            "cvar_95": cvar_95,
            "garch_volatility": garch_vol,
            "position_size_usd": 0.0,
        }

    return {
        **thesis,
        "approved": True,
        "veto_reason": None,
        "kelly_fraction": kelly_size_fraction,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "garch_volatility": garch_vol,
        "position_size_usd": round(position_size_usd, 2),
    }


def build_team_a_report(team_b_output: dict, team_c_output: dict, allocated_capital: float) -> dict:
    thesis = form_trade_thesis(team_b_output)
    return apply_risk_management(thesis, team_c_output, allocated_capital)
