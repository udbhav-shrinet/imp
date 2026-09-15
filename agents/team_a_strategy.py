"""
Team A: Strategy & Risk.

Two agents that receive Team B's quantitative output. The Trader
synthesizes a trade thesis; the Risker sizes it with the Kelly Criterion
and VaR/CVaR, holding veto power over any thesis that breaches risk
limits. Only a Risker-approved report is passed on to Team S for
execution -- Team A never places trades itself.

Theories applied:
  - Trader: statistical arbitrage / pairs-trading style synthesis of
    directional probability and mean-reversion signals into a thesis,
    gated by the macro regime score from Team B's Economist.
  - Risker: Kelly Criterion position sizing (with a dynamic win/loss ratio
    derived from the symbol's own historical return distribution, not a
    static assumption), Value at Risk (VaR) / Conditional VaR (CVaR),
    GARCH(1,1)-style volatility tightening.
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


DEFAULT_WIN_LOSS_RATIO = 1.5  # fallback when a symbol has too little history to derive one


def compute_historical_win_loss_ratio(returns: np.ndarray) -> float:
    """
    Dynamic half-Kelly win/loss ratio: average magnitude of up days over
    average magnitude of down days, across the 2-year daily return history
    -- replacing the previous static assumption of 1.5 with the symbol's
    own realized win/loss behavior.

    This is a realized-return proxy for "average win / average loss"
    rather than a literal day-by-day re-simulation of the Trader's thesis
    rule across 2 years of history: doing that would mean re-running the
    full ML/GBM/OU pipeline once per historical trading day, per symbol --
    on a ~57-76 symbol pool per run, that would multiply this run's compute
    cost by roughly 500x and blow through GitHub Actions' free-tier
    minutes. The realized win/loss magnitude is a defensible, much cheaper
    stand-in: it's exactly the quantity Kelly sizing needs (b = avg win /
    avg loss), just computed from actual historical outcomes instead of a
    guess.
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return DEFAULT_WIN_LOSS_RATIO
    avg_win = float(wins.mean())
    avg_loss = float(-losses.mean())
    if avg_loss <= 0:
        return DEFAULT_WIN_LOSS_RATIO
    return avg_win / avg_loss


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

MACRO_HOSTILE_THRESHOLD = -0.3  # below this, a long thesis is vetoed outright regardless of P(up)
MACRO_DOWNGRADE_CONFIDENCE_FACTOR = 0.5  # borderline macro (0 to MACRO_HOSTILE_THRESHOLD) halves conviction


def form_trade_thesis(team_b_output: dict) -> dict:
    """
    Trader logic: synthesize a directional thesis from Team B's output.

    A long candidate (P(up) > 0.55 in a bull/sideways regime) is gated by
    the Economist's macro_score:
        macro_score > 0                          -> full-conviction long
        MACRO_HOSTILE_THRESHOLD < macro_score <= 0 -> long, but downgraded
                                                       conviction (macro is
                                                       lukewarm, not yet a
                                                       reason to skip)
        macro_score <= MACRO_HOSTILE_THRESHOLD    -> vetoed to no_trade
                                                       (macro is outright
                                                       hostile to a long)
    Short candidates aren't macro-gated the same way: a hostile macro
    regime is generally *supportive* of a short thesis, not a reason to
    downgrade it.
    """
    probability_up = team_b_output["ml_engineer"]["probability_up_next_day"]
    regime = team_b_output["mathematician"]["market_regime"]
    ou = team_b_output["mathematician"]["ornstein_uhlenbeck"]
    rsi = team_b_output["team_c_statistics"].get("rsi_14")
    macro_score = team_b_output["economist"]["macro_score"]

    confidence = abs(probability_up - 0.5) * 2  # 0..1
    macro_note = ""

    if probability_up > 0.55 and regime in ("bull", "sideways"):
        if macro_score > 0:
            direction = "long"
        elif macro_score > MACRO_HOSTILE_THRESHOLD:
            direction = "long"
            confidence *= MACRO_DOWNGRADE_CONFIDENCE_FACTOR
            macro_note = f" Conviction downgraded -- lukewarm macro (macro_score={macro_score:.2f})."
        else:
            direction = "no_trade"
            macro_note = f" Long thesis vetoed -- hostile macro (macro_score={macro_score:.2f})."
    elif probability_up < 0.45 and regime in ("bear", "sideways"):
        direction = "short"
    else:
        direction = "no_trade"

    return {
        "symbol": team_b_output["symbol"],
        "direction": direction,
        "confidence": confidence,
        "regime": regime,
        "mean_reversion_speed": ou["theta"],
        "rsi_14": rsi,
        "macro_score": macro_score,
        "rationale": (
            f"P(up)={probability_up:.2f} in a '{regime}' regime with OU reversion "
            f"speed theta={ou['theta']:.3f}, macro_score={macro_score:.2f}.{macro_note}"
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
    win_loss_ratio = compute_historical_win_loss_ratio(returns)
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
        "win_loss_ratio": win_loss_ratio,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "garch_volatility": garch_vol,
        "position_size_usd": round(position_size_usd, 2),
    }


def build_team_a_report(team_b_output: dict, team_c_output: dict, allocated_capital: float) -> dict:
    thesis = form_trade_thesis(team_b_output)
    return apply_risk_management(thesis, team_c_output, allocated_capital)
