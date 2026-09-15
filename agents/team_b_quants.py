"""
Team B: Quants.

Three agents that receive Team C's signal packet and translate it into
probabilistic and regime-aware quantitative context. Their combined
output is handed up to Team A (Strategy). No agent in this team is
permitted to size or place trades.

Theories applied:
  - Mathematician: Ito's Lemma / Geometric Brownian Motion (GBM) price
    path simulation, Ornstein-Uhlenbeck mean reversion, Markov-chain
    regime detection (bull / bear / sideways), damped by the macro regime
    when reversion is weak (see apply_macro_regime_adjustment).
  - ML Engineer: Random Forest probability-of-direction model with
    walk-forward validation (purged, to avoid time leakage), trained on a
    rolling 252-day window with a macro_score feature aligned by date.
  - Economist: quantitative macro regime score from FRED's yield curve
    spread (T10Y2Y) and VIX (VIXCLS), normalized to [-1, 1].
"""

import numpy as np
import pandas as pd
from crewai import Agent
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------------------------------
# Mathematician: stochastic calculus
# --------------------------------------------------------------------------

def simulate_gbm_paths(spot: float, mu: float, sigma: float, days: int = 20, n_paths: int = 1000) -> np.ndarray:
    """
    Simulate future price paths under Geometric Brownian Motion:
        dS = mu * S * dt + sigma * S * dW   (Ito's Lemma applied to log S)
    """
    dt = 1 / 252
    rng = np.random.default_rng()
    shocks = rng.normal(0, 1, size=(n_paths, days))
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * shocks
    cum_log_returns = np.cumsum(log_returns, axis=1)
    return spot * np.exp(cum_log_returns)


def fit_ornstein_uhlenbeck(prices: pd.Series) -> dict:
    """
    Fit a discretized Ornstein-Uhlenbeck process to detect mean reversion:
        dX = theta * (mu - X) * dt + sigma * dW
    Returns the estimated speed of reversion (theta), long-run mean (mu),
    and volatility (sigma).
    """
    x = prices.values
    x_lag, x_now = x[:-1], x[1:]
    dt = 1 / 252

    # OLS regression: x_now = a + b * x_lag  =>  theta = -ln(b)/dt, mu = a/(1-b)
    b, a = np.polyfit(x_lag, x_now, 1)
    b = np.clip(b, 1e-6, 0.999999)
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    residuals = x_now - (a + b * x_lag)
    sigma = np.std(residuals) * np.sqrt(2 * theta / (1 - b ** 2)) if theta > 0 else np.std(residuals)

    return {"theta": float(theta), "mu": float(mu), "sigma": float(sigma)}


def detect_market_regime(prices: pd.Series, n_states: int = 3) -> str:
    """
    Detect the current market regime (bull / bear / sideways) using a
    simple Markov-chain-style classification over rolling returns.
    A full Hidden Markov Model can replace this with `hmmlearn` in
    production; this keeps the dependency footprint minimal.
    """
    returns = prices.pct_change().dropna()
    recent_mean = returns.tail(20).mean()
    recent_std = returns.tail(20).std()

    if recent_mean > recent_std * 0.5:
        return "bull"
    elif recent_mean < -recent_std * 0.5:
        return "bear"
    return "sideways"


MACRO_BEARISH_THRESHOLD = -0.3
WEAK_REVERSION_THETA_THRESHOLD = 1.0  # annualized theta below this = weak/slow mean reversion
DRIFT_DAMPING_FACTOR = 0.5  # halve the GBM drift under hostile-macro + weak-reversion conditions


def apply_macro_regime_adjustment(mu: float, theta: float, macro_score: float) -> dict:
    """
    Concrete rule combining the GBM drift estimate with OU reversion speed
    and the macro regime: hostile macro alone isn't reason to distrust a
    drift estimate if the series is strongly mean-reverting (high theta) --
    reversion dominates before a macro-driven regime shift would matter.
    But hostile macro *and* weak reversion (low theta -- the series behaves
    more like a persistent random walk) means the drift estimate is more
    exposed to a real regime change, so it gets damped.
    """
    damped = macro_score < MACRO_BEARISH_THRESHOLD and theta < WEAK_REVERSION_THETA_THRESHOLD
    adjusted_mu = mu * DRIFT_DAMPING_FACTOR if damped else mu
    return {"adjusted_mu": float(adjusted_mu), "damped": damped}


# --------------------------------------------------------------------------
# ML Engineer: probability of direction
# --------------------------------------------------------------------------

def _rolling_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def align_macro_score(dates: pd.Index, macro_score_history: pd.Series, fallback: float) -> pd.Series:
    """
    As-of join: for each date in `dates`, the most recent macro_score at or
    before that date, forward-filled across FRED's non-trading-day gaps.
    Falls back to `fallback` (typically today's live snapshot score)
    wherever no historical macro reading exists yet -- e.g. no FRED_API_KEY
    configured, or dates older than the cached history.
    """
    if macro_score_history.empty:
        return pd.Series(fallback, index=dates)

    # yfinance returns a tz-aware index (America/New_York); FRED dates are
    # plain calendar dates, so the macro series is tz-naive. Comparing the
    # two directly raises ("Cannot compare tz-naive and tz-aware
    # timestamps"), so both sides are normalized to a plain calendar date
    # before the as-of join, then the caller's original index is restored
    # so the result assigns back onto the price rows it came from.
    target = pd.DatetimeIndex(dates)
    if target.tz is not None:
        target = target.tz_localize(None)
    target = target.normalize()

    reindexed = macro_score_history.reindex(macro_score_history.index.union(target)).sort_index().ffill()
    aligned = reindexed.reindex(target).fillna(fallback)
    aligned.index = dates
    return aligned


def build_features(
    price_history: pd.DataFrame,
    macro_score_series: pd.Series | None = None,
    current_macro_score: float = 0.0,
) -> pd.DataFrame:
    df = price_history.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["rsi_14"] = _rolling_rsi(df["Close"])
    df["momentum_10d"] = df["Close"] / df["Close"].shift(10) - 1
    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    # Each historical row gets the macro_score that was actually live on
    # that date (not today's snapshot repeated across every row) -- see
    # align_macro_score(). Falls back to `current_macro_score` wherever no
    # FRED history is available at all.
    if macro_score_series is not None:
        df["macro_score"] = align_macro_score(df.index, macro_score_series, fallback=current_macro_score)
    else:
        df["macro_score"] = current_macro_score
    return df.dropna()


def walk_forward_predict_probability(
    price_history: pd.DataFrame,
    macro_score_series: pd.Series | None = None,
    current_macro_score: float = 0.0,
    train_window: int = 252,
) -> float:
    """
    Predict P(next-day up move) using a Random Forest with walk-forward
    validation: train only on data strictly preceding the prediction
    point, avoiding the look-ahead bias that standard K-fold CV creates
    in time-series data (purged cross-validation principle). Trains on a
    rolling 252-day (1 trading year) window across the 2-year history.
    """
    features = build_features(price_history, macro_score_series, current_macro_score)
    feature_cols = ["return_1d", "return_5d", "volatility_10d", "rsi_14", "momentum_10d", "macro_score"]

    if len(features) < train_window + 1:
        train_window = max(len(features) - 1, 1)

    train = features.iloc[-(train_window + 1):-1]
    latest = features.iloc[[-1]]

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(train[feature_cols], train["target_up"])
    probability_up = model.predict_proba(latest[feature_cols])[0][1]
    return float(probability_up)


# --------------------------------------------------------------------------
# CrewAI Agents
# --------------------------------------------------------------------------

mathematician_agent = Agent(
    role="Mathematician",
    goal=(
        "Apply stochastic calculus (Ito's Lemma, Geometric Brownian Motion, "
        "Ornstein-Uhlenbeck) and Markov-chain regime detection to Team C's data."
    ),
    backstory=(
        "A former quant researcher who thinks in stochastic differential equations "
        "and treats every price series as a realization of a random process."
    ),
    allow_delegation=False,
    verbose=True,
)

ml_engineer_agent = Agent(
    role="ML Engineer",
    goal=(
        "Predict the probability of an upward price move using Random Forest / "
        "XGBoost-style models validated with walk-forward, purged cross-validation."
    ),
    backstory=(
        "A machine learning engineer obsessed with avoiding time-leakage in "
        "financial models, having learned the hard way that standard K-fold CV lies."
    ),
    allow_delegation=False,
    verbose=True,
)

economist_agent = Agent(
    role="Economist",
    goal=(
        "Score the current macro regime from real FRED data -- the T10Y2Y yield "
        "curve spread and the VIX -- as a normalized macro_score in [-1, 1], so the "
        "Trader and Risker can gate conviction on real-world risk conditions."
    ),
    backstory=(
        "A macro strategist who refuses to let a trade thesis stand on price action "
        "alone -- rates and volatility regime come first."
    ),
    allow_delegation=False,
    verbose=True,
)


def compute_macro_score(t10y2y: float | None, vixcls: float | None) -> float:
    """
    Normalize the yield curve spread and VIX level into a single macro
    regime score in [-1, 1] (+1 = bullish risk-on, -1 = bearish risk-off).
    A missing input contributes a neutral 0 rather than skewing the score
    toward either extreme.

        T10Y2Y < 0      -> -1.0 (inverted curve, late-cycle/recession risk)
        T10Y2Y > 0.5     -> +1.0 (steepening, expansion)
        otherwise        -> linear interpolation between the two

        VIXCLS > 25      -> -1.0 (risk-off, turbulent)
        VIXCLS < 18      -> +1.0 (risk-on, calm)
        otherwise        -> linear interpolation between the two

    The two components are weighted equally and averaged.
    """
    yield_component = 0.0
    if t10y2y is not None:
        if t10y2y < 0:
            yield_component = -1.0
        elif t10y2y > 0.5:
            yield_component = 1.0
        else:
            yield_component = t10y2y / 0.5

    vix_component = 0.0
    if vixcls is not None:
        if vixcls > 25:
            vix_component = -1.0
        elif vixcls < 18:
            vix_component = 1.0
        else:
            vix_component = 1.0 - 2.0 * (vixcls - 18) / (25 - 18)

    return float(np.clip(0.5 * yield_component + 0.5 * vix_component, -1.0, 1.0))


def build_macro_score_history(macro_history: dict | None) -> pd.Series:
    """
    Combine T10Y2Y/VIXCLS daily histories (each a {date_str: value} dict, as
    returned by tools.fred_tools.get_macro_history) into a daily macro_score
    series indexed by date -- used to align a historical macro reading to
    each row of the ML Engineer's training window, instead of repeating
    today's snapshot across every historical row.
    """
    macro_history = macro_history or {}
    t10y2y_hist = macro_history.get("t10y2y") or {}
    vixcls_hist = macro_history.get("vixcls") or {}
    if not t10y2y_hist and not vixcls_hist:
        return pd.Series(dtype=float)

    all_dates = sorted(set(t10y2y_hist) | set(vixcls_hist))
    t10y2y_series = pd.Series({d: t10y2y_hist.get(d) for d in all_dates}, dtype=float).ffill()
    vixcls_series = pd.Series({d: vixcls_hist.get(d) for d in all_dates}, dtype=float).ffill()

    scores = {d: compute_macro_score(t10y2y_series.get(d), vixcls_series.get(d)) for d in all_dates}
    series = pd.Series(scores)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def analyze_team_b(
    team_c_output: dict,
    macro_snapshot: dict | None = None,
    macro_score_history: pd.Series | None = None,
) -> dict:
    """
    Orchestrate Team B's three agents' underlying logic on Team C's signal
    packet.

    `macro_snapshot` (tools.fred_tools.get_macro_snapshot()) and
    `macro_score_history` (build_macro_score_history() over
    tools.fred_tools.get_macro_history()) are fetched **once per pipeline
    run** by main.py -- not per symbol, since macro data doesn't vary by
    symbol -- and threaded through to every symbol's Team B call.
    """
    price_history = team_c_output["price_history"]
    close = price_history["Close"]
    returns = close.pct_change().dropna()

    macro_snapshot = macro_snapshot or {}
    macro_score = compute_macro_score(macro_snapshot.get("t10y2y"), macro_snapshot.get("vixcls"))

    spot = float(close.iloc[-1])
    mu = float(returns.mean() * 252)
    sigma = float(returns.std() * np.sqrt(252))
    ou = fit_ornstein_uhlenbeck(close)
    regime_adjustment = apply_macro_regime_adjustment(mu, ou["theta"], macro_score)

    mathematician_output = {
        "gbm_simulated_paths_sample": simulate_gbm_paths(
            spot, regime_adjustment["adjusted_mu"], sigma, days=20, n_paths=200
        )[:5].tolist(),
        "ornstein_uhlenbeck": ou,
        "market_regime": detect_market_regime(close),
        "annualized_drift": mu,
        "annualized_drift_macro_adjusted": regime_adjustment["adjusted_mu"],
        "drift_damped_by_macro": regime_adjustment["damped"],
        "annualized_volatility": sigma,
    }

    ml_engineer_output = {
        "probability_up_next_day": walk_forward_predict_probability(
            price_history, macro_score_history, current_macro_score=macro_score
        ),
    }

    economist_output = {
        "macro_score": macro_score,
        # A score of 0.0 is ambiguous on its own: it's both "genuinely
        # neutral macro" and "no FRED reading available at all". The Trader
        # needs to tell those apart -- unknown macro must not be treated as
        # a reason to downgrade conviction the way lukewarm macro is.
        "macro_available": macro_snapshot.get("t10y2y") is not None or macro_snapshot.get("vixcls") is not None,
        "t10y2y": macro_snapshot.get("t10y2y"),
        "vixcls": macro_snapshot.get("vixcls"),
        "sentiment_context": team_c_output["sentiment"],
    }

    return {
        "symbol": team_c_output["symbol"],
        "mathematician": mathematician_output,
        "ml_engineer": ml_engineer_output,
        "economist": economist_output,
        "team_c_statistics": team_c_output["statistics"],
        "team_c_forecast": team_c_output["forecast"],
    }
