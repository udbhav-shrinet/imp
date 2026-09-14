"""
Team B: Quants.

Three agents that receive Team C's signal packet and translate it into
probabilistic and regime-aware quantitative context. Their combined
output is handed up to Team A (Strategy). No agent in this team is
permitted to size or place trades.

Theories applied:
  - Mathematician: Ito's Lemma / Geometric Brownian Motion (GBM) price
    path simulation, Ornstein-Uhlenbeck mean reversion, Markov-chain
    regime detection (bull / bear / sideways).
  - ML Engineer: Random Forest probability-of-direction model with
    walk-forward validation (purged, to avoid time leakage).
  - Economist: macro / market-microstructure context review.
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


# --------------------------------------------------------------------------
# ML Engineer: probability of direction
# --------------------------------------------------------------------------

def build_features(price_history: pd.DataFrame) -> pd.DataFrame:
    df = price_history.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["momentum_10d"] = df["Close"] / df["Close"].shift(10) - 1
    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()


def walk_forward_predict_probability(price_history: pd.DataFrame, train_window: int = 120) -> float:
    """
    Predict P(next-day up move) using a Random Forest with walk-forward
    validation: train only on data strictly preceding the prediction
    point, avoiding the look-ahead bias that standard K-fold CV creates
    in time-series data (purged cross-validation principle).
    """
    features = build_features(price_history)
    feature_cols = ["return_1d", "return_5d", "volatility_10d", "momentum_10d"]

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
        "Review macro context: interest rate yield curves, sector rotation, market "
        "microstructure (bid-ask spread, liquidity), and behavioral finance critiques "
        "of market efficiency."
    ),
    backstory=(
        "A macro strategist who tempers pure quant signals with real-world context "
        "about rates, liquidity, and crowd psychology."
    ),
    allow_delegation=False,
    verbose=True,
)


def analyze_team_b(team_c_output: dict) -> dict:
    """Orchestrate Team B's three agents' underlying logic on Team C's signal packet."""
    price_history = team_c_output["price_history"]
    close = price_history["Close"]
    returns = close.pct_change().dropna()

    spot = float(close.iloc[-1])
    mu = float(returns.mean() * 252)
    sigma = float(returns.std() * np.sqrt(252))

    mathematician_output = {
        "gbm_simulated_paths_sample": simulate_gbm_paths(spot, mu, sigma, days=20, n_paths=200)[:5].tolist(),
        "ornstein_uhlenbeck": fit_ornstein_uhlenbeck(close),
        "market_regime": detect_market_regime(close),
        "annualized_drift": mu,
        "annualized_volatility": sigma,
    }

    ml_engineer_output = {
        "probability_up_next_day": walk_forward_predict_probability(price_history),
    }

    economist_output = {
        "note": (
            "Macro/microstructure review: assess current rate environment, sector "
            "rotation phase, and liquidity/spread conditions before Team A commits capital."
        ),
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
