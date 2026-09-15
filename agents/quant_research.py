"""Quant Research: factor scoring, regime classification and ML probability."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression


def macro_score(t10y2y: float | None, vix: float | None) -> float:
    """Score macro risk on [-1, 1]; positive is risk-on.

    Yield component: clip(T10Y2Y / 0.50, -1, 1).
    Volatility component: 1 below VIX 18, -1 above VIX 25, linear between.
    Final score is the equal-weight average of available components.
    """
    parts = []
    if t10y2y is not None:
        parts.append(float(np.clip(t10y2y / 0.50, -1, 1)))
    if vix is not None:
        parts.append(float(np.clip(1 - 2 * (vix - 18) / 7, -1, 1)))
    return float(np.mean(parts)) if parts else 0.0


def market_regime(features: dict) -> str:
    trend = features["return_20d"]
    volatility = features["volatility_20d"]
    if trend > max(0.03, volatility * 0.15):
        return "bull"
    if trend < -max(0.03, volatility * 0.15):
        return "bear"
    return "sideways"


def triple_barrier_labels(
    close: pd.Series,
    horizon: int = 5,
    profit_multiple: float = 1.5,
    stop_multiple: float = 1.0,
) -> pd.Series:
    """Label each entry using profit-taking, stop-loss and time barriers.

    For entry i, the upper/lower barriers are +/- volatility_i multiples.
    The first barrier touched in the next ``horizon`` bars wins; otherwise
    the sign of the final return is used. The last horizon rows are unknown
    and excluded from training.
    """
    returns = close.pct_change()
    volatility = returns.rolling(20).std()
    labels = pd.Series(index=close.index, dtype="float64")
    for i in range(len(close) - horizon):
        sigma = volatility.iloc[i]
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        entry = float(close.iloc[i])
        upper = entry * (1 + profit_multiple * sigma)
        lower = entry * (1 - stop_multiple * sigma)
        future = close.iloc[i + 1 : i + horizon + 1]
        up_hits = np.flatnonzero(future.to_numpy() >= upper)
        down_hits = np.flatnonzero(future.to_numpy() <= lower)
        if len(up_hits) and (not len(down_hits) or up_hits[0] < down_hits[0]):
            labels.iloc[i] = 1
        elif len(down_hits):
            labels.iloc[i] = -1
        else:
            labels.iloc[i] = np.sign(float(future.iloc[-1] / entry - 1))
    return labels


def _feature_frame(daily: pd.DataFrame, current_macro: float) -> tuple[pd.DataFrame, list[str]]:
    close = daily["Close"]
    frame = pd.DataFrame(index=daily.index)
    frame["r1"] = close.pct_change()
    frame["r5"] = close.pct_change(5)
    frame["vol"] = frame["r1"].rolling(20).std()
    frame["rsi"] = 100 - 100 / (
        1 + close.diff().clip(lower=0).rolling(14).mean()
        / (-close.diff().clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    )
    frame["mom20"] = close / close.shift(20) - 1
    # Historical macro observations are not available in this call. Do not
    # repeat today's macro value across old rows; that would contaminate the
    # walk-forward training distribution. Macro is fused after prediction.
    labels = triple_barrier_labels(close)
    frame["target"] = (labels > 0).astype(float)
    frame.loc[labels.isna(), "target"] = np.nan
    features = ["r1", "r5", "vol", "rsi", "mom20"]
    return frame, features


def purged_walk_forward_probability(
    daily: pd.DataFrame,
    current_macro: float,
    train_window: int = 252,
    validation_window: int = 20,
    embargo: int = 5,
) -> dict:
    """Estimate a calibrated probability without look-ahead leakage.

    Each validation block is preceded by a training block ending ``embargo``
    bars earlier. Predictions are out-of-sample and are used to fit isotonic
    calibration. This is a compact purged/embargoed walk-forward procedure,
    not a claim that calibration remains stable in a new regime.
    """
    frame, feature_cols = _feature_frame(daily, current_macro)
    usable = frame.dropna(subset=feature_cols + ["target"])
    latest_features = frame.dropna(subset=feature_cols).iloc[[-1]]
    if len(usable) < 100 or usable["target"].nunique() < 2:
        return {"probability": 0.5, "calibrated": False, "oos_brier": None, "oos_samples": 0}

    predictions, outcomes = [], []
    start = max(60, train_window)
    for validation_start in range(start, len(usable) - validation_window + 1, validation_window):
        train_end = validation_start - embargo
        train_start = max(0, train_end - train_window)
        train = usable.iloc[train_start:train_end]
        validation = usable.iloc[validation_start:validation_start + validation_window]
        if len(train) < 60 or train["target"].nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42
        )
        model.fit(train[feature_cols], train["target"])
        predictions.extend(model.predict_proba(validation[feature_cols])[:, 1])
        outcomes.extend(validation["target"].astype(int))

    final_train = usable.tail(min(train_window, len(usable)))
    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42
    )
    model.fit(final_train[feature_cols], final_train["target"])
    raw_probability = float(model.predict_proba(latest_features[feature_cols])[0][1])
    calibrated = False
    probability = raw_probability
    brier = None
    if len(predictions) >= 30 and len(set(outcomes)) > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(predictions, outcomes)
        probability = float(calibrator.predict([raw_probability])[0])
        brier = float(np.mean((np.asarray(predictions) - np.asarray(outcomes)) ** 2))
        calibrated = True
    return {
        "probability": float(np.clip(probability, 0.01, 0.99)),
        "raw_probability": raw_probability,
        "calibrated": calibrated,
        "oos_brier": brier,
        "oos_samples": len(predictions),
        "labeling": "triple_barrier",
        "embargo_bars": embargo,
    }


def research_symbol(intelligence: dict, macro: dict) -> dict:
    features = intelligence["daily_features"]
    macro_value = macro_score(macro.get("t10y2y"), macro.get("vixcls"))
    ml = purged_walk_forward_probability(intelligence["daily"], macro_value)
    probability = ml["probability"]
    trend_score = np.tanh(
        3 * features["return_20d"] + 2 * features["return_5d"]
    )
    intraday = intelligence["intraday_features"]
    intraday_score = np.tanh(4 * intraday["return_5d"] + 2 * intraday["return_1d"])
    candle_score = (
        int(features["candle"]["bullish_engulfing"])
        + int(features["candle"]["hammer"])
        + int(features["breakout_20d"])
        - int(features["candle"]["bearish_engulfing"])
        - int(features["candle"]["shooting_star"])
        - int(features["breakdown_20d"])
    ) / 3
    sentiment_score = intelligence["sentiment"]["score"] if intelligence["sentiment"]["available"] else 0.0
    recent_returns = intelligence["daily"]["Close"].pct_change().dropna().tail(60).tolist()
    average_dollar_volume = float(
        (intelligence["daily"]["Close"] * intelligence["daily"]["Volume"]).rolling(20).mean().iloc[-1]
    )
    estimated_cost = float(
        0.0015 + 0.0005 * np.sqrt(max(features["volatility_20d"], 0.01))
        + 0.0005 / max(np.sqrt(average_dollar_volume / 1_000_000), 1)
    )
    expected_edge = abs(probability - 0.5) * 2 * max(features["atr_14"] / features["last_price"], 0.01)
    combined = (
        0.40 * (probability * 2 - 1)
        + 0.20 * trend_score
        + 0.15 * np.clip(macro_value, -1, 1)
        + 0.10 * np.clip(sentiment_score, -1, 1)
        + 0.05 * np.clip(candle_score, -1, 1)
        + 0.10 * intraday_score
    )
    direction = "long" if combined >= 0.20 else "short" if combined <= -0.20 else "flat"
    return {
        "symbol": intelligence["symbol"],
        "direction": direction,
        "score": float(combined),
        "probability_up": probability,
        "ml_diagnostics": ml,
        "regime": market_regime(features),
        "macro_score": macro_value,
        "volatility": features["volatility_20d"],
        "atr": features["atr_14"],
        "price": features["last_price"],
        "sentiment": intelligence["sentiment"],
        "recent_returns": recent_returns,
        "average_dollar_volume": average_dollar_volume,
        "estimated_round_trip_cost": estimated_cost,
        "expected_edge": expected_edge,
        "technical": features,
        "reasons": {
            "ml": 0.40 * (probability * 2 - 1),
            "trend": 0.20 * trend_score,
            "macro": 0.15 * macro_value,
            "sentiment": 0.10 * sentiment_score,
            "candles": 0.05 * candle_score,
            "intraday": 0.10 * intraday_score,
        },
    }
