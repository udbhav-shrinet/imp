"""Market Intelligence: clean bars, candle structure, technical factors and sentiment.

This team only observes markets. It never sizes or submits orders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.reddit_news_scraper import (
    get_yahoo_price_history,
    scrape_news_headlines,
    scrape_reddit_mentions,
)


def _require_bars(frame: pd.DataFrame, minimum: int = 80) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    clean = frame.dropna(subset=sorted(required)).copy()
    clean = clean[(clean["Close"] > 0) & (clean["High"] >= clean["Low"])]
    if len(clean) < minimum:
        raise ValueError(f"Only {len(clean)} valid bars; need at least {minimum}")
    return clean


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    value = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return float(value.iloc[-1]) if pd.notna(value.iloc[-1]) else 50.0


def _atr(frame: pd.DataFrame, period: int = 14) -> float:
    previous = frame["Close"].shift()
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous).abs(),
            (frame["Low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.rolling(period).mean().iloc[-1])


def candle_patterns(frame: pd.DataFrame) -> dict:
    """Return interpretable candlestick flags from the latest two completed bars."""
    latest = frame.iloc[-1]
    prior = frame.iloc[-2]
    body = abs(float(latest["Close"] - latest["Open"]))
    candle_range = max(float(latest["High"] - latest["Low"]), 1e-9)
    upper_wick = float(latest["High"] - max(latest["Open"], latest["Close"]))
    lower_wick = float(min(latest["Open"], latest["Close"]) - latest["Low"])
    bullish_engulfing = (
        prior["Close"] < prior["Open"]
        and latest["Close"] > latest["Open"]
        and latest["Open"] <= prior["Close"]
        and latest["Close"] >= prior["Open"]
    )
    bearish_engulfing = (
        prior["Close"] > prior["Open"]
        and latest["Close"] < latest["Open"]
        and latest["Open"] >= prior["Close"]
        and latest["Close"] <= prior["Open"]
    )
    return {
        "doji": body <= candle_range * 0.10,
        "hammer": lower_wick >= body * 2 and upper_wick <= max(body, candle_range * 0.1),
        "shooting_star": upper_wick >= body * 2 and lower_wick <= max(body, candle_range * 0.1),
        "bullish_engulfing": bool(bullish_engulfing),
        "bearish_engulfing": bool(bearish_engulfing),
    }


def technical_snapshot(frame: pd.DataFrame) -> dict:
    close = frame["Close"]
    returns = close.pct_change()
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    volume_mean = frame["Volume"].rolling(20).mean()
    return {
        "last_price": float(close.iloc[-1]),
        "return_1d": float(returns.iloc[-1]),
        "return_5d": float(close.iloc[-1] / close.iloc[-6] - 1),
        "return_20d": float(close.iloc[-1] / close.iloc[-21] - 1),
        "volatility_20d": float(returns.rolling(20).std().iloc[-1] * np.sqrt(252)),
        "rsi_14": _rsi(close),
        "atr_14": _atr(frame),
        "ema_12": float(ema_fast.iloc[-1]),
        "ema_26": float(ema_slow.iloc[-1]),
        "macd_histogram": float((macd - signal).iloc[-1]),
        "bollinger_z": float(((close - rolling_mean) / rolling_std.replace(0, np.nan)).iloc[-1]),
        "volume_ratio": float(frame["Volume"].iloc[-1] / max(volume_mean.iloc[-1], 1)),
        "breakout_20d": bool(close.iloc[-1] >= close.rolling(20).max().iloc[-2]),
        "breakdown_20d": bool(close.iloc[-1] <= close.rolling(20).min().iloc[-2]),
        "candle": candle_patterns(frame),
    }


def _sentiment(symbol: str) -> dict:
    """Sentiment is optional evidence; unavailable feeds never become a trade signal."""
    try:
        reddit = scrape_reddit_mentions(symbol, limit=20)
    except Exception as exc:
        print(f"[Market Intelligence] Reddit unavailable for {symbol}: {exc}")
        reddit = []
    try:
        news = scrape_news_headlines(symbol, limit=20)
    except Exception as exc:
        print(f"[Market Intelligence] News unavailable for {symbol}: {exc}")
        news = []
    texts = [
        f"{item.get('title', '')} {item.get('selftext', '')}"
        for item in reddit
    ] + [
        f"{item.get('title', '')} {item.get('description', '')}"
        for item in news
    ]
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(text)["compound"] for text in texts if text.strip()]
    except LookupError:
        scores = []
    return {
        "score": float(np.mean(scores)) if scores else 0.0,
        "sample_count": len(scores),
        "reddit_count": len(reddit),
        "news_count": len(news),
        "available": bool(scores),
    }


def gather_market_intelligence(symbol: str) -> dict:
    daily = _require_bars(get_yahoo_price_history(symbol, period="2y", interval="1d"))
    try:
        intraday = _require_bars(
            get_yahoo_price_history(symbol, period="60d", interval="15m"), minimum=30
        )
    except Exception as exc:
        print(f"[Market Intelligence] Intraday data unavailable for {symbol}: {exc}")
        intraday = daily.tail(30)
    return {
        "symbol": symbol,
        "daily": daily,
        "intraday": intraday,
        "daily_features": technical_snapshot(daily),
        "intraday_features": technical_snapshot(intraday),
        "sentiment": _sentiment(symbol),
    }
