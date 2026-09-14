"""
Team C: Data & Signals.

Four agents that gather raw market data and compute statistical /
sentiment signals. Their combined output is handed up to Team B (Quants).
No agent in this team is permitted to size or place trades.

Theories applied:
  - Data Puller: raw OHLCV retrieval (Yahoo Finance).
  - Statistician: moving averages, RSI, standard deviation / Bollinger
    Bands, Average True Range (ATR), Z-scores for mean reversion.
  - Forecaster: ARIMA / exponential smoothing time-series projections.
  - Sentiment Analyst: VADER scoring + TF-IDF over Reddit/News text.
"""

import numpy as np
import pandas as pd
from crewai import Agent
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tools.reddit_news_scraper import (
    get_yahoo_price_history,
    scrape_news_headlines,
    scrape_reddit_mentions,
)


# --------------------------------------------------------------------------
# Statistical helpers
# --------------------------------------------------------------------------

def compute_moving_averages(prices: pd.Series, windows=(20, 50, 200)) -> dict:
    return {f"sma_{w}": prices.rolling(w).mean().iloc[-1] for w in windows if len(prices) >= w}


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> dict:
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return {
        "mid_band": float(mid.iloc[-1]),
        "upper_band": float((mid + num_std * std).iloc[-1]),
        "lower_band": float((mid - num_std * std).iloc[-1]),
        "std_dev": float(std.iloc[-1]),
    }


def compute_z_score(prices: pd.Series, window: int = 20) -> float:
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    z = (prices - mean) / std.replace(0, np.nan)
    return float(z.iloc[-1])


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(true_range.rolling(period).mean().iloc[-1])


def forecast_arima(prices: pd.Series, steps: int = 5) -> list:
    model = ARIMA(prices, order=(1, 1, 1)).fit()
    return model.forecast(steps=steps).tolist()


def forecast_exponential_smoothing(prices: pd.Series, steps: int = 5) -> list:
    model = ExponentialSmoothing(prices, trend="add", seasonal=None).fit()
    return model.forecast(steps).tolist()


def score_sentiment_vader(texts: list[str]) -> dict:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)["compound"] for t in texts if t]
    return {
        "mean_compound": float(np.mean(scores)) if scores else 0.0,
        "n_samples": len(scores),
    }


def score_tfidf_keywords(texts: list[str], top_n: int = 10) -> list[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [t for t in texts if t]
    if not texts:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    vectorizer.fit_transform(texts)
    return list(vectorizer.get_feature_names_out())


# --------------------------------------------------------------------------
# CrewAI Agents
# --------------------------------------------------------------------------

data_puller_agent = Agent(
    role="Data Puller",
    goal="Retrieve clean, complete OHLCV price history for the target symbol from Yahoo Finance.",
    backstory=(
        "A meticulous market-data engineer who ensures every downstream agent works from "
        "reliable, well-formed pandas DataFrames rather than raw, messy feeds."
    ),
    allow_delegation=False,
    verbose=True,
)

statistician_agent = Agent(
    role="Statistician",
    goal=(
        "Compute moving averages, RSI, Bollinger Bands / standard deviation, ATR, and "
        "Z-scores for mean reversion on the pulled price data."
    ),
    backstory=(
        "A quantitative analyst grounded in classical technical statistics, translating "
        "raw price series into normalized signals for the Quants team."
    ),
    allow_delegation=False,
    verbose=True,
)

forecaster_agent = Agent(
    role="Forecaster",
    goal="Project near-term price paths using ARIMA and exponential smoothing.",
    backstory=(
        "A time-series specialist who distrusts naive extrapolation and always checks "
        "stationarity before fitting a model."
    ),
    allow_delegation=False,
    verbose=True,
)

sentiment_analyst_agent = Agent(
    role="Sentiment Analyst",
    goal=(
        "Scrape Reddit and news sources for the target symbol and score aggregate "
        "sentiment using VADER and TF-IDF keyword extraction."
    ),
    backstory=(
        "A text-mining specialist who treats retail chatter and news coverage as a "
        "noisy but informative signal, never as ground truth on its own."
    ),
    allow_delegation=False,
    verbose=True,
)


def gather_team_c_signals(symbol: str) -> dict:
    """Orchestrate Team C's four agents' underlying logic into one signal packet."""
    price_history = get_yahoo_price_history(symbol)
    close = price_history["Close"]

    statistics = {
        **compute_moving_averages(close),
        "rsi_14": compute_rsi(close),
        **compute_bollinger_bands(close),
        "z_score_20": compute_z_score(close),
        "atr_14": compute_atr(price_history),
    }

    forecast = {
        "arima_forecast": forecast_arima(close),
        "exp_smoothing_forecast": forecast_exponential_smoothing(close),
    }

    reddit_posts = scrape_reddit_mentions(symbol)
    news_articles = scrape_news_headlines(symbol)
    texts = [p["title"] + " " + p.get("selftext", "") for p in reddit_posts] + [
        (a["title"] or "") + " " + (a["description"] or "") for a in news_articles
    ]
    sentiment = {
        **score_sentiment_vader(texts),
        "top_keywords": score_tfidf_keywords(texts),
        "reddit_post_count": len(reddit_posts),
        "news_article_count": len(news_articles),
    }

    return {
        "symbol": symbol,
        "price_history": price_history,
        "statistics": statistics,
        "forecast": forecast,
        "sentiment": sentiment,
    }
