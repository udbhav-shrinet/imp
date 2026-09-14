"""
Data-gathering tools used by Team C (Data Puller, Sentiment Analyst):
Yahoo Finance price history, Reddit (PRAW), and news headlines.
"""

import os
from datetime import datetime, timedelta

import praw
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))


def get_reddit_client() -> praw.Reddit:
    """Build an authenticated read-only PRAW client from environment variables."""
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "trading-system-sentiment-bot/1.0"),
    )


def scrape_reddit_mentions(symbol: str, subreddits: list[str] | None = None, limit: int = 50) -> list[dict]:
    """
    Pull recent posts mentioning `symbol` from the given subreddits
    (defaults to wallstreetbets, stocks, investing) for sentiment scoring.
    """
    subreddits = subreddits or ["wallstreetbets", "stocks", "investing"]
    reddit = get_reddit_client()
    posts = []
    for sub_name in subreddits:
        subreddit = reddit.subreddit(sub_name)
        for submission in subreddit.search(symbol, sort="new", time_filter="week", limit=limit):
            posts.append(
                {
                    "subreddit": sub_name,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": submission.created_utc,
                    "url": submission.url,
                }
            )
    return posts


def scrape_news_headlines(symbol: str, days_back: int = 7, page_size: int = 50) -> list[dict]:
    """
    Pull recent news headlines mentioning `symbol` via NewsAPI.
    Requires NEWS_API_KEY in the environment.
    """
    api_key = os.environ["NEWS_API_KEY"]
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": symbol,
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "language": "en",
            "apiKey": api_key,
        },
        timeout=15,
    )
    response.raise_for_status()
    articles = response.json().get("articles", [])
    return [
        {
            "source": a["source"]["name"],
            "title": a["title"],
            "description": a["description"],
            "published_at": a["publishedAt"],
            "url": a["url"],
        }
        for a in articles
    ]


def get_yahoo_price_history(symbol: str, period: str = "6mo", interval: str = "1d"):
    """
    Pull OHLCV price history from Yahoo Finance as a pandas DataFrame.
    Uses the `yfinance` library, which wraps Yahoo's public chart API.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    return ticker.history(period=period, interval=interval)
