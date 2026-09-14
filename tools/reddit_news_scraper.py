"""
Data-gathering tools used by Team C (Data Puller, Sentiment Analyst):
Yahoo Finance price history, Reddit (PRAW), and news headlines via RSS.
"""

import os

import feedparser
import praw
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))


def get_reddit_client() -> praw.Reddit:
    """Build an authenticated read-only PRAW client from environment variables."""
    return praw.Reddit(
        # .strip() guards against stray whitespace/tab characters that can
        # sneak in when a secret is copy-pasted into a CI provider's UI --
        # Reddit's OAuth endpoint rejects a corrupted secret with a plain 401.
        client_id=os.environ["REDDIT_CLIENT_ID"].strip(),
        client_secret=os.environ["REDDIT_CLIENT_SECRET"].strip(),
        user_agent=os.environ.get("REDDIT_USER_AGENT", "trading-system-sentiment-bot/1.0").strip(),
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


DEFAULT_NEWS_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
    "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
]


def scrape_news_headlines(symbol: str, feed_templates: list[str] | None = None, limit: int = 50) -> list[dict]:
    """
    Pull recent news headlines mentioning `symbol` from RSS feeds (Yahoo
    Finance and Google News by default). No API key required.
    """
    feed_templates = feed_templates or DEFAULT_NEWS_RSS_FEEDS
    articles = []
    for template in feed_templates:
        feed_url = template.format(symbol=symbol)
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries[:limit]:
            articles.append(
                {
                    "source": parsed.feed.get("title", feed_url),
                    "title": entry.get("title", ""),
                    "description": entry.get("summary", ""),
                    "published_at": entry.get("published", ""),
                    "url": entry.get("link", ""),
                }
            )
    return articles


def get_yahoo_price_history(symbol: str, period: str = "6mo", interval: str = "1d"):
    """
    Pull OHLCV price history from Yahoo Finance as a pandas DataFrame.
    Uses the `yfinance` library, which wraps Yahoo's public chart API.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    return ticker.history(period=period, interval=interval)
