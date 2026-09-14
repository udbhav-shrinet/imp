# Hierarchical Multi-Agent Trading System

A CrewAI/LangChain-based paper-trading pipeline with a strict four-team
hierarchy: **Team C (Data/Signals) → Team B (Quants) → Team A
(Strategy/Risk) → Team S (Execution)**. Lower teams never execute
trades; only Team S calls the Alpaca API, and only on a thesis Team A's
Risker has approved.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and fill in your paper trading / API keys:

```bash
cp config/.env.example config/.env
```

`config/.env` is git-ignored. Never commit real credentials — all agents
load keys dynamically via `os.environ`.

## Run

```bash
# Dry run: analysis only, no trades placed
python main.py --symbol AAPL --capital 10000

# Execute the approved thesis as a paper trade
python main.py --symbol AAPL --capital 10000 --execute
```

## Project Structure

```
agents/
  team_c_data.py        # Data Puller, Statistician, Forecaster, Sentiment Analyst
  team_b_quants.py       # Mathematician, ML Engineer, Economist
  team_a_strategy.py     # Trader, Risker (VaR/Kelly, veto power)
  team_s_execution.py    # Execution agent (Alpaca paper trading only)
tools/
  alpaca_tools.py        # Alpaca REST client, order submission, TWAP slicing
  reddit_news_scraper.py # Yahoo Finance, PRAW (Reddit), NewsAPI scraping
config/
  .env.example           # Template for required environment variables
knowledge_base/          # Reference PDFs/books for RAG (not bundled)
main.py                  # Pipeline entry point
```

## Safety

- All Alpaca calls target `https://paper-api.alpaca.markets` by default;
  `tools/alpaca_tools.py` refuses to run against any URL that doesn't
  contain `paper-api`.
- The Risker agent (Team A) can veto any trade whose 1-day 95% VaR
  breaches the configured limit — Team S never sees an unapproved thesis.
- Position sizing uses half-Kelly, capped at 25% of allocated capital,
  as an additional safety margin beyond the raw Kelly Criterion.
