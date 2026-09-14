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

Every run scans a **watchlist**, not one symbol -- each ticker goes
through the full pipeline independently and gets logged whether or not
Team A approves it.

```bash
# Dry run: analysis only, no trades placed, default watchlist
python main.py --capital 10000

# Your own watchlist
python main.py --symbols AAPL,MSFT,TSLA --capital 10000

# Execute every approved thesis as a paper trade
python main.py --capital 10000 --execute
```

The default watchlist (`main.py`'s `DEFAULT_WATCHLIST`) is ten liquid
large-caps spanning tech, finance, energy, and healthcare -- a stand-in
for "the market," since actually scanning every listed symbol with a
full ARIMA/GBM/Random-Forest pass per name isn't practical. Extend it
with `--symbols`.

## Running on a schedule (GitHub Actions)

`.github/workflows/trading-pipeline.yml` runs the pipeline automatically
every 4 hours on weekdays, and can also be triggered manually from the
Actions tab (`Run workflow`, with `symbols`, `capital`, and `execute`
inputs).

Add these as **repository secrets** (Settings → Secrets and variables →
Actions) before enabling it:

- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`

The workflow always targets the paper trading endpoint and executes by
default (a scheduled run has no dispatch inputs to opt out with, so
`execute` only turns off if a manual run explicitly sets it to `false`).
It commits the updated `logs/history.json` back to `main` after every
run.

## Dashboard (GitHub Pages)

`index.html` is a static dashboard reading `logs/history.json`
same-origin -- enable it once under Settings → Pages → Source: Deploy
from a branch → `main` / `(root)`, then visit
`https://<you>.github.io/<repo>/`. A published Claude Artifact can't
pull this off (its CSP blocks fetching external JSON), which is why this
lives as a plain page in the repo instead.

## Project Structure

```
agents/
  team_c_data.py        # Data Puller, Statistician, Forecaster, Sentiment Analyst
  team_b_quants.py       # Mathematician, ML Engineer, Economist
  team_a_strategy.py     # Trader, Risker (VaR/Kelly, veto power)
  team_s_execution.py    # Execution agent (Alpaca paper trading only)
tools/
  alpaca_tools.py        # Alpaca REST client, order submission, TWAP slicing
  reddit_news_scraper.py # Yahoo Finance, PRAW (Reddit), RSS news scraping
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
