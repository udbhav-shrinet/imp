# Hierarchical Multi-Agent Trading System

A CrewAI/LangChain paper-trading pipeline with a strict four-team
hierarchy. Every run scans a watchlist of symbols; each symbol is passed
down the chain independently:

```
Team C (Data & Signals) → Team B (Quants) → Team A (Strategy & Risk) → Team S (Execution)
     4 agents                3 agents              2 agents                 1 agent
```

Lower teams **cannot** place trades. `execute_team_a_report()` in
`agents/team_s_execution.py` is the only function in the codebase that
calls Alpaca's `submit_order`, and it refuses to act unless Team A's
Risker set `approved: true`. That makes the hierarchy a code-level
guarantee, not just a design intention.

---

## Table of contents

- [Setup](#setup)
- [How to trigger it](#how-to-trigger-it)
- [When it triggers](#when-it-triggers)
- [What each team and agent does](#what-each-team-and-agent-does)
- [Dashboard](#dashboard)
- [Project structure](#project-structure)
- [Safety rails](#safety-rails)
- [Known limitations](#known-limitations)

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon   # sentiment lexicon, one-time
```

Copy the environment template and fill in your keys:

```bash
cp config/.env.example config/.env
```

`config/.env` is git-ignored. Never commit real credentials — every
agent reads keys via `os.environ`, nothing is hardcoded.

| Variable | Needed for | Where to get it |
|---|---|---|
| `APCA_API_KEY_ID` | Alpaca paper trading | Alpaca dashboard → API keys |
| `APCA_API_SECRET_KEY` | Alpaca paper trading | Shown **once** when you generate the key pair — if you lost it, regenerate |
| `APCA_API_BASE_URL` | Alpaca endpoint | Leave as `https://paper-api.alpaca.markets` |
| `REDDIT_CLIENT_ID` | Team C sentiment | reddit.com/prefs/apps → create a **script** app |
| `REDDIT_CLIENT_SECRET` | Team C sentiment | same app |
| `REDDIT_USER_AGENT` | Reddit API etiquette | any descriptive string |

News needs no key — it comes from public RSS feeds (Yahoo Finance,
Google News).

---

## How to trigger it

### 1. Locally, from the CLI

```bash
# Dry run — full analysis, nothing submitted to the broker
python main.py

# Your own watchlist
python main.py --symbols AAPL,MSFT,TSLA

# Actually submit paper trades for every approved thesis
python main.py --execute

# Raise the per-position ceiling (see the sizing model below)
python main.py --max-position-usd 50 --execute
```

| Flag | Default | Meaning |
|---|---|---|
| `--symbols` | the ~75-symbol `DEFAULT_WATCHLIST` in `main.py` | Comma-separated tickers to scan |
| `--max-position-usd` | `20` | Hard dollar ceiling for a single position, regardless of equity |
| `--max-position-pct` | `0.10` | Ceiling as a fraction of current equity — whichever cap is smaller wins |
| `--execute` | off (dry run) | Without it, Team S is skipped entirely |

### 2. Manually, from GitHub Actions

Actions tab → **Trading Pipeline** → **Run workflow**. Inputs:
`symbols` (blank = the default watchlist), `max_position_usd`, `execute`
(defaults to `true`).

### Position sizing — built for a small account

There's no `--capital` flag anymore. The old design took a flat dollar
figure disconnected from the account (`$10,000` per thesis by default)
and let Kelly/VaR size within it — fine on a well-funded account, but on
a small one (this was built against a **$200 paper account**) that
figure has nothing to do with what's actually available, and a single
approved thesis could try to spend most of the account.

Now the cap is derived from live equity every run:

```
position_cap = min(max_position_usd, current_equity × max_position_pct)
```

On $200 with the defaults, that's `min($20, $20) = $20` — small enough
that several approved theses in the same batch still fit inside the
account. Two more guards sit underneath it:

- **Team A's own half-Kelly, capped at 25% of that figure** (unchanged —
  see below), so a single thesis rarely even reaches the $20 ceiling.
- **Team S re-checks live buying power immediately before submitting**
  (`agents/team_s_execution.py`), since Team A sizes each symbol
  independently and several approved theses in one batch can otherwise
  collectively exceed what's actually left to spend. It clamps down to
  whatever's really available and skips cleanly if that's under one
  share, rather than letting the broker reject the order.

One consequence worth knowing: with a $20 ceiling, a stock trading above
~$20/share can't buy even one whole share (fractional orders are
avoided — see Team S below) and will always skip with "buys less than
one whole share." On a small account, scanning ~75 symbols is partly
a way of finding the handful that are actually affordable, not an
expectation that most of them will trade. Raise `--max-position-usd`
(or your account's equity) if you want the pricier names in the
watchlist to be reachable.

### 3. Automatically, on a schedule

No action needed — see below.

---

## When it triggers

| Trigger | Cadence | Executes trades? |
|---|---|---|
| `schedule` (cron `0 */4 * * 1-5`) | Every 4 hours, Mon–Fri (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC) | **Yes** |
| `workflow_dispatch` (manual) | Whenever you click Run workflow | Yes, unless you set `execute: false` |
| Local CLI | Whenever you run it | Only with `--execute` |

Scheduled runs carry no dispatch inputs, so the workflow treats
"`execute` is not the string `false`" as execute-by-default — a cron run
trades. A manual run is the only way to opt into a dry run.

Two caveats worth knowing:

- The cron fires on a **fixed UTC clock, not market hours**. Some runs
  land outside US trading hours; orders are submitted `time_in_force:
  day` and simply queue for the next session.
- GitHub's scheduled workflows are best-effort and can be delayed during
  high load, and GitHub disables schedules on repos with no activity for
  60 days.

After every run the workflow commits the updated `logs/history.json`
back to `main`, which is what feeds the dashboard.

---

## What each team and agent does

Data flows strictly downward. Each team receives only the team above's
output packet.

### Team C — Data & Signals (4 agents)

Input: a ticker symbol. Output: raw prices + computed statistics +
forecasts + sentiment. Code: `agents/team_c_data.py`.

**1. Data Puller**
Pulls 6 months of daily OHLCV bars from Yahoo Finance (`yfinance`).
Everything downstream works off this one DataFrame.

**2. Statistician** — turns prices into normalized signals:

| Signal | Window | What it says |
|---|---|---|
| SMA | 20 / 50 / 200 | Trend direction across horizons |
| RSI | 14 | Overbought (>70) / oversold (<30) momentum |
| Bollinger Bands | 20, ±2σ | Volatility envelope around the mean |
| Z-score | 20 | How many σ price sits from its mean — the mean-reversion signal |
| ATR | 14 | Average true range; raw volatility for stop sizing |

**3. Forecaster** — projects 5 days ahead, two ways:
- **ARIMA(1,1,1)** via statsmodels — one autoregressive lag, one
  difference, one moving-average lag.
- **Holt's exponential smoothing** (additive trend).

**4. Sentiment Analyst** — scrapes and scores text:
- **Reddit** via PRAW — searches r/wallstreetbets, r/stocks, r/investing
  for the symbol, last week, up to 50 posts each.
- **News** via RSS — Yahoo Finance + Google News feeds, no API key.
- **VADER** compound sentiment (−1 to +1), averaged across all text.
- **TF-IDF** to extract the top distinctive keywords.

### Team B — Quants (3 agents)

Input: Team C's packet. Output: regime classification + probabilistic
forecast + macro context. Code: `agents/team_b_quants.py`.

**1. Mathematician** — stochastic calculus:
- **Geometric Brownian Motion** — simulates 200 forward price paths via
  Itô's Lemma applied to log(S): `d(logS) = (μ − σ²/2)dt + σ√dt·Z`.
- **Ornstein-Uhlenbeck fit** — OLS-regresses `X_t` on `X_{t-1}` to
  estimate mean-reversion speed `θ`, long-run mean `μ`, and `σ`. High
  `θ` = price snaps back to its mean quickly.
- **Regime detection** — classifies bull / bear / sideways by comparing
  the 20-day mean return against 0.5× its standard deviation.

**2. ML Engineer** — probability of an up move:
- **Features**: 1-day return, 5-day return, 10-day rolling volatility,
  10-day momentum.
- **Model**: `RandomForestClassifier` (200 trees, max depth 5).
- **Validation**: strict **walk-forward** — trains only on the 120 days
  immediately *preceding* the prediction point. This is the whole point:
  ordinary K-fold cross-validation leaks future data into the training
  set and makes financial models look far better than they are.
- **Output**: `probability_up_next_day`, the single number Team A leans
  on hardest.

**3. Economist** — reviews macro/microstructure context (rates, sector
rotation, liquidity, spreads) and passes sentiment context forward.
**This is the one agent that is still qualitative** — it emits a
checklist note, not a computed signal. Wiring in a real data source
(FRED yield curves, a spread feed) is the obvious next upgrade.

### Team A — Strategy & Risk (2 agents)

Input: Team B's packet. Output: an approved-or-vetoed, position-sized
report. Code: `agents/team_a_strategy.py`.

**1. Trader** — forms the directional thesis:

| Condition | Thesis |
|---|---|
| P(up) > 0.55 **and** regime is bull or sideways | `long` |
| P(up) < 0.45 **and** regime is bear or sideways | `short` |
| anything else | `no_trade` |

Confidence is `|P(up) − 0.5| × 2`, scaled 0–1.

**2. Risker** — sizes it, and holds **veto power**:
- **Kelly Criterion**: `f* = p − (1−p)/b`, with `b` (win/loss ratio)
  assumed 1.5. The result is **halved** (half-Kelly) and then **capped
  at 25%** of allocated capital — two independent safety margins on top
  of raw Kelly.
- **Historical VaR(95)**: the empirical 5th-percentile loss.
- **CVaR(95)**: average loss *beyond* VaR — the fat-tail measure.
- **GARCH(1,1)**: forward-looking volatility,
  `σ²_t = ω + α·r²_{t−1} + β·σ²_{t−1}`.
- **The veto**: if VaR(95) exceeds **5%**, the trade is rejected outright
  with a reason, position size forced to 0, and Team S never sees an
  actionable report.

### Team S — Execution (1 agent)

Input: Team A's report. Output: submitted paper orders. Code:
`agents/team_s_execution.py`, `tools/alpaca_tools.py`.

- Refuses to act unless `approved: true`.
- Converts dollars to **whole shares** (`position_size_usd // price`).
  Fractional shares are deliberately avoided: Alpaca rejects fractional
  short sales outright, and fractional orders carry other API
  restrictions.
- Skips with a clear reason if the allocation can't buy one full share.
- **TWAP-slices** the order into 4 equal whole-share child orders to
  simulate real-world slippage avoidance.
- Submits market orders, `time_in_force: day`, to the paper endpoint.

---

## Dashboard

`index.html` is a static page reading `logs/history.json` same-origin.
Enable it once: Settings → Pages → Source: **Deploy from a branch** →
`main` / `(root)`, then visit `https://<you>.github.io/<repo>/`.
It self-refreshes every 60 seconds.

| Tab | Shows |
|---|---|
| **Main** | Equity curve (hover for values), KPI tiles, trade log of actual buys/sells, all-scans table |
| **Team C** | Close, RSI, SMA, Z-score, ATR, VADER sentiment, Reddit/news counts, top keywords |
| **Team B** | Regime, annualized drift & volatility, OU θ and μ, P(up) |
| **Team A** | Thesis, confidence, approved/vetoed + reason, Kelly fraction, VaR, CVaR, size |
| **Team S** | Execution status, side, qty, reference price, child-order count, skip reason |

A published Claude Artifact can't serve this — its CSP blocks fetching
external JSON — which is why the dashboard lives in the repo and is
served by Pages.

---

## Project structure

```
agents/
  team_c_data.py         # Data Puller, Statistician, Forecaster, Sentiment Analyst
  team_b_quants.py       # Mathematician, ML Engineer, Economist
  team_a_strategy.py     # Trader, Risker (Kelly/VaR, veto power)
  team_s_execution.py    # Execution agent — the ONLY caller of submit_order
tools/
  alpaca_tools.py        # Alpaca REST client, orders, TWAP slicing
  reddit_news_scraper.py # Yahoo Finance, PRAW (Reddit), RSS news
  run_logger.py          # Appends each run's summary to logs/history.json
config/.env.example      # Template for required environment variables
knowledge_base/*.md      # Per-team formulas + pointers to implementing code
logs/history.json        # Run history (written by the workflow, read by the dashboard)
index.html               # Tabbed GitHub Pages dashboard
main.py                  # Entry point — watchlist loop
```

---

## Safety rails

- **Paper only.** `get_alpaca_client()` raises unless the base URL
  contains `paper-api`, so a live-trading URL can't be used by accident.
- **Risker veto.** VaR(95) above 5% blocks the trade; Team S can't
  override it.
- **Double-margined sizing.** Half-Kelly, then capped at 25% of
  allocated capital.
- **Single execution chokepoint.** One function submits orders; every
  other agent is structurally incapable of trading.
- **No hardcoded keys.** All credentials come from `os.environ`, and are
  `.strip()`ed to survive stray whitespace pasted into CI secret fields.

---

## Known limitations

Worth being clear about, since this trades real (paper) money:

- **No profit guarantee.** No trading system can promise one, this
  included. The rails bound risk; they don't create edge.
- **`win_loss_ratio` is assumed at 1.5**, not measured. Kelly sizing is
  only as good as that input — it should be derived from a real trade
  history.
- **GARCH parameters are fixed** (ω, α, β hardcoded) rather than fit by
  MLE. Use the `arch` package for a real fit.
- **Regime detection is a threshold rule**, not a true Hidden Markov
  Model.
- **The Economist agent is qualitative** — no live macro data feed.
- **No position/exit management.** The system opens positions; nothing
  closes them, sets stops, or prevents stacking duplicate positions in
  the same symbol across runs.
- **The watchlist is ~75 symbols**, not "the whole market" — a full
  ARIMA/GBM/Random-Forest pass per symbol doesn't scale to thousands,
  and each additional symbol also means more Reddit API calls in the
  same run, which risks rate-limiting on a large watchlist.
- **Whole-share sizing means a low dollar ceiling can't reach
  higher-priced stocks.** A $20 cap simply can't buy one share of a
  $650 stock; that scan will always skip, by design.
