# Team A — Strategy & Risk: Formulas & Code Reference

Same note as the other teams' files re: Chan's and Taleb's books being
commercial/copyrighted — formulas and code below instead.

Implemented in `agents/team_a_strategy.py`.

## Trader

Synthesizes a directional thesis from Team B's `probability_up_next_day`,
market regime, Ornstein-Uhlenbeck reversion speed, and the Economist's
`macro_score` (`form_trade_thesis()`). A long candidate is gated by
macro_score: `> 0` -> full-conviction long, `(-0.3, 0]` -> long with
conviction halved, `<= -0.3` -> vetoed to `no_trade`. A short candidate
isn't gated the same way — a hostile macro regime is generally
*supportive* of a short thesis, not a reason to downgrade it.

The statistical-arbitrage / pairs-trading / cointegration theories named
in the spec (Engle-Granger test) are the natural extension if you trade
a *pair* of correlated symbols instead of one outright directional bet —
not yet wired in since the current pipeline is single-symbol.

Engle-Granger test sketch, for reference if you add pairs trading:
```
1. Regress: Y_t = alpha + beta * X_t + residual_t
2. Test residual_t for stationarity (Augmented Dickey-Fuller test)
3. If residuals are stationary -> X and Y are cointegrated
   -> trade the spread (Y - beta*X) reverting to its mean
```

## Risker

**Kelly Criterion** — optimal fraction of capital to risk given win
probability `p` and win/loss ratio `b` (average win size / average loss
size):
```
f* = p - (1 - p) / b
```
This repo uses **half-Kelly** (`f* * 0.5`) as a safety margin, and caps
the result at `MAX_POSITION_FRACTION` (25% of allocated capital).
`kelly_fraction()`

`b` is **dynamic**, not a static assumption: `compute_historical_win_loss_ratio()`
derives it from the symbol's own 2-year daily return history —
`mean(positive returns) / -mean(negative returns)` — a cheap, realized
proxy for "average win / average loss" that avoids re-running the full
walk-forward ML/GBM/OU pipeline once per historical trading day (which
would multiply this run's compute cost by roughly the number of trading
days in the window, per symbol, and doesn't fit GitHub Actions' free-tier
minutes across a ~76-symbol pool). Falls back to `1.5` if a symbol's
history has no up days or no down days to measure from.

**Historical Value at Risk (VaR)** — the loss threshold not expected to
be exceeded more than `(1 - confidence)` of the time, estimated directly
from the empirical return distribution (no distributional assumption):
```
VaR_95 = -percentile(historical_returns, 5)
```
`historical_var()`

**Conditional VaR / Expected Shortfall (CVaR)** — the *average* loss in
the tail beyond VaR (captures fat-tail severity that VaR alone misses):
```
CVaR_95 = -mean(returns_below_negative_VaR_95_threshold)
```
`historical_cvar()`

**GARCH(1,1)** — forward-looking volatility estimate that weights recent
shocks more heavily than a flat historical standard deviation:
```
sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
```
`estimate_garch_volatility()` (a simplified fixed-parameter version; for
production use, fit omega/alpha/beta via MLE with the `arch` package
instead of the fixed defaults used here).

**The veto rule**: `apply_risk_management()` rejects any thesis whose
1-day 95% VaR exceeds `VAR_LIMIT` (5% of position value) — Team S never
sees an unapproved report, full stop.
