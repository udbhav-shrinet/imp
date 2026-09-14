# Team B — Quants: Formulas & Code Reference

Same note as Team C's file: the referenced books (Lopez de Prado, Shreve,
Harris) are commercial and copyrighted, so I haven't dropped PDFs here.
Below are the actual formulas/algorithms implemented, and where.

All implemented in `agents/team_b_quants.py`.

## Mathematician

**Ito's Lemma / Geometric Brownian Motion (GBM)**

The stock price SDE:
```
dS = mu * S * dt + sigma * S * dW
```
Applying Ito's Lemma to `log(S)` gives an exact discretization used for
simulation:
```
log(S_{t+dt}) - log(S_t) = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z,   Z ~ N(0,1)
```
`simulate_gbm_paths()` draws `n_paths` such trajectories over `days` steps.

**Ornstein-Uhlenbeck process (mean reversion)**
```
dX = theta * (mu - X) * dt + sigma * dW
```
Discretized and fit by OLS regression of `X_t` on `X_{t-1}`:
```
X_t = a + b * X_{t-1} + residual
theta = -ln(b) / dt
mu    = a / (1 - b)
sigma = std(residual) * sqrt(2*theta / (1 - b^2))
```
`fit_ornstein_uhlenbeck()` — `theta` is the speed of reversion (higher =
faster reversion to the mean `mu`).

**Regime detection (bull / bear / sideways)**

A lightweight Markov-chain-style classifier over the rolling mean/std of
recent returns (`detect_market_regime()`). A full Hidden Markov Model
(e.g. via `hmmlearn.GaussianHMM`) is a drop-in upgrade if you want
probabilistic regime membership instead of a hard classification.

## ML Engineer

**Feature set** (`build_features()`): 1-day return, 5-day return, 10-day
rolling volatility, 10-day momentum.

**Random Forest classifier** predicting P(next-day close > today's close).

**Walk-forward validation** (`walk_forward_predict_probability()`): the
model trains only on the `train_window` days strictly preceding the
prediction point — never on data from after the point being predicted.
This is the core fix for the "standard K-fold CV leaks the future into
the past" problem that plagues naive time-series ML:

```
train  = features[t - train_window : t]     # strictly in the past
predict = features[t]                        # today
```

For a fuller treatment (purged + embargoed CV, meta-labeling, fractional
differentiation), Lopez de Prado's *Advances in Financial Machine
Learning* is the standard reference — worth buying if you go deeper here.

## Economist

Currently a qualitative pass-through: surfaces Team C's sentiment context
alongside a reminder to check the macro backdrop (rate environment,
sector rotation phase, bid-ask spread / liquidity) before Team A commits
capital. This is the one agent left deliberately unquantified — plugging
in a real data source (FRED for yield curves, a liquidity/spread feed)
is the natural next step if you want it to contribute a hard signal
rather than a checklist.
