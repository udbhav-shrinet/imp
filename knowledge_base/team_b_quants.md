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

A quantitative macro regime score, `macro_score`, computed from two free
FRED series and normalized to [-1, 1]:

```
T10Y2Y (10Y-2Y yield spread): <0 -> -1 (inverted/recession risk)
                               >0.5 -> +1 (steepening/expansion)
                               otherwise -> linear interpolation
VIXCLS (VIX close):           >25 -> -1 (risk-off)
                               <18 -> +1 (risk-on)
                               otherwise -> linear interpolation
macro_score = clip(0.5 * yield_component + 0.5 * vix_component, -1, 1)
```

`compute_macro_score()` in `agents/team_b_quants.py`. Fetched once per
pipeline run (not once per symbol) via `tools/fred_tools.py`, with a
neutral-0 fallback per component if a reading is unavailable. Feeds both
the ML Engineer (as a per-date-aligned training feature,
`build_macro_score_history()` / `align_macro_score()`) and the Trader's
long-thesis gate (see `team_a_strategy_risk.md`).
