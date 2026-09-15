# Hierarchical 30-minute bulk trading design

This repository uses a deterministic hierarchy rather than letting a language
model invent orders:

1. **Market Intelligence Directorate**: validates OHLCV data, computes candle
   structure, technical factors, and optional news/Reddit sentiment.
2. **Quant Research Bureau**: estimates trend, volatility, macro regime and
   next-bar direction probability with a time-ordered Random Forest.
3. **Portfolio Council**: ranks all symbols together, removes held/pending
   symbols, limits count and total new exposure, then creates one batch plan.
4. **Execution Commander**: rechecks buying power and exposure, assigns unique
   client order IDs, and is the only module that can submit paper orders.
   It also executes the Position Manager's stop/opposite/flat-signal exits.

## Calculations and formulas

### Log return

`r_t = ln(C_t / C_(t-1))`

Used for volatility and model features because returns are more stable than
raw prices.

### Annualized volatility

`sigma_annual = stdev(r_1 ... r_n) * sqrt(252)`

This is a risk estimate, not a return forecast.

### RSI(14)

`RSI = 100 - 100 / (1 + average_gain / average_loss)`

The implementation uses rolling arithmetic means over 14 bars.

### ATR(14)

`TR_t = max(H_t-L_t, |H_t-C_(t-1)|, |L_t-C_(t-1)|)`

`ATR = rolling_mean(TR, 14)`

The council uses `max(2 * ATR, 1% of price)` as a planning stop distance.
That distance is recorded for later position-management work; it is not a
promise that a market order will fill at the stop.

### Bollinger z-score

`z_t = (C_t - SMA_20) / stdev_20`

It is evidence about extension, not a standalone entry rule.

### Macro score

Yield component:

`Y = clip(T10Y2Y / 0.50, -1, 1)`

VIX component:

`V = clip(1 - 2*(VIX - 18)/7, -1, 1)`

`macro_score = mean(available(Y, V))`

The score is neutral when both inputs are unavailable.

### Model and portfolio score

The Random Forest predicts `P(next close > current close)` using recent
returns, volatility, RSI and momentum. The live macro score is fused after the
prediction; it is not repeated across historical rows because doing so would
contaminate walk-forward training without point-in-time macro history. Training labels
use a triple barrier: a 5-bar vertical barrier, a profit barrier at
`+1.5 * rolling_volatility`, and a stop barrier at
`-1.0 * rolling_volatility`. The first barrier touched determines the label.
The model is trained with purged walk-forward blocks and a 5-bar embargo:
validation data is never allowed to overlap the training horizon. Out-of-
sample probabilities are calibrated with isotonic regression when at least 30
validation predictions are available. The record stores the Brier score and
sample count so calibration can be evaluated rather than assumed.

The combined research score is:

`0.40*(2P-1) + 0.20*tanh(3*r20 + 2*r5) + 0.15*macro`
`+ 0.10*sentiment + 0.05*candle_score + 0.10*intraday_score`

`intraday_score = tanh(4*r5_intraday + 2*r1_intraday)`

The system labels a signal long at `score >= 0.20`, short at `score <= -0.20`,
and flat otherwise. Sentiment and candle evidence are deliberately small
weights because they are noisy.

### Bulk allocation

For selected trades:

`batch_cap = equity * max_new_exposure_pct`

`notional_i = min(equity * max_position_pct, batch_cap / selected_count)`

The default is at most five new positions, 3% per position and 10% new
exposure per cycle. These are safety defaults, not claims of optimality.

### Volatility targeting and transaction-cost gate

Each notional is scaled by:

`volatility_scale = min(1, target_annual_volatility / forecast_volatility)`

The initial target annualized volatility is 12%. The research layer estimates
a conservative round-trip cost:

`cost = 0.0015 + 0.0005*sqrt(volatility) +`
`0.0005/sqrt(average_dollar_volume / 1,000,000)`

An opportunity is rejected unless its estimated edge exceeds this cost. This
is only a proxy until historical bid/ask and fill data are available.

### Correlation and portfolio tail risk

Candidates with trailing return correlation above 0.85 to an already selected
candidate are skipped. For selected weights `w_i`, the historical portfolio
return is:

`R_p,t = sum_i(w_i * R_i,t)`

Historical 95% VaR is `-percentile_5%(R_p)`, and CVaR is the mean loss in that
5% tail. The Portfolio Council rejects a batch above 4% VaR or 6% CVaR.

## Position exits and reconciliation

For an existing long position:

`stop = entry_price - 2 * ATR(14)`

For an existing short position:

`stop = entry_price + 2 * ATR(14)`

The Position Manager requests an exit when the stop is breached, the new
research direction is opposite to the position, or the new direction is flat.
The Execution Commander submits a closing market order only after checking
existing open orders. Each broker order is then queried for `status`,
`filled_qty`, `filled_avg_price` and `filled_at`; snapshots are persisted in
`logs/trade_ledger.json`. Submitted is not treated as filled.

## Candle topics implemented

The Market Intelligence Directorate detects doji, hammer, shooting star,
bullish engulfing, bearish engulfing, 20-bar breakout and 20-bar breakdown.
Candles are confirmation features. They should not override portfolio risk,
liquidity, or data-quality checks.

## Operating policy

The scheduled workflow runs every 30 minutes during the configured market
window and submits only paper orders. A batch is analyzed completely before
any order is submitted. Existing positions and open orders are excluded to
make overlapping cycles idempotent at the symbol level. These methods improve
discipline and reduce known biases; they do not establish profitability without
a point-in-time backtest using survivorship-free data, realistic costs and
completed trade outcomes.
