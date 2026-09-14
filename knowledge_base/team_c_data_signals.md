# Team C — Data & Signals: Formulas & Code Reference

I couldn't legitimately source the actual copyrighted book PDFs (Hilpisch,
*Text Mining in Practice*) to drop in this folder — downloading/redistributing
them would be copyright infringement. What follows instead are the exact
formulas each Team C agent uses, and where they're implemented in this repo.
If you own those books, they're a good deeper read on the same material.

## Statistician

Implemented in `agents/team_c_data.py`.

**Simple Moving Average (SMA, window `w`)**
```
SMA_t = (1/w) * sum(P_{t-i} for i in 0..w-1)
```
`compute_moving_averages()`

**Relative Strength Index (RSI, period 14)**
```
RS  = avg_gain / avg_loss   (over the period, gains/losses from day-over-day deltas)
RSI = 100 - 100 / (1 + RS)
```
`compute_rsi()`

**Bollinger Bands (window 20, k=2 standard deviations)**
```
mid   = SMA_20(P)
upper = mid + k * std_20(P)
lower = mid - k * std_20(P)
```
`compute_bollinger_bands()`

**Z-score (mean reversion signal, window 20)**
```
z_t = (P_t - mean_20(P)) / std_20(P)
```
`compute_z_score()`

**Average True Range (ATR, period 14)**
```
TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
ATR  = SMA_14(TR)
```
`compute_atr()`

## Forecaster

**ARIMA(1,1,1)** — one autoregressive lag, one order of differencing, one
moving-average lag. Fit via `statsmodels.tsa.arima.model.ARIMA` and
forecast N steps ahead. `forecast_arima()`

**Exponential Smoothing (additive trend, Holt's method)**
```
level_t = alpha * P_t + (1 - alpha) * (level_{t-1} + trend_{t-1})
trend_t = beta  * (level_t - level_{t-1}) + (1 - beta) * trend_{t-1}
forecast_{t+h} = level_t + h * trend_t
```
Parameters (alpha, beta) are fit by `statsmodels.tsa.holtwinters.ExponentialSmoothing`.
`forecast_exponential_smoothing()`

## Sentiment Analyst

**VADER compound score** — a lexicon + rule-based sentiment score in
[-1, 1] from `nltk.sentiment.vader.SentimentIntensityAnalyzer`, averaged
across all scraped Reddit posts + RSS headlines. `score_sentiment_vader()`

**TF-IDF (Term Frequency-Inverse Document Frequency)**
```
tf(term, doc)  = count of term in doc / total terms in doc
idf(term)      = log(N_docs / docs containing term)
tfidf          = tf * idf
```
Used here to extract the top-N most distinctive keywords across scraped
text via `sklearn.feature_extraction.text.TfidfVectorizer`. `score_tfidf_keywords()`
