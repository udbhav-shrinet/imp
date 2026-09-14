# Knowledge Base

The actual books referenced in the original spec (Hilpisch, Lopez de
Prado, Shreve, Harris, Chan, Taleb) are commercial and copyrighted, so
they aren't bundled here — downloading/redistributing full copies would
be infringement. If you own them, they remain a good deeper read.

Instead, each team has a markdown file with the exact formulas/algorithms
this codebase implements and a pointer to the corresponding function:

- [`team_c_data_signals.md`](./team_c_data_signals.md) — SMA, RSI,
  Bollinger Bands, Z-score, ATR, ARIMA, exponential smoothing, VADER, TF-IDF
- [`team_b_quants.md`](./team_b_quants.md) — Ito's Lemma / GBM,
  Ornstein-Uhlenbeck, regime detection, Random Forest + walk-forward validation
- [`team_a_strategy_risk.md`](./team_a_strategy_risk.md) — Kelly
  Criterion, VaR/CVaR, GARCH(1,1), cointegration (sketch)
- [`team_s_execution.md`](./team_s_execution.md) — TWAP slicing, VWAP (sketch)

If you want true retrieval-augmented context instead (the original ask),
buy the PDFs, drop them in this folder, and wire a LangChain
`RetrievalQA` chain over a vector store into each team's agents.
