# Knowledge Base

Drop reference PDFs here for retrieval-augmented context per team. Suggested
titles (source your own copies; none are bundled in this repo):

**Team C — Data & Signals**
- *Python for Finance: Mastering Data-Driven Finance* — Yves Hilpisch
- *Text Mining in Practice with R and Python*

**Team B — Quants**
- *Advances in Financial Machine Learning* — Marcos Lopez de Prado
- *Stochastic Calculus for Finance I & II* — Steven E. Shreve
- *Trading and Exchanges: Market Microstructure for Practitioners* — Larry Harris

**Team A — Strategy & Risk**
- *Quantitative Trading* — Ernie Chan
- *Dynamic Hedging: Managing Vanilla and Exotic Options* — Nassim Nicholas Taleb
- *Algorithmic Trading: Winning Strategies and Their Rationale* — Ernie Chan

Wire these into a retriever (e.g. a LangChain `RetrievalQA` chain over a
vector store) and attach it to each team's agents as needed.
