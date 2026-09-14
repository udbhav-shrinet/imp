# Team S — Execution: Formulas & Code Reference

Implemented in `agents/team_s_execution.py` and `tools/alpaca_tools.py`.

## TWAP (Time-Weighted Average Price) slicing

Splits one large order into `N` equal-sized child orders instead of
firing the whole size at once, to avoid moving the price against
yourself (even on paper, this simulates real-world slippage avoidance):
```
child_qty_i = total_qty / N   for i in 1..N
```
`slice_order_twap()`. A true TWAP execution algorithm would also space
these child orders out over a time window (e.g. one every few minutes)
rather than submitting all N back-to-back — that's the natural next
step if order sizes get large enough for intraday price impact to
matter.

## VWAP (Volume-Weighted Average Price)

Not yet implemented. The formula, for reference:
```
VWAP = sum(price_i * volume_i) / sum(volume_i)
```
over the bars in the execution window. A VWAP-style slicer would size
each child order proportional to that period's typical volume profile
rather than splitting evenly like TWAP does — useful once intraday
volume data is wired into Team C.

## The gate

`execute_team_a_report()` is the *only* function in the whole codebase
that calls `submit_order` / `slice_order_twap`, and it refuses to act
unless `report["approved"]` is `True` — i.e. unless Team A's Risker
signed off. This is what makes "lower teams cannot execute trades" an
actual code-level guarantee, not just a design intention.
