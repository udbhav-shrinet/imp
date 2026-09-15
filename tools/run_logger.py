"""
Persists a JSON-serializable summary of each pipeline run to logs/history.json
so the dashboard (a published Artifact reading this file straight off GitHub)
can render trade history, team outputs, and account equity over time.
"""

import json
import os
from datetime import datetime, timezone

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
HISTORY_PATH = os.path.join(LOGS_DIR, "history.json")
MAX_HISTORY_ENTRIES = 1000


def _summarize_team_c(team_c_output: dict) -> dict:
    price_history = team_c_output["price_history"]
    return {
        "symbol": team_c_output["symbol"],
        "last_close": float(price_history["Close"].iloc[-1]),
        "last_date": str(price_history.index[-1]),
        "bars_used": len(price_history),
        "statistics": team_c_output["statistics"],
        "forecast": team_c_output["forecast"],
        "sentiment": team_c_output["sentiment"],
    }


def build_run_record(
    symbol: str,
    capital: float,
    dry_run: bool,
    team_c_output: dict,
    team_b_output: dict,
    team_a_report: dict,
    execution_result: dict | None,
    account_info: dict | None,
) -> dict:
    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "capital": capital,
        "dry_run": dry_run,
        "team_c": _summarize_team_c(team_c_output),
        "team_b": {
            # gbm_simulated_paths_sample is 100 floats of Monte-Carlo noise that
            # nothing reads back -- not the dashboard, not any later run -- and it
            # was ~30% of every record's size. At 14 runs/day the log rolls over
            # fast enough already; don't spend the budget on unread simulation
            # samples. The summary stats derived from it (drift, volatility) stay.
            "mathematician": {k: v for k, v in team_b_output["mathematician"].items()
                              if k != "gbm_simulated_paths_sample"},
            "ml_engineer": team_b_output["ml_engineer"],
            "economist": team_b_output["economist"],
        },
        "team_a_report": team_a_report,
        "execution_result": execution_result,
        "account": account_info,
    }


def append_run_record(record: dict) -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []

    history.append(record)
    history = history[-MAX_HISTORY_ENTRIES:]

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)
