"""
Macro data tools used by Team B's Economist: St. Louis Fed (FRED) API.

Free, keyed API (register at https://fred.stlouisfed.org/docs/api/api_key.html).
Every function degrades gracefully -- a missing key, a network failure, or a
missing observation returns None (or an empty dict/list) rather than raising,
since a stale or unavailable macro read should never take down a trading run.

Two macro series are used, both free and updated daily on business days:
    T10Y2Y  -- 10-Year minus 2-Year Treasury Constant Maturity spread (the
               yield curve). Negative = inverted = late-cycle/recession risk.
               Steepening positive = expansion.
    VIXCLS  -- CBOE Volatility Index close. >25 = risk-off/turbulent,
               <18 = risk-on/calm.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT_SECONDS = 10


def _fred_get(series_id: str, params: dict) -> list[dict] | None:
    """Low-level FRED call. Returns the raw `observations` list, or None on any failure."""
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        print(f"[fred_tools] FRED_API_KEY not set -- skipping {series_id}.")
        return None

    query = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        **params,
    }
    try:
        response = requests.get(FRED_BASE_URL, params=query, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json().get("observations", [])
    except Exception as exc:
        print(f"[fred_tools] Failed to fetch {series_id}: {exc}")
        return None


def get_latest_value(series_id: str) -> float | None:
    """
    Latest available reading for a FRED series (e.g. "T10Y2Y", "VIXCLS").
    FRED marks missing days as "." (weekends/holidays) -- these are skipped
    in favor of the most recent real observation. Returns None on any failure
    or if no real observation is found in the lookback window.
    """
    observations = _fred_get(
        series_id,
        {"sort_order": "desc", "limit": 10},  # a few days' cushion past holidays/weekends
    )
    if not observations:
        return None

    for obs in observations:
        value = obs.get("value")
        if value not in (None, "."):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def get_history(series_id: str, years: int = 2) -> dict[str, float]:
    """
    Daily history for a FRED series over the trailing `years`, as a
    {date_str ("YYYY-MM-DD"): value} dict -- used to align a historical macro
    reading to each row of a model's training window (see
    agents/team_b_quants.py's ML Engineer), rather than repeating today's
    single snapshot across every historical row.

    Missing/"." observations are simply omitted; callers should forward-fill
    from the nearest earlier date when aligning by date.
    """
    import datetime

    start = (datetime.date.today() - datetime.timedelta(days=365 * years)).isoformat()
    observations = _fred_get(series_id, {"observation_start": start, "sort_order": "asc"})
    if not observations:
        return {}

    history = {}
    for obs in observations:
        value = obs.get("value")
        if value in (None, "."):
            continue
        try:
            history[obs["date"]] = float(value)
        except ValueError:
            continue
    return history


def get_macro_snapshot() -> dict:
    """
    One-shot snapshot of both macro series' latest values -- call this once
    per pipeline run (not once per symbol; macro data doesn't vary by
    symbol), then pass the result down to every symbol's Team B Economist.
    """
    return {
        "t10y2y": get_latest_value("T10Y2Y"),
        "vixcls": get_latest_value("VIXCLS"),
    }


def get_macro_history(years: int = 2) -> dict:
    """
    One-shot fetch of both series' trailing history -- call this once per
    run alongside get_macro_snapshot() so the ML Engineer can align a
    historical macro_score to each row of its training window.
    """
    return {
        "t10y2y": get_history("T10Y2Y", years=years),
        "vixcls": get_history("VIXCLS", years=years),
    }
