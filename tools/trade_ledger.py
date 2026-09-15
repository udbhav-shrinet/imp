"""Durable order/fill snapshots for paper-trading analysis."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

LEDGER_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "trade_ledger.json")


def order_ids_from_execution(*executions: dict) -> list[str]:
    ids = []
    for execution in executions:
        for item in execution.get("orders", []):
            order = item.get("order", item)
            if order.get("id"):
                ids.append(order["id"])
            for child in item.get("orders", []):
                if child.get("id"):
                    ids.append(child["id"])
    return list(dict.fromkeys(ids))


def append_fill_snapshots(snapshots: list[dict]) -> None:
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    try:
        with open(LEDGER_PATH, encoding="utf-8") as handle:
            ledger = json.load(handle)
    except (OSError, json.JSONDecodeError):
        ledger = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for snapshot in snapshots:
        ledger.append({"observed_at": timestamp, **snapshot})
    with open(LEDGER_PATH, "w", encoding="utf-8") as handle:
        json.dump(ledger[-5000:], handle, indent=2, default=str)
