"""
Resolves a git-push race on logs/history.json for the "Commit updated
trade history log" step in .github/workflows/trading-pipeline.yml.

Two Actions runs (a manual trigger racing the scheduled cron, or two
manual triggers) can both finish around the same time; whichever pushes
second gets rejected (non-fast-forward), since there's no merge/rebase
logic for a plain `git push`. A text-level git rebase/merge of the JSON
file would likely conflict anyway -- both runs append to the same
trailing region of the same array. Reconciling by content instead sidesteps
that entirely: extract just the records this run actually produced, and
re-append them onto whatever the freshest remote copy turns out to be.

Two subcommands, run from the repo root by the workflow step:

    python3 scripts/reconcile_history_log.py save <job_start_iso>
        Reads logs/history.json (this run's own local copy, potentially
        stale relative to origin/main by now) and saves just the records
        with run_at >= job_start_iso to /tmp/my_new_records.json.

    python3 scripts/reconcile_history_log.py merge
        Reads logs/history.json (expected to already be a fresh copy of
        origin/main's version -- the workflow does `git reset --hard
        origin/main` before calling this) and /tmp/my_new_records.json,
        appends the latter onto the former, trims to the same history cap
        tools/run_logger.py uses, and writes the result back to
        logs/history.json.
"""

import json
import sys

HISTORY_PATH = "logs/history.json"
SAVED_RECORDS_PATH = "/tmp/my_new_records.json"
MAX_HISTORY_ENTRIES = 500  # matches tools/run_logger.py


def _load_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save(job_start_iso: str) -> None:
    local = _load_json(HISTORY_PATH, [])
    mine = [r for r in local if r.get("run_at", "") >= job_start_iso]
    with open(SAVED_RECORDS_PATH, "w") as f:
        json.dump(mine, f)
    print(f"Saved {len(mine)} of this run's own records (run_at >= {job_start_iso}) for reconciliation.")


def merge() -> None:
    remote = _load_json(HISTORY_PATH, [])
    mine = _load_json(SAVED_RECORDS_PATH, [])
    combined = (remote + mine)[-MAX_HISTORY_ENTRIES:]
    with open(HISTORY_PATH, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"Reconciled: {len(remote)} remote + {len(mine)} mine -> {len(combined)} combined")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: reconcile_history_log.py save <job_start_iso> | merge", file=sys.stderr)
        sys.exit(2)

    command = sys.argv[1]
    if command == "save":
        if len(sys.argv) != 3:
            print("save requires a job_start_iso argument", file=sys.stderr)
            sys.exit(2)
        save(sys.argv[2])
    elif command == "merge":
        merge()
    else:
        print(f"Unknown command: {command!r}", file=sys.stderr)
        sys.exit(2)
