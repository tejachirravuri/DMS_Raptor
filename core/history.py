"""Persistent run history — save/load RunSummary as JSON files."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import List

from .config import RunSummary, HISTORY_DIR_NAME

logger = logging.getLogger(__name__)


def _history_dir(app_root: str) -> str:
    """Return the history directory path, creating it if necessary."""
    d = os.path.join(app_root, HISTORY_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _safe_filename(s: str) -> str:
    """Sanitize a string for use in a filename."""
    return re.sub(r"[^\w\-.]", "_", s)


def save_run(app_root: str, summary: RunSummary) -> str:
    """Save a RunSummary to a JSON file.  Returns the file path."""
    hdir = _history_dir(app_root)
    ts = _safe_filename(summary.run_timestamp or "unknown")
    policy = _safe_filename(summary.policy or "unknown")
    fname = f"{ts}_{policy}.json"
    path = os.path.join(hdir, fname)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        logger.info("Saved run history: %s", path)
    except Exception:
        logger.exception("Failed to save run history")
    return path


def load_all_runs(app_root: str) -> List[RunSummary]:
    """Load all RunSummary objects from the history directory."""
    hdir = _history_dir(app_root)
    results: List[RunSummary] = []

    if not os.path.isdir(hdir):
        return results

    for fname in sorted(os.listdir(hdir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(hdir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
            results.append(RunSummary.from_dict(d))
        except Exception:
            logger.warning("Failed to load history file: %s", fpath, exc_info=True)

    logger.info("Loaded %d runs from history", len(results))
    return results


def delete_run(app_root: str, summary: RunSummary) -> bool:
    """Delete a specific run's history file."""
    hdir = _history_dir(app_root)
    ts = _safe_filename(summary.run_timestamp or "unknown")
    policy = _safe_filename(summary.policy or "unknown")
    fname = f"{ts}_{policy}.json"
    path = os.path.join(hdir, fname)

    if os.path.isfile(path):
        os.remove(path)
        logger.info("Deleted history file: %s", path)
        return True
    return False


def clear_history(app_root: str) -> int:
    """Delete all history files.  Returns count of deleted files."""
    hdir = _history_dir(app_root)
    count = 0
    for fname in os.listdir(hdir):
        if fname.endswith(".json"):
            try:
                os.remove(os.path.join(hdir, fname))
                count += 1
            except Exception:
                pass
    logger.info("Cleared %d history files", count)
    return count
