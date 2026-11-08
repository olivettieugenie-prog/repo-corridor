"""Read the live forecast log that the daily workflow commits to the data branch.

The app calls load() through st.cache_data(ttl=3600). Order of sources:
  1. the raw GitHub URL of data/live/forecast.csv on the "data" branch;
     a successful fetch also refreshes the local copy;
  2. the local copy at data/live/forecast.csv (the last good fetch, or the
     output of running src/forecast_next.py here);
  3. nothing, and the app says so.
REPO_LIVE_URL overrides the URL, which the tests use to simulate a failure.
Uses only the standard library and pandas, so requirements.txt is unchanged.
"""
from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_URL = ("https://raw.githubusercontent.com/olivettieugenie-prog/repo-corridor/"
           "data/data/live/forecast.csv")
LOCAL = ROOT / "data" / "live" / "forecast.csv"
DATES = ["as_of_date", "target_date"]


def _parse(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), parse_dates=DATES)
    if df.empty or "forecast_bp" not in df.columns:
        raise ValueError("forecast log has no rows")
    return df


def load(url: str | None = None, local: Path = LOCAL, timeout: float = 10) -> tuple[pd.DataFrame | None, str]:
    """Return (log, source) where source is 'github', 'local' or 'none'."""
    url = url or os.environ.get("REPO_LIVE_URL", RAW_URL)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "repo-corridor-app"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8")
        df = _parse(text)
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(text)
        except OSError:
            pass  # read-only filesystem: the fetch still counts
        return df, "github"
    except Exception:
        pass
    try:
        return _parse(local.read_text()), "local"
    except Exception:
        return None, "none"


def latest_per_target(df: pd.DataFrame) -> pd.DataFrame:
    """One row per target date: the most recent run, since a holiday or a
    manual rerun can forecast the same target twice."""
    return (df.sort_values("run_time_utc")
              .drop_duplicates("target_date", keep="last")
              .sort_values("target_date").reset_index(drop=True))
