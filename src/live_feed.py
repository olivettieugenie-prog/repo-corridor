"""Read the live forecast log that the daily workflow commits to the data branch.

The app calls load() through st.cache_data(ttl=3600). Order of sources:
  1. the raw GitHub URL of data/live/forecast.csv on the "data" branch;
     a successful fetch also refreshes the local copy;
  2. the local copy at data/live/forecast.csv (the last good fetch, or the
     output of running src/forecast_next.py here);
  3. nothing, and the app says so.
REPO_LIVE_URL and REPO_LIVE_LOCAL override the URL and the local path, which
the tests use to simulate a failed fetch without touching the real copy.
Uses only the standard library and pandas, so requirements.txt is unchanged.
"""
from __future__ import annotations

import io
import os
import ssl
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_URL = ("https://raw.githubusercontent.com/olivettieugenie-prog/repo-corridor/"
           "data/data/live/forecast.csv")
LOCAL = Path(os.environ.get("REPO_LIVE_LOCAL", ROOT / "data" / "live" / "forecast.csv"))
DATES = ["as_of_date", "target_date"]
# The 20-day forward path the same job writes next to the log; it is
# replaced on every run. REPO_FORWARD_URL and REPO_FORWARD_LOCAL override it.
FORWARD_URL = RAW_URL.replace("forecast.csv", "forward.csv")
FORWARD_LOCAL = ROOT / "data" / "live" / "forward.csv"
FORWARD_DATES = ["as_of_date", "date"]
# The daily table the job rebuilt on its last run, for display (header, strip,
# charts, analogs). REPO_DAILY_URL and REPO_DAILY_LOCAL override it; the last
# fallback is main's data/processed/daily.csv, the frozen backtest snapshot.
DAILY_URL = RAW_URL.replace("forecast.csv", "daily.csv")
DAILY_LOCAL = ROOT / "data" / "live" / "daily.csv"
DAILY_MAIN = ROOT / "data" / "processed" / "daily.csv"


def _ssl_context() -> ssl.SSLContext:
    """certifi's CA bundle when it is installed, the system's otherwise.
    The python.org builds for macOS ship without system root certificates
    until "Install Certificates.command" is run, and urllib then fails every
    HTTPS fetch; certifi avoids that without adding a dependency."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _parse(text: str, dates=DATES, required: str = "forecast_bp") -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), parse_dates=dates)
    if df.empty or required not in df.columns:
        raise ValueError(f"table has no rows or no {required} column")
    return df


def load(url: str | None = None, local: Path | None = None, timeout: float = 10,
         dates=DATES, required: str = "forecast_bp") -> tuple[pd.DataFrame | None, str]:
    """Return (log, source) where source is 'github', 'local' or 'none'."""
    url = url or os.environ.get("REPO_LIVE_URL", RAW_URL)
    local = Path(local or os.environ.get("REPO_LIVE_LOCAL", LOCAL))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "repo-corridor-app"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            text = r.read().decode("utf-8")
        df = _parse(text, dates, required)
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(text)
        except OSError:
            pass  # read-only filesystem: the fetch still counts
        return df, "github"
    except Exception:
        pass
    try:
        return _parse(local.read_text(), dates, required), "local"
    except Exception:
        return None, "none"


def load_forward(url: str | None = None, local: Path | None = None,
                 timeout: float = 10) -> tuple[pd.DataFrame | None, str]:
    """The forward path, with the same GitHub-then-local fallback."""
    url = url or os.environ.get("REPO_FORWARD_URL", FORWARD_URL)
    local = Path(local or os.environ.get("REPO_FORWARD_LOCAL", FORWARD_LOCAL))
    return load(url, local, timeout, dates=FORWARD_DATES)


def load_daily(url: str | None = None, local: Path | None = None, main: Path | None = None,
               timeout: float = 10) -> tuple[pd.DataFrame, str]:
    """The rebuilt daily table: GitHub's data branch, then the local copy,
    then main's backtest snapshot. Returns (table, 'github'|'local'|'main')."""
    url = url or os.environ.get("REPO_DAILY_URL", DAILY_URL)
    local = Path(local or os.environ.get("REPO_DAILY_LOCAL", DAILY_LOCAL))
    df, source = load(url, local, timeout, dates=["date"], required="sofr")
    if df is not None:
        return df.sort_values("date").reset_index(drop=True), source
    snap = pd.read_csv(main or DAILY_MAIN, parse_dates=["date"])
    return snap.sort_values("date").reset_index(drop=True), "main"


def latest_per_target(df: pd.DataFrame) -> pd.DataFrame:
    """One row per target date: the most recent run, since a holiday or a
    manual rerun can forecast the same target twice."""
    return (df.sort_values("run_time_utc")
              .drop_duplicates("target_date", keep="last")
              .sort_values("target_date").reset_index(drop=True))
