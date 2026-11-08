"""80% prediction intervals from past walk-forward errors only.

Run from the project root, after model_lasso.py:
    python src/intervals.py

For the forecast of target date t, the error of every earlier forecast whose
target date is before t is already known when the forecast is made (the
forecast for t is made on the business day before t, after that day's SOFR is
published). The interval is the forecast plus the empirical 10th and 90th
percentiles of the last 250 of those errors. A forecast with fewer than 250
past errors gets no interval, so the first 250 out-of-sample days have none.
Nothing from t or later enters the bounds; check_no_lookahead() asserts it.

Errors are on the same winsorised target the model forecasts (dev_w_bp).

Adds lo80 and hi80 next to the forecasts in data/processed/oos_predictions.csv
and writes the coverage table to data/processed/interval_coverage.csv.
Re-run after model_lasso.py, which rewrites the forecasts file without them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
WINDOW = 250
Q = (0.10, 0.90)


def error_quantiles(errors: np.ndarray, window: int = WINDOW, q=Q) -> tuple[float, float]:
    """10th and 90th percentile of the last `window` errors, or NaN if too few."""
    e = np.asarray(errors, float)[-window:]
    if len(e) < window:
        return np.nan, np.nan
    lo, hi = np.quantile(e, q)
    return float(lo), float(hi)


def add_bounds(oos: pd.DataFrame, col: str = "two_stage", window: int = WINDOW) -> pd.DataFrame:
    o = oos.sort_values("date").reset_index(drop=True).copy()
    err = (o["y"] - o[col]).to_numpy(float)
    lo = np.full(len(o), np.nan)
    hi = np.full(len(o), np.nan)
    for i in range(len(o)):
        ql, qh = error_quantiles(err[:i], window)   # errors of targets before i only
        lo[i], hi[i] = o.at[i, col] + ql, o.at[i, col] + qh
    o["lo80"], o["hi80"] = lo, hi
    return o


def check_no_lookahead(oos: pd.DataFrame) -> None:
    """Changing every outcome from day i on must leave bounds before and at i unchanged."""
    o = add_bounds(oos)
    i = len(o) // 2
    bumped = oos.sort_values("date").reset_index(drop=True).copy()
    bumped.loc[i:, "y"] += 1000.0
    b = add_bounds(bumped)
    assert np.allclose(o.loc[:i, ["lo80", "hi80"]], b.loc[:i, ["lo80", "hi80"]],
                       equal_nan=True), "interval uses outcomes from its own date or later"


def coverage(o: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    g = o.dropna(subset=["lo80"]).merge(flags, on="date", how="left")
    g["hit"] = (g["y"] >= g["lo80"]) & (g["y"] <= g["hi80"])
    g["width"] = g["hi80"] - g["lo80"]
    g["below"] = g["y"] < g["lo80"]
    g["above"] = g["y"] > g["hi80"]

    def row(label: str, s: pd.DataFrame) -> dict:
        return {"group": label, "n": len(s), "coverage": s["hit"].mean(),
                "below": s["below"].mean(), "above": s["above"].mean(),
                "mean_width_bp": s["width"].mean()}

    rows = [row("all", g)]
    rows += [row(str(y), s) for y, s in g.groupby(g["date"].dt.year)]
    rows += [row("month-end", g[g["month_end"] == 1]),
             row("other days", g[g["month_end"] != 1]),
             row("quarter-end", g[g["quarter_end"] == 1]),
             row("month-end, 2024+", g[(g["month_end"] == 1) & (g["date"].dt.year >= 2024)]),
             row("other days, 2024+", g[(g["month_end"] != 1) & (g["date"].dt.year >= 2024)])]
    return pd.DataFrame(rows)


def main() -> int:
    oos = pd.read_csv(PROC / "oos_predictions.csv", parse_dates=["date"])
    oos = oos.drop(columns=["lo80", "hi80"], errors="ignore")
    check_no_lookahead(oos)
    o = add_bounds(oos)
    o.to_csv(PROC / "oos_predictions.csv", index=False)

    flags = pd.read_csv(PROC / "daily.csv", parse_dates=["date"],
                        usecols=["date", "month_end", "quarter_end"])
    cov = coverage(o, flags)
    cov.to_csv(PROC / "interval_coverage.csv", index=False)
    print(cov.round(3).to_string(index=False))
    print(f"\nfirst interval {o.dropna(subset=['lo80'])['date'].min().date()}; "
          f"wrote lo80/hi80 to oos_predictions.csv and interval_coverage.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
