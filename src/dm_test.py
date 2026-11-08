"""Diebold-Mariano test: two-stage LASSO against AR(1), squared-error loss.

Run from the project root, after model_lasso.py:
    python src/dm_test.py

Input: data/processed/oos_predictions.csv. Its ar1 column is the walk-forward
AR(1) refitted on the same rolling 500-day window as the two-stage model for
every 60-day block; it is also the model's own first stage. The target y is
the winsorised deviation (dev_w_bp), so the losses here are on that target,
the same one behind the R² figures in the app.

Loss differential d_t = e_model,t^2 - e_ar1,t^2. A negative mean means the
model forecasts better.

Variance: Newey-West HAC with a Bartlett kernel and the Newey-West (1994)
automatic bandwidth, which estimates the lag from the autocorrelation of d_t
rather than fixing it. Then the Harvey-Leybourne-Newbold (1997) small-sample
correction, DM * sqrt((T + 1 - 2h + h(h - 1) / T) / T), compared with Student
t critical values on T - 1 degrees of freedom. These are one-day-ahead
forecasts, so h = 1 and the factor is sqrt((T - 1) / T): close to one at these
sample sizes. Most of the difference from a naive t-test comes from the HAC
variance, not from HLN.

Writes data/processed/dm_test.csv with one row for the full period, one for
2024 onwards and one per calendar year.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
H = 1  # forecast horizon in days


def autocov(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    return float(np.dot(x[lag:], x[:len(x) - lag]) / len(x))


def nw_bandwidth(d: np.ndarray) -> int:
    """Newey-West (1994) automatic lag for the Bartlett kernel."""
    T = len(d)
    n = int(np.floor(4 * (T / 100) ** (2 / 9)))
    g = np.array([autocov(d, j) for j in range(n + 1)])
    s0 = g[0] + 2 * g[1:].sum()
    s1 = 2 * np.sum(np.arange(1, n + 1) * g[1:])
    if s0 <= 0:
        return 0
    gamma = 1.1447 * ((s1 / s0) ** 2) ** (1 / 3)
    return int(min(np.floor(gamma * T ** (1 / 3)), T - 1))


def hac_var(d: np.ndarray, lag: int) -> float:
    """Long-run variance of d with Bartlett weights up to `lag`."""
    v = autocov(d, 0)
    for j in range(1, lag + 1):
        v += 2 * (1 - j / (lag + 1)) * autocov(d, j)
    return v


def dm(y: np.ndarray, f_model: np.ndarray, f_bench: np.ndarray, h: int = H) -> dict:
    e1, e2 = y - f_model, y - f_bench
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    base = {"n": T,
            "rmse_model_bp": float(np.sqrt(np.mean(e1 ** 2))),
            "rmse_ar1_bp": float(np.sqrt(np.mean(e2 ** 2))),
            "r2_vs_ar1": float(1 - np.sum(e1 ** 2) / np.sum(e2 ** 2))}
    # When the second stage selects nothing, the model *is* AR(1) and d is
    # rounding noise of order 1e-16. A t-ratio of noise over noise can come out
    # "significant", so the test is reported as undefined instead.
    if np.max(np.abs(f_model - f_bench)) < 1e-9:
        return {**base, "mean_loss_diff": 0.0, "nw_lag": np.nan, "dm_stat": np.nan,
                "hln_stat": np.nan, "p_two_sided": np.nan, "p_model_better": np.nan,
                "note": "forecasts identical to AR(1); test undefined"}
    lag = nw_bandwidth(d)
    lrv = hac_var(d, lag)
    stat = d.mean() / np.sqrt(lrv / T)
    hln = stat * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    df = T - 1
    return {
        **base,
        "mean_loss_diff": float(d.mean()),
        "nw_lag": lag,
        "dm_stat": float(stat),
        "hln_stat": float(hln),
        "p_two_sided": float(2 * stats.t.sf(abs(hln), df)),
        "p_model_better": float(stats.t.cdf(hln, df)),
        "note": "",
    }


def table(oos: pd.DataFrame) -> pd.DataFrame:
    def run(label: str, g: pd.DataFrame) -> dict:
        return {"period": label,
                "start": g["date"].min().date(), "end": g["date"].max().date(),
                **dm(g["y"].to_numpy(float), g["two_stage"].to_numpy(float),
                     g["ar1"].to_numpy(float))}

    rows = [run("all", oos), run("2024+", oos[oos["date"].dt.year >= 2024])]
    rows += [run(str(yr), g) for yr, g in oos.groupby(oos["date"].dt.year)]
    return pd.DataFrame(rows)


def main() -> int:
    oos = pd.read_csv(PROC / "oos_predictions.csv", parse_dates=["date"])
    out = table(oos)
    out.to_csv(PROC / "dm_test.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.round(4).to_string(index=False))
    print(f"\nwrote {PROC / 'dm_test.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
