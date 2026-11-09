"""Quantile forecasts of SOFR on turn days, scored against two baselines.

Run from the project root, after model_lasso.py and intervals.py:
    python src/turn_quantiles.py

Target
    SOFR minus the top of the target range, in bp (spread_hi_bp), on turn
    days: the month_end flag in daily.csv. Each turn has one type, highest
    first: year-end (year_end), quarter-end (quarter_end), month-end.

Model
    Linear quantile regression (scikit-learn QuantileRegressor, solver
    "highs", alpha=0, so no penalty) at the 50th, 90th and 95th percentiles.
    Features: a quarter-end flag and past-only scarcity state, all as
    build_dataset.py already defines them for the turn date: rrp_bn_lag1 and
    tga_bn_lag1 (last observation strictly before the turn) and base_bp (the
    median spread over the 60 previous days). Month-ends and quarter-ends are
    pooled with the flag; year-ends get no model.
    Walk-forward: each turn is forecast from a fit on all earlier month-end
    and quarter-end turns only, standardised on those turns only. A variant
    fits on the earlier turns within the previous 500 SOFR days.

Baselines
    (a) pooled: the existing interval method mapped to these quantiles. The
        two-stage forecast of the deviation for the turn day, plus the 50th,
        90th and 95th percentile of the last 250 walk-forward errors before
        it (as src/intervals.py does for 10 and 90), plus base_bp.
    (b) empirical: the same percentiles of all earlier turns of the same type.

Scores, by turn type, on the turns where every method has a forecast:
pinball loss for each quantile (lower is better) and coverage, the share of
turns that printed at or below the forecast quantile (target 50, 90, 95%).

Writes data/processed/turn_quantiles.csv (one row per turn and method) and
data/processed/turn_quantile_scores.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
from intervals import WINDOW as ERR_WINDOW  # noqa: E402

TAUS = (0.50, 0.90, 0.95)
QCOLS = ["q50", "q90", "q95"]
STATE = ["rrp_bn_lag1", "tga_bn_lag1", "base_bp"]
FEATURES = ["is_qe", *STATE]
MIN_TRAIN = 24        # earlier turns needed before the model forecasts
RECENT_DAYS = 500     # window of the recent-only variant, in SOFR days
MIN_TYPE = 8          # earlier turns of a type needed for baseline (b)


def turn_days(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values("date").reset_index(drop=True)
    d["row"] = np.arange(len(d))
    t = d[d["month_end"] == 1].copy()
    t["type"] = np.where(t["year_end"] == 1, "year-end",
                         np.where(t["quarter_end"] == 1, "quarter-end", "month-end"))
    t["is_qe"] = (t["type"] == "quarter-end").astype(float)
    t["y"] = t["spread_hi_bp"]
    return t.reset_index(drop=True)


def fit_quantiles(train: pd.DataFrame, x: pd.DataFrame) -> list[float]:
    X = train[FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z, z = (X - mu) / sd, (x[FEATURES].to_numpy(float) - mu) / sd
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tau in TAUS:
            m = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs").fit(Z, train["y"])
            out.append(float(m.predict(z)[0]))
    return list(np.maximum.accumulate(out))  # no crossing: q50 <= q90 <= q95


def walk_forward(t: pd.DataFrame) -> pd.DataFrame:
    """Model forecasts for every month-end and quarter-end, both variants."""
    pool = t[t["type"] != "year-end"].dropna(subset=STATE)
    rows = []
    for i, r in pool.iterrows():
        earlier = pool[pool["date"] < r["date"]]
        recent = earlier[earlier["row"] >= r["row"] - RECENT_DAYS]
        for method, train in (("model, all turns", earlier), ("model, last 500 days", recent)):
            if len(train) >= MIN_TRAIN or (method.endswith("days") and len(train) >= 12):
                q = fit_quantiles(train, r.to_frame().T)
                rows.append({"date": r["date"], "type": r["type"], "method": method,
                             "n_train": len(train), **dict(zip(QCOLS, q))})
    return pd.DataFrame(rows)


def baseline_empirical(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in t[t["type"] != "year-end"].iterrows():
        past = t[(t["date"] < r["date"]) & (t["type"] == r["type"])]["y"].dropna()
        if len(past) >= MIN_TYPE:
            rows.append({"date": r["date"], "type": r["type"], "method": "empirical, same type",
                         "n_train": len(past), **dict(zip(QCOLS, np.quantile(past, TAUS)))})
    return pd.DataFrame(rows)


def baseline_pooled(t: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    o = oos.sort_values("date").reset_index(drop=True)
    err = (o["y"] - o["two_stage"]).to_numpy(float)
    pos = {d: i for i, d in enumerate(o["date"])}
    rows = []
    for _, r in t[t["type"] != "year-end"].iterrows():
        i = pos.get(r["date"])
        if i is None or i < ERR_WINDOW or pd.isna(r["base_bp"]):
            continue
        e = err[i - ERR_WINDOW:i]               # errors of earlier targets only
        q = r["base_bp"] + o.at[i, "two_stage"] + np.quantile(e, TAUS)
        rows.append({"date": r["date"], "type": r["type"], "method": "pooled 80% method",
                     "n_train": ERR_WINDOW, **dict(zip(QCOLS, q))})
    return pd.DataFrame(rows)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    u = y - q
    return float(np.mean(np.maximum(tau * u, (tau - 1) * u)))


def score(fc: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    f = fc.merge(t[["date", "y"]], on="date")
    common = set.intersection(*(set(g["date"]) for _, g in f.groupby("method")))
    f = f[f["date"].isin(common)]
    rows = []
    for typ in ("month-end", "quarter-end", "all"):
        for method, g in f.groupby("method", sort=False):
            g = g if typ == "all" else g[g["type"] == typ]
            row = {"type": typ, "method": method, "n": len(g),
                   "from": g["date"].min().date(), "to": g["date"].max().date()}
            for c, tau in zip(QCOLS, TAUS):
                row[f"pinball_{c}"] = pinball(g["y"].to_numpy(), g[c].to_numpy(), tau)
                row[f"cover_{c}"] = float((g["y"] <= g[c]).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    daily = pd.read_csv(PROC / "daily.csv", parse_dates=["date"])
    oos = pd.read_csv(PROC / "oos_predictions.csv", parse_dates=["date"])
    t = turn_days(daily)

    # No look-ahead: every state variable is dated strictly before the turn,
    # and base_bp is built from earlier days only (build_dataset.py).
    assert (t["rrp_lag1_date"].isna() | (pd.to_datetime(t["rrp_lag1_date"]) < t["date"])).all()
    assert (t["tga_lag1_date"].isna() | (pd.to_datetime(t["tga_lag1_date"]) < t["date"])).all()

    fc = pd.concat([walk_forward(t), baseline_pooled(t, oos), baseline_empirical(t)],
                   ignore_index=True)
    fc = fc.merge(t[["date", "y"]], on="date").sort_values(["date", "method"])
    fc.to_csv(PROC / "turn_quantiles.csv", index=False)
    sc = score(fc.drop(columns="y"), t)
    sc.to_csv(PROC / "turn_quantile_scores.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(sc.round(3).to_string(index=False))
    ye = t[t["type"] == "year-end"]
    print(f"\nyear-ends: n = {len(ye)} ({', '.join(str(d.year) for d in ye['date'])}); no model")
    print(f"wrote {PROC / 'turn_quantiles.csv'} and turn_quantile_scores.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
