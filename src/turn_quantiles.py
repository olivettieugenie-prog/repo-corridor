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
data/processed/turn_quantile_scores.csv. It also refits the model on every
month-end and quarter-end so far and writes the coefficients, in raw units,
to data/processed/turn_quantile_coefs.csv: the app evaluates them with numpy,
so it needs no scikit-learn at runtime.

Horizon and calibration
    The app forecasts turns weeks ahead, so the model is re-scored with the
    scarcity state taken h settlement days before each turn, h = 1 (the day
    before, as above), 5, 10 and 15. A forecast made h days ahead is fitted
    only on turns whose outcome was known by then.
    Split-conformal calibration then shifts each quantile by the matching
    empirical quantile of earlier out-of-sample errors (y minus forecast) at
    the same horizon, using only turns already printed when the forecast is
    made. It needs 12 earlier errors. A conservative variant uses the
    finite-sample level ceil((n + 1) tau) / n instead of tau; with 12 to 50
    errors that pushes the 95th percentile shift to the largest error seen,
    so it is scored for comparison but not used. Raw and calibrated scores, by type and horizon, go to
    data/processed/turn_quantile_horizons.csv; the forecasts to
    turn_quantiles_horizon.csv; the full-sample coefficients and the current
    calibration shift for each horizon to turn_quantile_coefs_horizon.csv.
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


def final_coefs(t: pd.DataFrame) -> pd.DataFrame:
    """Coefficients of the fit on all month-end and quarter-end turns, mapped
    back from standardised to raw units: q = intercept + sum(coef * x)."""
    pool = t[t["type"] != "year-end"].dropna(subset=STATE)
    X = pool[FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tau, col in zip(TAUS, QCOLS):
            m = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs").fit((X - mu) / sd, pool["y"])
            raw = m.coef_ / sd
            rows.append({"quantile": col, "intercept": float(m.intercept_ - np.sum(raw * mu)),
                         **{f: float(c) for f, c in zip(FEATURES, raw)},
                         "n_turns": len(pool), "fit_through": pool["date"].max().date()})
    return pd.DataFrame(rows)


def predict_coefs(coefs: pd.DataFrame, x: dict) -> list[float]:
    """Evaluate exported coefficients for one turn; no scikit-learn needed."""
    q = [float(r["intercept"] + sum(r[f] * x[f] for f in FEATURES)) for _, r in coefs.iterrows()]
    return list(np.maximum.accumulate(q))


HORIZONS = (1, 5, 10, 15)
MIN_CAL = 12


def horizon_turns(t: pd.DataFrame, daily: pd.DataFrame, h: int) -> pd.DataFrame:
    """Month- and quarter-end turns with the state known h settlement days
    before: the build_dataset.py features of row (turn - h + 1), which use
    data up to the end of row (turn - h). made_row is the row the forecast is
    made on; made_date its date."""
    d = daily.sort_values("date").reset_index(drop=True)
    pool = t[t["type"] != "year-end"].copy()
    src = pool["row"] - h + 1
    pool = pool[src >= 0]
    src = src[src >= 0]
    pool[STATE] = d.loc[src, STATE].to_numpy()
    pool["made_row"] = pool["row"] - h
    pool["made_date"] = d.loc[pool["made_row"].clip(lower=0), "date"].to_numpy()
    return pool.dropna(subset=STATE)


def walk_forward_horizon(pool: pd.DataFrame, h: int) -> pd.DataFrame:
    rows = []
    for _, r in pool.iterrows():
        train = pool[pool["row"] <= r["made_row"]]       # printed by the forecast date
        if len(train) >= MIN_TRAIN:
            q = fit_quantiles(train, r.to_frame().T)
            rows.append({"date": r["date"], "type": r["type"], "horizon": h,
                         "made_date": r["made_date"], "y": r["y"],
                         **dict(zip(QCOLS, q))})
    return pd.DataFrame(rows)


def conformal_level(n: int, tau: float) -> float:
    return min(1.0, np.ceil((n + 1) * tau) / n)


def calibrate(fc: pd.DataFrame, finite_sample: bool = False, suffix: str = "_cal") -> pd.DataFrame:
    """Shift each forecast by the quantile of errors on turns printed before it."""
    fc = fc.sort_values("date").reset_index(drop=True)
    out = fc.copy()
    for c in QCOLS:
        out[c + suffix] = np.nan
    for i, r in fc.iterrows():
        past = fc[fc["date"] <= r["made_date"]]
        if len(past) < MIN_CAL:
            continue
        for c, tau in zip(QCOLS, TAUS):
            e = (past["y"] - past[c]).to_numpy()
            level = conformal_level(len(e), tau) if finite_sample else tau
            out.at[i, c + suffix] = r[c] + np.quantile(e, level)
    cal = [c + suffix for c in QCOLS]
    ok = out[cal].notna().all(axis=1)
    out.loc[ok, cal] = np.maximum.accumulate(out.loc[ok, cal].to_numpy(), axis=1)
    return out


def score_horizons(all_fc: pd.DataFrame) -> pd.DataFrame:
    """Raw and calibrated scores on the turns every horizon has calibrated."""
    have = all_fc.dropna(subset=["q50_cal"])
    common = set.intersection(*(set(g["date"]) for _, g in have.groupby("horizon")))
    f = have[have["date"].isin(common)]
    rows = []
    for typ in ("month-end", "quarter-end", "all"):
        for h, g in f.groupby("horizon"):
            g = g if typ == "all" else g[g["type"] == typ]
            for version, suffix in (("raw", ""), ("calibrated", "_cal"),
                                    ("calibrated, finite-sample", "_fs")):
                row = {"type": typ, "horizon": h, "version": version, "n": len(g),
                       "from": g["date"].min().date(), "to": g["date"].max().date()}
                for c, tau in zip(QCOLS, TAUS):
                    q = g[c + suffix].to_numpy()
                    row[f"pinball_{c}"] = pinball(g["y"].to_numpy(), q, tau)
                    row[f"cover_{c}"] = float((g["y"] <= q).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def horizon_coefs(t: pd.DataFrame, daily: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Full-sample coefficients per horizon, plus today's calibration shift:
    the conformal quantile of every out-of-sample error at that horizon."""
    out = []
    for h in HORIZONS:
        pool = horizon_turns(t, daily, h)
        c = final_coefs(pool.assign(type=pool["type"]))
        g = cal[cal["horizon"] == h]
        for i, (col, tau) in enumerate(zip(QCOLS, TAUS)):
            e = (g["y"] - g[col]).to_numpy()
            c.loc[i, "shift"] = float(np.quantile(e, tau))
            c.loc[i, "n_errors"] = len(e)
        c.insert(0, "horizon", h)
        out.append(c)
    return pd.concat(out, ignore_index=True)


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
    final_coefs(t).to_csv(PROC / "turn_quantile_coefs.csv", index=False)

    hz = pd.concat([calibrate(calibrate(walk_forward_horizon(horizon_turns(t, daily, h), h)),
                              finite_sample=True, suffix="_fs")
                    for h in HORIZONS], ignore_index=True)
    hz.to_csv(PROC / "turn_quantiles_horizon.csv", index=False)
    hs = score_horizons(hz)
    hs.to_csv(PROC / "turn_quantile_horizons.csv", index=False)
    horizon_coefs(t, daily, hz).to_csv(PROC / "turn_quantile_coefs_horizon.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(sc.round(3).to_string(index=False))
    ye = t[t["type"] == "year-end"]
    print(f"\nyear-ends: n = {len(ye)} ({', '.join(str(d.year) for d in ye['date'])}); no model")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\nby horizon, raw and calibrated:")
        print(hs.round(3).to_string(index=False))
    print(f"wrote {PROC / 'turn_quantiles.csv'} and turn_quantile_scores.csv, "
          "turn_quantiles_horizon.csv, turn_quantile_horizons.csv, "
          "turn_quantile_coefs_horizon.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
