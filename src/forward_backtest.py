"""Backtest of the forward path: the deviation forecast h settlement days ahead.

Run from the project root, after model_lasso.py:
    python src/forward_backtest.py

For every out-of-sample forecast date t and h in 1, 2, 5, 10, 15, 20:
  forecast  the AR(1) term iterated h times from today's deviation, plus the
            second-stage LASSO evaluated with the calendar flags of the target
            day and the scarcity state held at its value on t. Both stages
            are the ones fitted for t's walk-forward block (rolling 500 days,
            refit every 60), so nothing after t is used. At h = 1 this is
            exactly the two-stage forecast in oos_predictions.csv, which the
            script asserts.
  benchmark AR(1) alone, iterated to the same horizon.
  target    SOFR minus the top of the range on the target day, minus the base
            held at t (the 60-day median the app uses), winsorised at
            +/-25 bp like the model's own target. At h = 1 it is the model's
            target exactly.
  interval  80%: the 10th and 90th percentile of the last 250 errors at the
            same horizon whose target day is on or before t.

The target day's rate_change flag is the realised one, as in model_lasso.py;
the app reads it from the FOMC table instead.

Scores by horizon, for turn days (month_end flag on the target day) and other
days: n, RMSE of the model and of iterated AR(1), R² against AR(1), and 80%
coverage. Writes data/processed/forward_backtest.csv (scores) and
forward_paths.csv (one row per forecast date and horizon).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
import model_lasso as m  # noqa: E402

HORIZONS = (1, 2, 5, 10, 15, 20)
ERR_WINDOW = 250
COVER_BAND = (0.70, 0.90)


def fit_block(Xtr: pd.DataFrame, ytr: np.ndarray):
    """Both stages of the two-stage model, as model_lasso.walk_forward fits them."""
    ar1 = [m.AR[0]]
    b1 = LinearRegression().fit(Xtr[ar1], ytr)
    other = [c for c in Xtr.columns if c not in m.AR]
    resid = ytr - b1.predict(Xtr[ar1])
    O = Xtr[other].to_numpy(float)
    mu, sd = O.mean(0), O.std(0)
    sd[sd == 0] = 1.0
    cv = LassoCV(alphas=m.ALPHAS, cv=TimeSeriesSplit(n_splits=5), max_iter=50000).fit((O - mu) / sd, resid)
    st2 = Lasso(alpha=cv.alpha_, max_iter=50000).fit((O - mu) / sd, resid)
    return float(b1.intercept_), float(b1.coef_[0]), st2, mu, sd, other


def paths(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values("date").reset_index(drop=True)
    X, y, when = m.build(d)
    # X row i is as of SOFR date asof[i], with target when[i] = the next date.
    pos = {dt: k for k, dt in enumerate(d["date"])}
    asof_dates = when.map(lambda w: d.at[pos[w] - 1, "date"])
    row_of = {w: k for k, w in enumerate(when)}
    spread = d.set_index("date")["spread_hi_bp"]
    base_next = d.set_index("date")["base_bp"]          # base at t+1: median up to t
    month_end = d.set_index("date")["month_end"]

    rows = []
    start = m.TRAIN_MIN
    while start < len(X):
        stop = min(start + m.STEP, len(X))
        lo = max(0, start - m.WINDOW)
        c, phi, st2, mu, sd, other = fit_block(X.iloc[lo:start], y.iloc[lo:start].to_numpy(float))
        for i in range(start, stop):
            t = asof_dates.iloc[i]
            base = float(base_next[when.iloc[i]])
            dev0 = float(X.at[i, "dev_w"])
            for h in HORIZONS:
                # The target is h settlement days after t in the daily table.
                # build() drops one mid-sample date (6 August 2021), so X rows
                # are not consecutive days; look the target up by date, and
                # skip a target that has no feature row of its own.
                k = pos[t] + h
                if k >= len(d) or d.at[k, "date"] not in row_of:
                    continue
                target = d.at[k, "date"]
                j = row_of[target]                 # X row whose target is that day
                xh = X.loc[[i], other].copy()
                xh[m.AHEAD] = X.loc[j, m.AHEAD].to_numpy()   # the target day's calendar
                ar = dev0
                for _ in range(h):
                    ar = c + phi * ar
                f = ar + float(st2.predict((xh.to_numpy(float) - mu) / sd)[0])
                actual = float(np.clip(spread[target] - base, -m.WINSOR_BP, m.WINSOR_BP))
                rows.append({"asof": t, "target": target, "h": h, "turn": int(month_end[target] == 1),
                             "y": actual, "f": f, "ar1": ar, "base_bp": base})
        start = stop
    return pd.DataFrame(rows)


def add_intervals(p: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h, g in p.groupby("h"):
        g = g.sort_values(["asof", "target"]).reset_index(drop=True)
        e = (g["y"] - g["f"]).to_numpy()
        tgt = g["target"].to_numpy()
        lo = np.full(len(g), np.nan)
        hi = np.full(len(g), np.nan)
        for k, t in enumerate(g["asof"]):
            known = e[tgt <= t]                    # errors already observable at t
            if len(known) >= ERR_WINDOW:
                q10, q90 = np.quantile(known[-ERR_WINDOW:], (0.10, 0.90))
                lo[k], hi[k] = g.at[k, "f"] + q10, g.at[k, "f"] + q90
        out.append(g.assign(lo80=lo, hi80=hi))
    return pd.concat(out, ignore_index=True)


def scores(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in p.groupby("h"):
        for label, s in (("all", g), ("turn days", g[g["turn"] == 1]), ("other days", g[g["turn"] == 0])):
            iv = s.dropna(subset=["lo80"])
            rmse = float(np.sqrt(np.mean((s["y"] - s["f"]) ** 2)))
            rmse_ar = float(np.sqrt(np.mean((s["y"] - s["ar1"]) ** 2)))
            rows.append({"h": h, "days": label, "n": len(s),
                         "rmse_model_bp": rmse, "rmse_ar1_bp": rmse_ar,
                         "r2_vs_ar1": 1 - rmse ** 2 / rmse_ar ** 2,
                         "n_interval": len(iv),
                         "coverage80": float(((iv["y"] >= iv["lo80"]) & (iv["y"] <= iv["hi80"])).mean()),
                         "width80_bp": float((iv["hi80"] - iv["lo80"]).mean())})
    return pd.DataFrame(rows)


def tested_horizon(sc: pd.DataFrame, days: str = "other days") -> int:
    """Longest horizon h such that coverage at h and every shorter horizon
    stays within the 70-90% band."""
    ok = 0
    for h in HORIZONS:
        c = sc[(sc["h"] == h) & (sc["days"] == days)]["coverage80"].iloc[0]
        if COVER_BAND[0] <= c <= COVER_BAND[1]:
            ok = h
        else:
            break
    return ok


def main() -> int:
    daily = pd.read_csv(PROC / "daily.csv", parse_dates=["date"])
    oos = pd.read_csv(PROC / "oos_predictions.csv", parse_dates=["date"])
    p = add_intervals(paths(daily))
    # At h = 1 the path is the walk-forward two-stage forecast itself.
    h1 = p[p["h"] == 1].merge(oos[["date", "two_stage", "y"]], left_on="target", right_on="date")
    assert np.allclose(h1["f"], h1["two_stage"], atol=1e-8), "h=1 path differs from two_stage"
    assert np.allclose(h1["y_x"], h1["y_y"]), "h=1 target differs from the model's target"
    p.to_csv(PROC / "forward_paths.csv", index=False)
    sc = scores(p)
    sc.to_csv(PROC / "forward_backtest.csv", index=False)
    with pd.option_context("display.width", 200):
        print(sc.round(3).to_string(index=False))
    print(f"\nlongest tested horizon (other days, coverage within 70-90% at every "
          f"horizon up to it): {tested_horizon(sc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
