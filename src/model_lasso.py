"""Walk-forward LASSO on the fast component of the repo spread.

Run from the project root, after build_dataset.py:
    python src/model_lasso.py

Target
    dev_w_bp at t+1: tomorrow's spread minus its own trailing 60-day median,
    winsorised at a fixed +/-25bp.

Feature timing, which decides whether any of this means anything
    Calendar flags are taken at t+1. They are deterministic and known in
    advance, so lagging them would discard information a desk really has.
    Everything measured from the market or the Fed's balance sheet is taken at
    t or earlier. The script asserts no training window overlaps the block it
    predicts, and that standardisation never sees the test block.

Every deviation feature is winsorised at the same fixed cap as the target.
Without that, 18 September 2019 enters the design matrix at 309bp against a
test-window standard deviation of 3.4bp, and one day in the training window
sets the scale for every coefficient.

Benchmarks, each refitted on the training window only
    zero        predict no deviation
    naive       predict today's deviation unchanged
    AR(1)/AR(2) least squares on today's deviation, and on the last two
Zero is a weak benchmark here and should not be quoted on its own. The trailing
median removes the level of the spread but not a slow trend in it, so the
deviation stays strongly autocorrelated: in the test window the correlation
between today's deviation and tomorrow's is 0.81. An autoregression therefore
explains most of the target by itself, and the only question worth asking is
whether the calendar and flow variables add anything on top of it. R2 against
AR(1) is the headline number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent.parent
if "--root" in sys.argv:
    ROOT = Path(sys.argv[sys.argv.index("--root") + 1])
OUT = ROOT / "data" / "processed"

TRAIN_MIN = 500
STEP = 60
WINDOW = 500  # rolling training window; None trains on all history
ALPHAS = np.logspace(-4, 1, 80)
WINSOR_BP = 25.0  # must match build_dataset.py

AHEAD = ["month_end", "quarter_end", "year_end", "mid_settle", "month_start",
         "rate_change", "rate_change_lag1"]
STATE = ["disp_bp", "iqr_bp", "log_volume", "rrp_bn_lag1", "d_rrp_bn_lag1",
         "tga_bn_lag1", "d_tga_bn_lag1"]
AR = ["dev_w", "dev_w_lag1", "dev_w_lag2"]


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    d = df.sort_values("date").reset_index(drop=True)
    cap = lambda s: s.clip(-WINSOR_BP, WINSOR_BP)

    X = pd.DataFrame(index=d.index)
    for c in AHEAD:
        X[c] = d[c].shift(-1)
    for c in STATE:
        X[c] = cap(d[c]) if c.endswith("_bp") else d[c]
    X["dev_w"] = cap(d["dev_bp"])
    X["dev_w_lag1"] = cap(d["dev_bp"]).shift(1)
    X["dev_w_lag2"] = cap(d["dev_bp"]).shift(2)

    y = d["dev_w_bp"].shift(-1)
    when = d["date"].shift(-1)
    keep = X.notna().all(axis=1) & y.notna() & when.notna()
    return (X[keep].reset_index(drop=True), y[keep].reset_index(drop=True),
            when[keep].reset_index(drop=True))


def walk_forward(X: pd.DataFrame, y: pd.Series, when: pd.Series, window: int | None = None):
    """window=None trains on everything so far; an int keeps only the last N days.

    The choice matters because the calendar effects are regime-dependent: the
    month-end premium is about 9 to 13bp in 2018-19 and 2025, and essentially
    zero from 2020 to 2022. A model that remembers every regime equally
    averages them and fits none.
    """
    rows, picks, picks2_all = [], [], []
    ar1, ar2 = [AR[0]], AR[:2]
    start = TRAIN_MIN
    while start < len(X):
        stop = min(start + STEP, len(X))
        lo = 0 if window is None else max(0, start - window)
        tr, te = slice(lo, start), slice(start, stop)
        assert when[te].min() > when[tr].max(), "train window overlaps test block"

        Xtr, Xte = X.iloc[tr].to_numpy(float), X.iloc[te].to_numpy(float)
        ytr = y.iloc[tr].to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

        cv = LassoCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5),
                     max_iter=50000).fit(Ztr, ytr)
        model = Lasso(alpha=cv.alpha_, max_iter=50000).fit(Ztr, ytr)

        b1 = LinearRegression().fit(X.iloc[tr][ar1], ytr)
        b2 = LinearRegression().fit(X.iloc[tr][ar2], ytr)

        # Two-stage: take the autoregression out of the penalty, then ask the
        # LASSO whether the calendar and flow variables explain what is left.
        # With one shared penalty the strong AR term is shrunk to control the
        # weak ones, which costs more than the weak ones are worth.
        other = [c for c in X.columns if c not in AR]
        resid = ytr - b1.predict(X.iloc[tr][ar1])
        Otr, Ote = X.iloc[tr][other].to_numpy(float), X.iloc[te][other].to_numpy(float)
        omu, osd = Otr.mean(0), Otr.std(0)
        osd[osd == 0] = 1.0
        cv2 = LassoCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5),
                      max_iter=50000).fit((Otr - omu) / osd, resid)
        st2 = Lasso(alpha=cv2.alpha_, max_iter=50000).fit((Otr - omu) / osd, resid)
        picks2 = pd.Series(st2.coef_, index=other)

        rows.append(pd.DataFrame({
            "date": when.iloc[te].to_numpy(),
            "y": y.iloc[te].to_numpy(float),
            "pred": model.predict(Zte),
            "naive": X.iloc[te]["dev_w"].to_numpy(float),
            "ar1": b1.predict(X.iloc[te][ar1]),
            "ar2": b2.predict(X.iloc[te][ar2]),
            "two_stage": b1.predict(X.iloc[te][ar1]) + st2.predict((Ote - omu) / osd),
            "alpha": cv.alpha_,
            "alpha2": cv2.alpha_,
        }))
        picks.append(pd.Series(model.coef_, index=X.columns))
        picks2_all.append(picks2)
        start = stop
    return pd.concat(rows, ignore_index=True), pd.DataFrame(picks), pd.DataFrame(picks2_all)


def r2(y, p, base) -> float:
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.asarray(base)) ** 2)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main() -> int:
    df = pd.read_csv(OUT / "daily.csv", parse_dates=["date"])
    X, y, when = build(df)
    print(f"rows {len(X)}, features {X.shape[1]}, {when.min().date()} -> {when.max().date()}")
    print(f"feature |dev_w| max {X['dev_w'].abs().max():.1f}bp (cap {WINSOR_BP:.0f})")

    exp_res, _, _ = walk_forward(X, y, when, window=None)
    res, coefs, coefs2 = walk_forward(X, y, when, window=WINDOW)
    res["year"] = res["date"].dt.year
    yv = res["y"].to_numpy()
    zero = np.zeros_like(yv)

    print(f"\ntraining window: rolling {WINDOW} days (expanding shown for comparison)")
    for lab, r in (("expanding", exp_res), (f"rolling {WINDOW}", res)):
        rec = r[r["date"].dt.year >= 2024]
        print(f"  {lab:<16} two-stage R2 vs AR(1): all {r2(r['y'], r['two_stage'], r['ar1']):+.4f}"
              f"   2024+ {r2(rec['y'], rec['two_stage'], rec['ar1']):+.4f}")

    print(f"\nout-of-sample, {len(res)} forecasts "
          f"({res['date'].min().date()} -> {res['date'].max().date()})")
    print(f"  {'RMSE (bp)':<18}{'vs zero':>10}")
    for name, p in (("zero", zero), ("naive", res["naive"]), ("AR(1)", res["ar1"]),
                    ("AR(2)", res["ar2"]), ("LASSO one-stage", res["pred"]),
                    ("LASSO two-stage", res["two_stage"])):
        print(f"  {name:<18}{rmse(yv, p):>7.3f}   R2 {r2(yv, p, zero):+.4f}")

    print("\nheadline: does anything add to the autoregression?")
    for name, p in (("one-stage", res["pred"]), ("two-stage", res["two_stage"])):
        print(f"  {name:<10} R2 vs AR(1) {r2(yv, p, res['ar1']):+.4f}"
              f"   vs AR(2) {r2(yv, p, res['ar2']):+.4f}"
              f"   vs naive {r2(yv, p, res['naive']):+.4f}")

    print("\nby year, R2 against AR(1):")
    for yr, g in res.groupby("year"):
        print(f"  {yr}  n={len(g):3d}  one-stage {r2(g['y'], g['pred'], g['ar1']):+.4f}"
              f"   two-stage {r2(g['y'], g['two_stage'], g['ar1']):+.4f}"
              f"   RMSE ar1 {rmse(g['y'], g['ar1']):6.3f}")

    print(f"\nfeature selection across {len(coefs)} refits:")
    sel = pd.DataFrame({"picked_pct": 100 * (coefs != 0).mean(),
                        "mean_coef": coefs.mean(),
                        "mean_abs": coefs.abs().mean()})
    sel = sel[sel["picked_pct"] > 0].sort_values("mean_abs", ascending=False).round(3)
    print(sel.to_string() if len(sel) else "  nothing selected in any refit")
    dropped = [c for c in coefs.columns if (coefs[c] == 0).all()]
    print(f"\nnever selected ({len(dropped)}): {', '.join(dropped) if dropped else 'none'}")

    print(f"\ntwo-stage: what explains the AR residual, across {len(coefs2)} refits:")
    sel2 = pd.DataFrame({"picked_pct": 100 * (coefs2 != 0).mean(),
                         "mean_coef": coefs2.mean(),
                         "mean_abs": coefs2.abs().mean()})
    sel2 = sel2[sel2["picked_pct"] > 0].sort_values("mean_abs", ascending=False).round(3)
    print(sel2.to_string() if len(sel2) else "  nothing selected in any refit")
    print(f"\nalpha one-stage: median {res['alpha'].median():.4f}; "
          f"two-stage: median {res['alpha2'].median():.4f}")

    res.to_csv(OUT / "oos_predictions.csv", index=False)
    print(f"\nwrote {OUT / 'oos_predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
