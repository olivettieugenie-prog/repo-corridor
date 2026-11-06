"""Permutation importance, with correlated predictors shuffled together.

Run from the project root, after model_lasso.py:
    python src/importance.py

Method, following session 5 of the course
    Freeze the fitted models, shuffle one predictor in the out-of-sample
    period, and measure how much worse the forecast gets. A predictor the
    model leans on should hurt when scrambled. A predictor it ignores should
    not. Negative importance means the shuffle happened to reduce error in
    this sample, which is noise rather than evidence against the feature.

Two details matter and both come from the deck.

    Shuffle in blocks, not row by row. These predictors are serially
    correlated, and a row-wise shuffle destroys that structure, which makes
    the feature look more important than it is because the model is being fed
    something impossible rather than something merely wrong. Consecutive
    blocks of BLOCK days preserve the local structure.

    Shuffle correlated predictors together. The calendar flags here are not
    merely correlated, they are nested: every quarter-end is a month-end and
    every year-end is a quarter-end. Permuting one at a time lets the others
    stand in for it, so each looks unimportant while the group carries real
    weight. The deck's own example shows two volatility measures scoring +0.2
    and -0.9 bp alone but +3.9 bp together.

Repeated N_SHUFFLE times; the spread of those repeats is reported as the 10th
to 90th percentile, so a number smaller than its own noise is visible as such.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_lasso import AR, ALPHAS, OUT, STEP, TRAIN_MIN, WINDOW, build, rmse  # noqa: E402

BLOCK = 21        # trading days per shuffled block, as in the course example
N_SHUFFLE = 30
SEED = 0

# Groups shuffled jointly. The calendar flags are nested, so individually they
# cover for each other.
GROUPS = {
    "calendar, all five": ["month_end", "quarter_end", "year_end",
                           "month_start", "mid_settle"],
    "month/quarter/year nest": ["month_end", "quarter_end", "year_end"],
    "balance sheet, all four": ["rrp_bn_lag1", "d_rrp_bn_lag1",
                                "tga_bn_lag1", "d_tga_bn_lag1"],
    "dispersion pair": ["disp_bp", "iqr_bp"],
    "rate-change pair": ["rate_change", "rate_change_lag1"],
}


def fit_windows(X: pd.DataFrame, y: pd.Series, when: pd.Series):
    """Walk forward once, keeping each window's fitted pieces for reuse."""
    other = [c for c in X.columns if c not in AR]
    ar1 = [AR[0]]
    windows = []
    start = TRAIN_MIN
    while start < len(X):
        stop = min(start + STEP, len(X))
        lo = max(0, start - WINDOW)
        tr, te = slice(lo, start), slice(start, stop)
        assert when[te].min() > when[tr].max(), "train window overlaps test block"

        ytr = y.iloc[tr].to_numpy(float)
        b1 = LinearRegression().fit(X.iloc[tr][ar1], ytr)
        resid = ytr - b1.predict(X.iloc[tr][ar1])

        Otr = X.iloc[tr][other].to_numpy(float)
        mu, sd = Otr.mean(0), Otr.std(0)
        sd[sd == 0] = 1.0
        cv = LassoCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5),
                     max_iter=100000, tol=1e-3).fit((Otr - mu) / sd, resid)
        m = Lasso(alpha=cv.alpha_, max_iter=100000, tol=1e-3).fit((Otr - mu) / sd, resid)

        windows.append({"te": te, "ar": b1.predict(X.iloc[te][ar1]),
                        "model": m, "mu": mu, "sd": sd})
        start = stop
    return windows, other


def predict_all(windows, feats: np.ndarray) -> np.ndarray:
    """Predictions for the whole out-of-sample period from frozen models."""
    out = []
    for w in windows:
        block = feats[w["te"], :]
        out.append(w["ar"] + w["model"].predict((block - w["mu"]) / w["sd"]))
    return np.concatenate(out)


def block_order(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of 0..n-1 built by shuffling consecutive blocks."""
    blocks = [np.arange(i, min(i + BLOCK, n)) for i in range(0, n, BLOCK)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)


def main() -> int:
    daily = pd.read_csv(OUT / "daily.csv", parse_dates=["date"])
    X, y, when = build(daily)
    windows, other = fit_windows(X, y, when)

    oos = np.concatenate([np.arange(w["te"].start, w["te"].stop) for w in windows])
    feats = X[other].to_numpy(float)
    truth = y.to_numpy(float)[oos]
    base_pred = predict_all(windows, feats)
    base = rmse(truth, base_pred)
    print(f"out-of-sample RMSE {base:.4f} bp over {len(oos)} days "
          f"({when.iloc[oos[0]].date()} -> {when.iloc[oos[-1]].date()})")
    print(f"shuffling in blocks of {BLOCK} days, {N_SHUFFLE} repeats\n")

    col = {c: i for i, c in enumerate(other)}
    targets = [(c, [c]) for c in other] + list(GROUPS.items())
    rows = []
    for name, cols in targets:
        idx = [col[c] for c in cols if c in col]
        if len(idx) != len(cols):
            continue
        rng = np.random.default_rng(SEED)
        deltas = []
        for _ in range(N_SHUFFLE):
            scrambled = feats.copy()
            order = oos[block_order(len(oos), rng)]
            # One shared order for every column in the group, which keeps the
            # columns' relationship to each other intact while breaking their
            # relationship to the target.
            for j in idx:
                scrambled[oos, j] = feats[order, j]
            deltas.append(rmse(truth, predict_all(windows, scrambled)) - base)
        d = np.array(deltas)
        rows.append({"what": name, "n_cols": len(idx), "mean_bp": d.mean(),
                     "p10": np.percentile(d, 10), "p90": np.percentile(d, 90)})

    res = pd.DataFrame(rows).sort_values("mean_bp", ascending=False)
    singles = res[res["n_cols"] == 1]
    groups = res[res["n_cols"] > 1]

    def show(frame: pd.DataFrame, title: str) -> None:
        print(title)
        print(f"  {'':<26}{'RMSE increase':>14}{'p10':>9}{'p90':>9}")
        for _, r in frame.iterrows():
            print(f"  {r['what']:<26}{r['mean_bp']:>+14.4f}{r['p10']:>+9.4f}{r['p90']:>+9.4f}")
        print()

    show(singles, "One predictor at a time (bp of RMSE, higher = more relied on)")
    show(groups, "Correlated predictors shuffled together")

    cal = res.loc[res["what"] == "calendar, all five", "mean_bp"]
    parts = singles[singles["what"].isin(GROUPS["calendar, all five"])]["mean_bp"].sum()
    if len(cal):
        print(f"The five calendar flags together: {cal.iloc[0]:+.4f} bp.")
        print(f"The same five added up one at a time: {parts:+.4f} bp.")
        print("The gap is the nesting: shuffled alone, each flag is covered by")
        print("the others, because every quarter-end is also a month-end.")

    res.to_csv(OUT / "permutation_importance.csv", index=False)
    print(f"\nwrote {OUT / 'permutation_importance.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
