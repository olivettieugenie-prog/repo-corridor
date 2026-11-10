"""Live forecast of the next business day's deviation, with its 80% interval.

Run from the project root:
    python src/forecast_next.py            refresh the data, forecast, append a row
    python src/forecast_next.py --no-fetch reuse the data already in the work dir

What it does, in order
  1. Downloads every required series again with fetch_data.py and rebuilds the
     daily table with build_dataset.py, both into a work directory (.live_work/)
     so the committed files in data/raw and data/processed are never touched.
     Set REPO_SKIP_FRED=1 to skip the optional FRED series, as CI does.
  2. Fits the same two-stage model as model_lasso.py (AR(1) first, then a LASSO
     on its residual) on the latest 500 rows, and forecasts the deviation for
     the next business day after the latest published SOFR date.
  3. Bounds it with the 10th and 90th percentiles of the last 250 walk-forward
     errors, as src/intervals.py does; every one of those errors is for a
     target date already published.
  4. Appends one row to data/live/forecast.csv and fills in the realised value
     of every earlier row whose target date has since been published.

Timing: what exists when the job runs
  The workflow runs at 14:37 UTC on weekdays (backup 15:47), 9:37 a.m. ET in
  winter and 10:37 a.m. ET in summer. At that hour, on business day D:
    SOFR for D-1   published about 8:00 a.m. ET on D (revisable about 2:30 p.m.)
    EFFR for D-1   published about 9:00 a.m. ET on D, with the target range
    ON RRP for D-1 operation closes 1:15 p.m. ET on D-1, results the same day
    TGA for D-2    Daily Treasury Statement for D-1 is out only at 4:00 p.m. ET on D
  The model uses SOFR, EFFR and the target range at t = D-1, ON RRP and TGA at
  t-1 = D-2 (build_dataset.py lags both strictly), and calendar flags at D. So
  every input exists at run time, and the target D is the run day itself.

The target range for D is the range at D-1 unless data/fed_administered_rates.csv
holds a note effective on or before D. If D is the first business day after a
scheduled FOMC decision and no such note exists, the row is marked
range_unconfirmed = 1: the decision is public by then but not yet in the table.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import forward_backtest as fb  # noqa: E402
import model_lasso as m  # noqa: E402
from build_dataset import BASE_MIN, BASE_WIN, calendar_flags  # noqa: E402
from intervals import error_quantiles  # noqa: E402
from market_calendar import business_days, fomc_effective_day, next_business_day  # noqa: E402

WORK = ROOT / ".live_work"
LIVE = ROOT / "data" / "live" / "forecast.csv"
FORWARD = ROOT / "data" / "live" / "forward.csv"
FORWARD_DAYS = 20
NOTES = ROOT / "data" / "fed_administered_rates.csv"
FLAGS = ["month_end", "quarter_end", "year_end", "mid_settle", "month_start"]

COLUMNS = [
    "run_time_utc", "as_of_date", "target_date",
    "forecast_bp", "lower_bp", "upper_bp", "ar1_bp",
    *FLAGS, "rate_change", "range_unconfirmed",
    "target_lo", "target_hi", "base_bp", "implied_sofr_pct",
    "implied_sofr_lo_pct", "implied_sofr_hi_pct",
    "rrp_date", "rrp_bn", "d_rrp_bn", "tga_date", "tga_bn", "d_tga_bn",
    "realised_bp", "realised_sofr_pct", "hit", "scored_at_utc",
]


# ------------------------------------------------------------------ data
def refresh(work: Path, fetch: bool = True) -> pd.DataFrame:
    (work / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(NOTES, work / "data" / NOTES.name)
    py = sys.executable
    if fetch:
        subprocess.run([py, str(ROOT / "src" / "fetch_data.py"), "--refresh",
                        "--root", str(work)], check=True)
    subprocess.run([py, str(ROOT / "src" / "build_dataset.py"), "--root", str(work)],
                   check=True, stdout=subprocess.DEVNULL)
    return pd.read_csv(work / "data" / "processed" / "daily.csv", parse_dates=["date"])


def latest_level(path: Path, col: str) -> tuple[str, float, float]:
    s = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return (s["date"].iloc[-1].date().isoformat(), float(s[col].iloc[-1]),
            float(s[col].iloc[-1] - s[col].iloc[-2]))


# ------------------------------------------------------------- features
def target_flags(dates: pd.Series, target: pd.Timestamp) -> dict:
    """Calendar flags for `target`, defined exactly as build_dataset.py defines
    them: the future business days are added so the month can be completed."""
    future = business_days(target, target + pd.Timedelta(days=70))
    alld = pd.Series(pd.DatetimeIndex(dates).union(future))
    f = calendar_flags(alld).set_index("date").loc[target]
    return {c: int(f[c]) for c in FLAGS}


def target_range(d: pd.DataFrame, target: pd.Timestamp) -> tuple[float, float, int, int]:
    last = d.iloc[-1]
    lo, hi = float(last["target_lo"]), float(last["target_hi"])
    notes = pd.read_csv(NOTES, parse_dates=["effective_date"])
    new = notes[(notes["effective_date"] > last["date"]) & (notes["effective_date"] <= target)]
    if len(new):
        lo, hi = float(new.iloc[-1]["target_lo"]), float(new.iloc[-1]["target_hi"])
    change = int(abs(hi - last["target_hi"]) > 1e-9 or abs(lo - last["target_lo"]) > 1e-9)
    unconfirmed = int(fomc_effective_day(target) and new.empty)
    return lo, hi, change, unconfirmed


def feature_row(d: pd.DataFrame, target: pd.Timestamp, rate_change: int) -> pd.DataFrame:
    """The X row model_lasso.build() would produce for as-of date d[-1] and
    target `target`, built without needing the target's own data."""
    cap = lambda v: float(np.clip(v, -m.WINSOR_BP, m.WINSOR_BP))
    last = d.iloc[-1]
    row = target_flags(d["date"], target)
    row["rate_change"] = rate_change
    row["rate_change_lag1"] = int(last["rate_change"])
    for c in m.STATE:
        row[c] = cap(last[c]) if c.endswith("_bp") else float(last[c])
    row["dev_w"] = cap(last["dev_bp"])
    row["dev_w_lag1"] = cap(d["dev_bp"].iloc[-2])
    row["dev_w_lag2"] = cap(d["dev_bp"].iloc[-3])
    cols = m.AHEAD + m.STATE + m.AR
    return pd.DataFrame([row])[cols]


# ---------------------------------------------------------------- model
def fit_predict(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Two-stage forecast and its AR(1) first stage, as in model_lasso.walk_forward."""
    ar1 = [m.AR[0]]
    b1 = LinearRegression().fit(Xtr[ar1], ytr)
    other = [c for c in Xtr.columns if c not in m.AR]
    resid = ytr - b1.predict(Xtr[ar1])
    Otr, Ote = Xtr[other].to_numpy(float), Xte[other].to_numpy(float)
    omu, osd = Otr.mean(0), Otr.std(0)
    osd[osd == 0] = 1.0
    cv2 = LassoCV(alphas=m.ALPHAS, cv=TimeSeriesSplit(n_splits=5),
                  max_iter=50000).fit((Otr - omu) / osd, resid)
    st2 = Lasso(alpha=cv2.alpha_, max_iter=50000).fit((Otr - omu) / osd, resid)
    a = b1.predict(Xte[ar1])
    return a + st2.predict((Ote - omu) / osd), a


def implied_sofr(d: pd.DataFrame, hi: float, dev_bp: float) -> tuple[float, float]:
    """SOFR = top of range + (trailing median + deviation) / 100, which inverts
    build_dataset.py's definitions exactly. Returns (sofr %, base bp)."""
    spread = d["spread_hi_bp"].iloc[-BASE_WIN:]
    base = float(spread.median()) if spread.notna().sum() >= BASE_MIN else np.nan
    return hi + (base + dev_bp) / 100, base


# ---------------------------------------------------------------- score
def score(live: pd.DataFrame, d: pd.DataFrame, now: str) -> pd.DataFrame:
    if live.empty:
        return live
    for c in ("realised_bp", "realised_sofr_pct", "hit", "scored_at_utc"):
        if c not in live.columns:
            live[c] = np.nan
    live["scored_at_utc"] = live["scored_at_utc"].astype(object)
    known =d.set_index("date")[["dev_w_bp", "sofr"]]
    for i, r in live.iterrows():
        t = pd.Timestamp(r["target_date"])
        if pd.notna(r.get("realised_bp")) or t not in known.index:
            continue
        y = round(float(known.at[t, "dev_w_bp"]), 6)
        live.at[i, "realised_bp"] = y
        live.at[i, "realised_sofr_pct"] = float(known.at[t, "sofr"])
        live.at[i, "hit"] = int(r["lower_bp"] <= y <= r["upper_bp"]) if pd.notna(r["lower_bp"]) else np.nan
        live.at[i, "scored_at_utc"] = now
    return live


# ---------------------------------------------------------- forward path
def range_path(d: pd.DataFrame, dates: list[pd.Timestamp]) -> list[dict]:
    """Target range and rate-change flags for each future date, from the
    implementation-note table (the last range carried when no note covers it)."""
    out, prev_hi, prev_lo = [], float(d["target_hi"].iloc[-1]), float(d["target_lo"].iloc[-1])
    prev_change = int(d["rate_change"].iloc[-1])
    for t in dates:
        lo, hi, _, unconfirmed = target_range(d, t)
        change = int(abs(hi - prev_hi) > 1e-9 or abs(lo - prev_lo) > 1e-9)
        out.append({"target_lo": lo, "target_hi": hi, "rate_change": change,
                    "rate_change_lag1": prev_change, "range_unconfirmed": unconfirmed})
        prev_lo, prev_hi, prev_change = lo, hi, change
    return out


def forward_path(d: pd.DataFrame, X: pd.DataFrame, y: pd.Series, days: int = FORWARD_DAYS,
                 history: pd.DataFrame | None = None) -> pd.DataFrame:
    """The next `days` settlement days, forecast as forward_backtest.py tests it:
    AR(1) iterated from today's deviation plus the second stage at each day's
    calendar flags, scarcity held at today's value, base held at today's
    60-day median. The 80% range at horizon h comes from the last 250
    walk-forward errors at that same horizon."""
    as_of = d["date"].iloc[-1]
    c, phi, st2, mu, sd, other = fb.fit_block(X.iloc[-m.WINDOW:], y.iloc[-m.WINDOW:].to_numpy(float))
    dates, t = [], as_of
    for _ in range(days):
        t = next_business_day(t)
        dates.append(t)
    rng = range_path(d, dates)
    dev0 = float(np.clip(d["dev_bp"].iloc[-1], -m.WINSOR_BP, m.WINSOR_BP))
    _, base = implied_sofr(d, float(d["target_hi"].iloc[-1]), 0.0)

    hist = history if history is not None else fb.paths(d, horizons=range(1, days + 1))
    sc = pd.read_csv(ROOT / "data" / "processed" / "forward_backtest.csv")
    tested = fb.tested_horizon(sc)
    cover = sc.set_index(["h", "days"])["coverage80"]

    rows, ar = [], dev0
    for h, (t, r) in enumerate(zip(dates, rng), start=1):
        ar = c + phi * ar
        x = feature_row(d, t, r["rate_change"])
        x["rate_change_lag1"] = r["rate_change_lag1"]
        f = ar + float(st2.predict((x[other].to_numpy(float) - mu) / sd)[0])
        e = hist[hist["h"] == h].sort_values("target")
        e = (e["y"] - e["f"]).to_numpy()[-250:]
        q10, q90 = (np.quantile(e, (0.10, 0.90)) if len(e) >= 50 else (np.nan, np.nan))
        flags = x[FLAGS].iloc[0]
        kind = ("year-end" if flags["year_end"] else "quarter-end" if flags["quarter_end"]
                else "month-end" if flags["month_end"] else "")
        hs = fb.nearest_scored(h)
        rows.append({
            "h": h, "date": t.date().isoformat(), "kind": kind,
            "forecast_bp": f, "lo80_bp": f + q10, "hi80_bp": f + q90, "p90_bp": f + q90,
            "base_bp": base, "target_hi": r["target_hi"],
            "implied_sofr_pct": r["target_hi"] + (base + f) / 100,
            "tested": int(h <= tested), "tested_horizon": tested, "scored_h": hs,
            "cov80_other": float(cover[(hs, "other days")]),
            "cov80_turn": float(cover[(hs, "turn days")]),
            "range_unconfirmed": r["range_unconfirmed"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    fetch = "--no-fetch" not in sys.argv
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    d = refresh(WORK, fetch=fetch).sort_values("date").reset_index(drop=True)
    as_of = d["date"].iloc[-1]
    target = next_business_day(as_of)

    X, y, when = m.build(d)
    assert when.max() <= as_of
    Xtr, ytr = X.iloc[-m.WINDOW:], y.iloc[-m.WINDOW:].to_numpy(float)
    lo, hi, change, unconfirmed = target_range(d, target)
    x = feature_row(d, target, change)
    f, a = (float(v[0]) for v in fit_predict(Xtr, ytr, x))

    res, _, _ = m.walk_forward(X, y, when, window=m.WINDOW)
    ql, qh = error_quantiles((res["y"] - res["two_stage"]).to_numpy())
    sofr_f, base = implied_sofr(d, hi, f)

    rrp_date, rrp, drrp = latest_level(WORK / "data" / "raw" / "rrp.csv", "rrp_bn")
    tga_date, tga, dtga = latest_level(WORK / "data" / "raw" / "tga.csv", "tga_bn")
    row = {
        "run_time_utc": now, "as_of_date": as_of.date().isoformat(),
        "target_date": target.date().isoformat(),
        "forecast_bp": f, "lower_bp": f + ql, "upper_bp": f + qh, "ar1_bp": a,
        **x[FLAGS].iloc[0].astype(int).to_dict(),
        "rate_change": change, "range_unconfirmed": unconfirmed,
        "target_lo": lo, "target_hi": hi, "base_bp": base,
        "implied_sofr_pct": sofr_f,
        "implied_sofr_lo_pct": hi + (base + f + ql) / 100,
        "implied_sofr_hi_pct": hi + (base + f + qh) / 100,
        "rrp_date": rrp_date, "rrp_bn": rrp, "d_rrp_bn": drrp,
        "tga_date": tga_date, "tga_bn": tga, "d_tga_bn": dtga,
    }

    row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    live = pd.read_csv(LIVE) if LIVE.exists() else pd.DataFrame(columns=COLUMNS)
    live = score(live, d, now)
    live = pd.concat([live, pd.DataFrame([row])], ignore_index=True)[COLUMNS]
    live.to_csv(LIVE, index=False)

    print(f"as of {as_of.date()} -> target {target.date()}: "
          f"{f:+.2f} bp [{f + ql:+.2f}, {f + qh:+.2f}], AR(1) {a:+.2f}; "
          f"implied SOFR {sofr_f:.3f}%  flags "
          + ", ".join(c for c in FLAGS if row[c]) + (" rate change" if change else "")
          + (" RANGE UNCONFIRMED" if unconfirmed else ""))
    print(f"scored rows: {int(live['realised_bp'].notna().sum())} of {len(live)}; wrote {LIVE}")

    fwd = forward_path(d, X, y)
    assert abs(fwd.at[0, "forecast_bp"] - f) < 1e-8, "day 1 of the path must be the call above"
    fwd.insert(0, "as_of_date", as_of.date().isoformat())
    fwd.insert(0, "run_time_utc", now)
    fwd.round(6).to_csv(FORWARD, index=False)
    print(f"forward path: {len(fwd)} days to {fwd['date'].iloc[-1]}, tested to day "
          f"{int(fwd['tested_horizon'].iloc[0])}; wrote {FORWARD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
