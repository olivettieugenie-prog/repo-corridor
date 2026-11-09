"""Term versus roll: the expected cost of rolling overnight across a window.

Pure functions with pandas and numpy only, so the app can run them without
scikit-learn. The window is [start, end): cash is placed on each settlement
day d with start <= d < end, and each placement accrues interest for the
calendar days to the next settlement day, or to `end` if sooner. A Friday
placement therefore carries the weekend. Both legs are act/360, so the
day-weighted average overnight rate is directly comparable to a term quote.

Rate on each day, as SOFR = top of range + (base + deviation) / 100:
  ordinary day  the two-stage forecast for the first day it covers (the live
                log), then the AR(1) recursion dev' = c + phi * dev, which
                decays toward its long-run level; base held at the trailing
                60-day median. Its 90th percentile adds the 90th percentile of
                the last 250 one-day forecast errors.
  month-end and quarter-end
                the turn quantile model (src/turn_quantiles.py), evaluated
                from its exported coefficients at the current scarcity state.
  year-end      no model: the window is not priced.

The 90th percentile roll cost puts every day at its own 90th percentile at
once. That is a stress case rather than the 90th percentile of the average,
which would need the joint distribution of daily errors: read it as an upper
reference.

A window containing an FOMC decision or its effective day is not priced:
the cost depends on a policy path the model does not have.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from market_calendar import business_days, fomc_meetings, next_business_day

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["is_qe", "rrp_bn_lag1", "tga_bn_lag1", "base_bp"]
WINSOR_BP = 25.0


# ---------------------------------------------------------------- calendar
def roll_days(start, end) -> pd.DataFrame:
    """Settlement days in [start, end) with the calendar days each accrues."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end <= start:
        return pd.DataFrame({"date": pd.DatetimeIndex([]), "days": []})
    days = business_days(start, end - pd.Timedelta(days=1))
    nxt = [min(next_business_day(d), end) for d in days]
    return pd.DataFrame({"date": days, "days": [(n - d).days for d, n in zip(days, nxt)]})


def turn_kind(d) -> str | None:
    """month-end, quarter-end or year-end if d is the month's last settlement day."""
    d = pd.Timestamp(d)
    if next_business_day(d).month == d.month:
        return None
    if d.month == 12:
        return "year-end"
    return "quarter-end" if d.month in (3, 6, 9) else "month-end"


def upcoming_turns(after, until) -> list[tuple[pd.Timestamp, str]]:
    days = business_days(pd.Timestamp(after), pd.Timestamp(until))
    return [(d, k) for d in days if (k := turn_kind(d))]


def fomc_in_window(start, end) -> list[pd.Timestamp]:
    """Decision days whose meeting or effective day touches [start, end)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hits = []
    for _, m in fomc_meetings().iterrows():
        eff = next_business_day(m["meeting_end"])
        if (m["meeting_start"] < end and m["meeting_end"] >= start) or (start <= eff < end):
            hits.append(m["meeting_end"])
    return hits


# ------------------------------------------------------------------ inputs
@dataclass
class State:
    anchor_date: pd.Timestamp    # first day with a model forecast
    anchor_dev: float            # its deviation forecast, bp
    base_bp: float               # trailing 60-day median of SOFR minus top
    target_hi: float             # top of the target range, percent
    ar_c: float
    ar_phi: float
    err_q90: float               # 90th percentile of one-day errors, bp
    scarcity: dict = field(default_factory=dict)   # rrp_bn_lag1, tga_bn_lag1


def ar1_coefs(daily: pd.DataFrame, window: int = 500) -> tuple[float, float]:
    """AR(1) of tomorrow's winsorised deviation on today's, last `window` days:
    the first stage of the two-stage model, by least squares."""
    d = daily.sort_values("date")
    x = d["dev_bp"].clip(-WINSOR_BP, WINSOR_BP)
    y = d["dev_w_bp"].shift(-1)
    ok = x.notna() & y.notna()
    x, y = x[ok].iloc[-window:].to_numpy(), y[ok].iloc[-window:].to_numpy()
    phi, c = np.polyfit(x, y, 1)
    return float(c), float(phi)


def turn_quantiles(coefs: pd.DataFrame, kind: str, scarcity: dict, base_bp: float) -> dict:
    """q50, q90, q95 of SOFR minus top of range, bp, for a month- or quarter-end."""
    x = {"is_qe": float(kind == "quarter-end"), "base_bp": base_bp, **scarcity}
    q = [float(r["intercept"] + sum(r[f] * x[f] for f in FEATURES)) for _, r in coefs.iterrows()]
    q = np.maximum.accumulate(q)
    return dict(zip(coefs["quantile"], q))


# ------------------------------------------------------------------- price
def day_rates(days: pd.DataFrame, st: State, coefs: pd.DataFrame) -> pd.DataFrame:
    """Central and 90th percentile rate, percent, for every day in the window."""
    out = days.copy()
    out["kind"] = [turn_kind(d) for d in out["date"]]
    steps = {d: i for i, d in enumerate(business_days(st.anchor_date, out["date"].max()))} \
        if len(out) else {}
    central, high = [], []
    for d, kind in zip(out["date"], out["kind"]):
        if kind in ("month-end", "quarter-end"):
            q = turn_quantiles(coefs, kind, st.scarcity, st.base_bp)
            central.append(st.target_hi + q["q50"] / 100)
            high.append(st.target_hi + q["q90"] / 100)
            continue
        dev = st.anchor_dev
        for _ in range(steps[d]):
            dev = st.ar_c + st.ar_phi * dev
        central.append(st.target_hi + (st.base_bp + dev) / 100)
        high.append(st.target_hi + (st.base_bp + dev + st.err_q90) / 100)
    out["central_pct"], out["p90_pct"] = central, high
    return out


def weighted_rate(rates, days) -> float:
    """Day-weighted average rate: what one placement for the whole window earns."""
    days = np.asarray(days, float)
    return float(np.sum(np.asarray(rates, float) * days) / days.sum())


def price(start, end, quote_pct: float, st: State, coefs: pd.DataFrame) -> dict:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = roll_days(start, end)
    res = {"days": days, "n_days": int(days["days"].sum()) if len(days) else 0,
           "fomc": fomc_in_window(start, end), "year_end": [], "too_early": False}
    if days.empty:
        return res
    res["year_end"] = [d for d in days["date"] if turn_kind(d) == "year-end"]
    res["too_early"] = bool(days["date"].min() < st.anchor_date)
    if res["fomc"] or res["year_end"] or res["too_early"]:
        return res
    r = day_rates(days, st, coefs)
    central = weighted_rate(r["central_pct"], r["days"])
    high = weighted_rate(r["p90_pct"], r["days"])
    res.update(rates=r, central_pct=central, p90_pct=high,
               breakeven_bp=(central - quote_pct) * 100,
               breakeven_p90_bp=(high - quote_pct) * 100)
    return res


def verdict(res: dict) -> str | None:
    """'term' if the quote is below the central roll cost, 'insure' if it sits
    between the central and the 90th percentile, 'roll' if above both."""
    if "central_pct" not in res:
        return None
    if res["breakeven_bp"] > 0:
        return "term"
    return "insure" if res["breakeven_p90_bp"] > 0 else "roll"
