"""What the 90% line is worth to a borrower who must fund over a turn.

Run from the project root, after turn_quantiles.py:
    python src/turn_value.py

The question
    A desk must borrow overnight cash across a month- or quarter-end. Days
    ahead, it can lock a fixed rate for the turn or roll and pay whatever
    prints. The 90% line is read as a break-even: the highest fixed rate the
    borrower should accept for the turn. Two numbers follow for every turn:
      cushion    what the borrower overpays if it locks at the line and the
                 turn prints below it, max(line - realised, 0);
      overshoot  what it loses if it rolls and the line fails,
                 max(realised - line, 0).
    A good line keeps the overshoot rare and small without a wide cushion.

Inputs, nothing refitted
    data/processed/turn_quantiles_horizon.csv: the walk-forward, out-of-sample
    quantile forecasts; q90_cal is the calibrated 90% line at horizons 1, 5,
    10 and 15 settlement days.
    The naive range, rebuilt exactly as turn_quantiles.py builds it
    (add_naive, then naive_ranges): the spread on the forecast day plus the
    90th percentile of past naive errors at the same horizon.
    The sample is the one turn_naive_ranges.csv scores: 40 month-ends and
    16 quarter-ends with both methods available at every horizon. The script
    asserts that its hit counts match that file. A hit is realised <= line,
    compared on the spread scale exactly as that file does: three naive lines
    (31 May 2023 and 30 Nov 2023 at h = 1, 31 Aug 2022 at h = 5) equal the
    print up to floating-point noise and count as misses with zero overshoot.

Units
    line      target_hi on the turn + q90 / 100, a SOFR level in percent. The
              quantiles are spreads to the top of the range, so the turn's own
              range is used; the overshoot and cushion are the same as on the
              spread scale.
    realised  SOFR printed on the turn.
    nights    calendar days the turn placement accrues, from the turn to the
              next settlement day in market_calendar: 1, or 3 when the turn
              is a Friday and carries the weekend.
    usd_per_bn = bp / 10,000 * 1e9 * nights / 360 (actual/360, like SOFR).

What this is not
    This is not a P&L. Public data has no term repo quotes, so there is no
    rate at which the turn could actually have been locked, and SOFR is a
    benchmark, a volume-weighted median of many trades, not a rate any one
    desk can deal at. The figures size what a range is worth as a budget
    line, not money anyone made.

Writes data/processed/turn_value.csv (one row per turn, horizon and method)
and data/processed/turn_value_summary.csv (by type, horizon and method).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
import market_calendar  # noqa: E402
import turn_quantiles as tq  # noqa: E402

METHODS = {"calibrated model": "q90_cal", "naive range": "nq90"}
TYPES = ("quarter-end", "month-end")


def usd_per_bn(bp, nights):
    """Interest on $1bn for `nights` nights at `bp` basis points, actual/360."""
    return np.asarray(bp, float) / 10_000 * 1e9 * np.asarray(nights, float) / 360


def nights_after(day) -> int:
    day = pd.Timestamp(day)
    return int((market_calendar.next_business_day(day) - day).days)


def sample(daily: pd.DataFrame, hz: pd.DataFrame) -> pd.DataFrame:
    """The turns turn_naive_ranges.csv scores, with both 90% lines."""
    t = tq.turn_days(daily)
    nz = tq.naive_ranges(tq.add_naive(hz, daily, t))
    f = nz.dropna(subset=["q50_cal", "nq50", "naive", "prev_turn"])
    common = set.intersection(*(set(g["date"]) for _, g in f.groupby("horizon")))
    return f[f["date"].isin(common)].reset_index(drop=True)


def value(daily: pd.DataFrame, hz: pd.DataFrame) -> pd.DataFrame:
    f = sample(daily, hz)
    day = daily.set_index("date")
    rows = []
    for _, r in f.iterrows():
        top, sofr = float(day.at[r["date"], "target_hi"]), float(day.at[r["date"], "sofr"])
        nights = nights_after(r["date"])
        for method, col in METHODS.items():
            q = float(r[col])
            # Hit as turn_quantiles.py scores it, y <= q on the spread scale,
            # so ties fall the same way; y is exactly (SOFR - top) * 100.
            gap = float(r["y"]) - q                        # bp, realised minus line
            rows.append({"date": r["date"].date(), "type": r["type"], "horizon": int(r["horizon"]),
                         "method": method, "made_date": pd.Timestamp(r["made_date"]).date(),
                         "target_hi_pct": top, "line_pct": top + q / 100, "realised_pct": sofr,
                         "hit": int(gap <= 0), "nights": nights,
                         "overshoot_bp": max(round(gap, 6), 0.0),
                         "cushion_bp": max(round(-gap, 6), 0.0)})
    v = pd.DataFrame(rows)
    v["overshoot_usd_per_bn"] = usd_per_bn(v["overshoot_bp"], v["nights"])
    v["cushion_usd_per_bn"] = usd_per_bn(v["cushion_bp"], v["nights"])
    return v.sort_values(["type", "horizon", "method", "date"]).reset_index(drop=True)


def summary(v: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for typ in TYPES:
        for h in sorted(v["horizon"].unique()):
            for method in METHODS:
                g = v[(v["type"] == typ) & (v["horizon"] == h) & (v["method"] == method)]
                wb = g.loc[g["overshoot_bp"].idxmax()]
                wu = g.loc[g["overshoot_usd_per_bn"].idxmax()]
                rows.append({
                    "type": typ, "horizon": h, "method": method, "n": len(g),
                    "hits": int(g["hit"].sum()),
                    "mean_cushion_bp": g["cushion_bp"].mean(),
                    "mean_overshoot_bp": g["overshoot_bp"].mean(),
                    "worst_overshoot_bp": wb["overshoot_bp"],
                    "worst_overshoot_date": wb["date"] if wb["overshoot_bp"] > 0 else None,
                    "mean_cushion_usd_per_bn": g["cushion_usd_per_bn"].mean(),
                    "mean_overshoot_usd_per_bn": g["overshoot_usd_per_bn"].mean(),
                    "worst_overshoot_usd_per_bn": wu["overshoot_usd_per_bn"],
                    "worst_overshoot_usd_date": (wu["date"] if wu["overshoot_usd_per_bn"] > 0
                                                 else None),
                })
    return pd.DataFrame(rows)


def check_hits(s: pd.DataFrame, ranges: pd.DataFrame) -> None:
    """Hit counts must equal n x at_or_below_q90 in turn_naive_ranges.csv."""
    ref = ranges[ranges["type"].isin(TYPES)].set_index(["type", "horizon", "method"])
    for _, r in s.iterrows():
        k = (r["type"], r["horizon"], r["method"])
        expect = int(round(ref.at[k, "n"] * ref.at[k, "at_or_below_q90"]))
        assert int(ref.at[k, "n"]) == r["n"], f"{k}: n {r['n']} != {ref.at[k, 'n']}"
        assert r["hits"] == expect, f"{k}: {r['hits']} hits, turn_naive_ranges.csv has {expect}"


def main() -> int:
    daily = pd.read_csv(PROC / "daily.csv", parse_dates=["date"])
    hz = pd.read_csv(PROC / "turn_quantiles_horizon.csv", parse_dates=["date", "made_date"])
    v = value(daily, hz)
    s = summary(v)
    check_hits(s, pd.read_csv(PROC / "turn_naive_ranges.csv"))
    qe = s[s["type"] == "quarter-end"]
    assert list(qe[qe["method"] == "calibrated model"]["hits"]) == [15, 16, 16, 15]
    assert list(qe[qe["method"] == "naive range"]["hits"]) == [8, 8, 9, 10]
    v.to_csv(PROC / "turn_value.csv", index=False)
    s.to_csv(PROC / "turn_value_summary.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(s.round(2).to_string(index=False))
    print(f"\nwrote {PROC / 'turn_value.csv'} ({len(v)} rows) and turn_value_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
