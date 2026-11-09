"""Past turn days whose scarcity state looks most like today's.

    python src/analogs.py                 the 8 closest analogs for each turn type now
    python src/analogs.py 2025-09-30      as of an earlier date, for checking

State: lagged ON RRP (rrp_bn_lag1), lagged TGA (tga_bn_lag1) and the trailing
60-day median of SOFR minus the top of the range (base_bp), as build_dataset.py
defines them. Each variable is standardised with the mean and standard
deviation of daily observations strictly before the query date, so nothing
after it shapes the distance. Candidates are turn days of the same type, also
strictly before the query date; distance is Euclidean in standardised units.

For each analog: date, type, the three state values, the distance, what SOFR
minus the top of the range printed that day, and the day's 99th percentile of
SOFR transactions (p99, already in daily.csv) in the same units. Year-ends are
included but labelled history only: there are 8 of them and no model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
STATE = ["rrp_bn_lag1", "tga_bn_lag1", "base_bp"]
TYPES = ("month-end", "quarter-end", "year-end")
K = 8


def turn_type(row) -> str:
    if row["year_end"] == 1:
        return "year-end"
    return "quarter-end" if row["quarter_end"] == 1 else "month-end"


def current_state(daily: pd.DataFrame) -> dict:
    """The latest state a desk has, for a turn still ahead: the last ON RRP
    and TGA levels in the table and the median spread over the last 60 days."""
    d = daily.sort_values("date")
    last = d.iloc[-1]
    return {"rrp_bn_lag1": float(last["rrp_bn_t"]) if pd.notna(last["rrp_bn_t"])
            else float(last["rrp_bn_lag1"]),
            "tga_bn_lag1": float(last["tga_bn_lag1"]),
            "base_bp": float(d["spread_hi_bp"].iloc[-60:].median())}


def analogs(daily: pd.DataFrame, state: dict, kind: str, as_of, k: int = K) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    past = daily[daily["date"] < as_of]
    mu = past[STATE].mean()
    sd = past[STATE].std().replace(0, 1.0)
    turns = past[past["month_end"] == 1].dropna(subset=STATE).copy()
    turns["type"] = turns.apply(turn_type, axis=1)
    turns = turns[turns["type"] == kind]
    z = (turns[STATE] - mu) / sd
    q = (pd.Series(state)[STATE] - mu) / sd
    turns["distance"] = np.sqrt(((z - q) ** 2).sum(axis=1))
    out = turns.nsmallest(k, "distance")
    out["printed_bp"] = out["spread_hi_bp"]
    out["p99_bp"] = (out["p99"] - out["target_hi"]) * 100
    out["history_only"] = kind == "year-end"
    return out[["date", "type", *STATE, "distance", "printed_bp", "p99_bp",
                "history_only"]].reset_index(drop=True)


def main() -> int:
    daily = pd.read_csv(PROC / "daily.csv", parse_dates=["date"])
    if len(sys.argv) > 1:
        as_of = pd.Timestamp(sys.argv[1])
        state = daily.set_index("date").loc[as_of, STATE].to_dict()
    else:
        as_of = daily["date"].max() + pd.Timedelta(days=1)
        state = current_state(daily)
    print("state:", {k: round(v, 1) for k, v in state.items()}, "as of", as_of.date())
    for kind in TYPES:
        print(f"\n{kind}{' (history only)' if kind == 'year-end' else ''}")
        a = analogs(daily, state, kind, as_of)
        print(a.round({c: 2 for c in [*STATE, "distance", "printed_bp", "p99_bp"]})
               .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
