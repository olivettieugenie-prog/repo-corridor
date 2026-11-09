"""Standing Repo Facility: daily take-up, and a check of its rate table.

Run from the project root:
    python src/fetch_srf.py

Take-up: the NY Fed markets API, /api/rp/results/search.json with
operationTypes=Repo, from 29 July 2021, the first operation of the standing
facility (earlier repo operations were temporary ones and are left out).
Amounts accepted are summed per operation date, in billions of dollars, and
written to data/processed/srf_takeup.csv.

Rate: data/srf_rates.csv holds one row per FOMC implementation note since
28 July 2021, with its URL. Through October 2025 the notes set a minimum bid
rate; from December 2025 a fixed full-allotment rate. The API's range query
carries no rate, so the notes are the source. The check below asserts that
every rate equals the top of the target range on its effective date, as the
notes state, using the range in data/processed/daily.csv.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fetch_data import NYFED, get  # noqa: E402

START = "2021-07-29"
OUT = ROOT / "data" / "processed" / "srf_takeup.csv"


def fetch_takeup() -> pd.DataFrame:
    rows = []
    for year in range(2021, date.today().year + 1):
        params = {"startDate": max(START, f"{year}-01-01"), "endDate": f"{year}-12-31",
                  "operationTypes": "Repo"}
        js = get(f"{NYFED}/rp/results/search.json", params).json()
        rows += js.get("repo", {}).get("operations", [])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["operationDate"])
    df["srf_bn"] = pd.to_numeric(df["totalAmtAccepted"], errors="coerce").fillna(0) / 1e9
    out = df[df["date"] >= START].groupby("date", as_index=False)["srf_bn"].sum()
    return out.sort_values("date").reset_index(drop=True)


def check_rates() -> pd.DataFrame:
    rates = pd.read_csv(ROOT / "data" / "srf_rates.csv", parse_dates=["effective_date"])
    daily = pd.read_csv(ROOT / "data" / "processed" / "daily.csv", parse_dates=["date"])
    top = daily.set_index("date")["target_hi"]
    inside = rates[rates["effective_date"] <= top.index.max()]
    gap = (inside["srf_rate_pct"] - inside["effective_date"].map(top.asof)).abs()
    assert (gap < 1e-9).all(), inside[gap >= 1e-9]
    print(f"srf_rates.csv: {len(rates)} notes; rate equals the top of the range on "
          f"all {len(inside)} effective dates inside daily.csv")
    return rates


def main() -> int:
    check_rates()
    t = fetch_takeup()
    t.to_csv(OUT, index=False)
    print(f"take-up {t['date'].min().date()} -> {t['date'].max().date()}, {len(t)} days, "
          f"{int((t['srf_bn'] > 0).sum())} with any take-up, max {t['srf_bn'].max():.1f}bn on "
          f"{t.loc[t['srf_bn'].idxmax(), 'date'].date()}; wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
