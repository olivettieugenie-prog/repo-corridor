"""Download the public series for the repo-spread project and check them.

Run from the project root:
    python src/fetch_data.py            use cached files, download what is missing
    python src/fetch_data.py --refresh  download everything again
    python src/fetch_data.py --refresh tga   download only the named series

Required sources (no API key):
  NY Fed     SOFR (daily rate, percentiles, volume)
  NY Fed     EFFR (daily rate) with the Fed target range
  NY Fed     overnight reverse repo (ON RRP) operations
  Treasury   daily Treasury General Account (TGA) balance, Fiscal Data API

Optional source:
  FRED       IORB and weekly reserves. FRED timed out from the author's
             network, so a failure here is reported and does not stop the run.

Nothing is invented: a series that fails is reported and skipped.
"""
from __future__ import annotations

import io
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if "--root" in sys.argv:  # same option as build_dataset.py and model_lasso.py
    ROOT = Path(sys.argv[sys.argv.index("--root") + 1])
RAW = ROOT / "data" / "raw"
START_YEAR = 2018  # SOFR was first published in April 2018
START_DATE = "2018-04-01"

NYFED = "https://markets.newyorkfed.org/api"
FISCAL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# The Daily Treasury Statement changed its row labels over time. Listed by
# priority: (label, column that holds the closing balance).
TGA_LABELS = [
    ("Treasury General Account (TGA) Closing Balance", "open_today_bal"),
    ("Treasury General Account (TGA)", "close_today_bal"),
    ("Federal Reserve Account", "close_today_bal"),
]

FRED_OPTIONAL = {
    "IORB": "Interest on reserve balances, daily, from 2021-07-29",
    "WRESBAL": "Reserve balances at the Fed, weekly (Wednesday level)",
}

HEADERS = {"User-Agent": "repo-spread-student-project/0.2"}


def get(url: str, params: dict, tries: int = 4, timeout=(10, 120)) -> requests.Response:
    """GET with a read timeout and retries."""
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last = exc
            if attempt < tries:
                wait = 5 * attempt
                print(f"  retry {attempt}/{tries} after {type(exc).__name__}, waiting {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"gave up on {url}: {last!r}")


def _nyfed_rate(path: str) -> pd.DataFrame:
    """A NY Fed reference-rate endpoint, fetched one calendar year at a time."""
    frames = []
    for year in range(START_YEAR, date.today().year + 1):
        params = {"startDate": f"{year}-01-01", "endDate": f"{year}-12-31"}
        rows = get(f"{NYFED}/rates/{path}/search.json", params).json().get("refRates", [])
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise RuntimeError(f"NY Fed returned no rows for {path}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["effectiveDate"])
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_sofr() -> pd.DataFrame:
    df = _nyfed_rate("secured/sofr")
    keep = {
        "percentRate": "sofr",
        "percentPercentile1": "p01",
        "percentPercentile25": "p25",
        "percentPercentile75": "p75",
        "percentPercentile99": "p99",
        "volumeInBillions": "volume_bn",
    }
    return df.rename(columns=keep)[["date", *keep.values()]]


def fetch_effr() -> pd.DataFrame:
    """EFFR with the Fed target range (targetRateFrom is the lower bound)."""
    df = _nyfed_rate("unsecured/effr")
    keep = {
        "percentRate": "effr",
        "targetRateFrom": "target_lo",
        "targetRateTo": "target_hi",
        "volumeInBillions": "effr_volume_bn",
    }
    present = {k: v for k, v in keep.items() if k in df.columns}
    return df.rename(columns=present)[["date", *present.values()]]


def fetch_rrp() -> pd.DataFrame:
    """ON RRP amounts accepted, in billions of dollars, summed per day."""
    rows = []
    for year in range(START_YEAR, date.today().year + 1):
        params = {"startDate": f"{year}-01-01", "endDate": f"{year}-12-31"}
        js = get(f"{NYFED}/rp/reverserepo/propositions/search.json", params).json()
        rows += js.get("repo", {}).get("operations", [])
    if not rows:
        raise RuntimeError("NY Fed returned no reverse repo operations")
    df = pd.DataFrame(rows)
    print("  operationType values:", sorted(df["operationType"].dropna().unique()))
    df["date"] = pd.to_datetime(df["operationDate"])
    df["rrp_bn"] = pd.to_numeric(df["totalAmtAccepted"], errors="coerce") / 1e9
    return df.groupby("date", as_index=False)["rrp_bn"].sum().sort_values("date")


RRP_N_CANDIDATES = (500, 400, 250, 100, 50)


def fetch_rrp_results() -> pd.DataFrame:
    """ON RRP operations with the award rate, from the results endpoint.

    The propositions endpoint used by fetch_rrp() accepts a date range but
    carries no rate. This endpoint carries percentAwardRate but only accepts a
    count of recent operations, and that count is capped: 400 works, 1000 and
    3000 return 400 Bad Request. The cap is undocumented here, so the largest
    accepted value is found by probing. How far back the result reaches is
    whatever that count buys, and the printed earliest date is the answer.

    Field names were read off a summarised response, not raw JSON, so every
    field is optional here and the ones actually found are printed.
    """
    ops = []
    for n in RRP_N_CANDIDATES:
        try:
            js = get(f"{NYFED}/rp/reverserepo/all/results/last/{n}.json", {}, tries=1).json()
        except RuntimeError as exc:
            print(f"  n={n} rejected ({type(exc).__name__}), trying smaller")
            continue
        ops = js.get("repo", {}).get("operations", [])
        if ops:
            print(f"  n={n} accepted, {len(ops)} operations")
            break
    if not ops:
        raise RuntimeError("no accepted count returned operations")
    print("  top-level fields found:", sorted(ops[0].keys()))

    rows = []
    for op in ops:
        if "reverse" not in str(op.get("operationType", "")).lower():
            continue
        details = op.get("details") or []
        if isinstance(details, dict):
            details = [details]
        # Treasury collateral is the ON RRP facility; fall back to any detail.
        pick = next((d for d in details
                     if "treasury" in str(d.get("securityType", "")).lower()), None)
        if pick is None:
            pick = details[0] if details else {}
        rows.append({
            "date": op.get("operationDate"),
            "rrp_award_pct": pick.get("percentAwardRate"),
            "rrp_offering_pct": pick.get("percentOfferingRate"),
            "rrp_accepted_bn": (op.get("totalAmtAccepted") or 0) / 1e9,
            "rrp_submitted_bn": (op.get("totalAmtSubmitted") or 0) / 1e9,
            "rrp_cpty": op.get("participatingCpty") or op.get("acceptedCpty"),
        })
    if details and rows and rows[0]["rrp_award_pct"] is None:
        print("  WARNING: no percentAwardRate found. detail fields:",
              sorted(details[0].keys()) if details else "none")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns.drop("date"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.groupby("date", as_index=False).agg(
        rrp_award_pct=("rrp_award_pct", "max"),
        rrp_offering_pct=("rrp_offering_pct", "max"),
        rrp_accepted_bn=("rrp_accepted_bn", "sum"),
        rrp_submitted_bn=("rrp_submitted_bn", "sum"),
        rrp_cpty=("rrp_cpty", "max"),
    ).sort_values("date")
    have = df["rrp_award_pct"].notna().sum()
    print(f"  award rate present on {have}/{len(df)} days; "
          f"earliest {df['date'].min().date()}, latest {df['date'].max().date()}")
    return df.reset_index(drop=True)


def fetch_tga() -> pd.DataFrame:
    """Daily TGA closing balance in billions, from the Daily Treasury Statement.

    The statement changed its row labels. The new label keeps the closing
    balance in open_today_bal, the old one in close_today_bal. The new label
    wins when both exist on a date.
    """
    url = f"{FISCAL}/accounting/dts/operating_cash_balance"
    rows, page = [], 1
    while True:
        params = {
            "filter": f"record_date:gte:{START_DATE}",
            "fields": "record_date,account_type,open_today_bal,close_today_bal",
            "sort": "record_date",
            "page[size]": 10000,
            "page[number]": page,
        }
        js = get(url, params).json()
        rows += js["data"]
        if page >= js["meta"]["total-pages"]:
            break
        page += 1
    df = pd.DataFrame(rows)
    parts = []
    for rank, (label, col) in enumerate(TGA_LABELS):
        part = df[df["account_type"] == label].copy()
        part["tga"] = pd.to_numeric(part[col], errors="coerce")
        part["rank"], part["label"] = rank, label
        parts.append(part)
    both = pd.concat(parts).dropna(subset=["tga"])
    both["date"] = pd.to_datetime(both["record_date"])
    both = both.sort_values(["date", "rank"]).drop_duplicates("date", keep="first")
    for label, grp in both.groupby("label", sort=False):
        print(f"  label '{label}': {grp['date'].min().date()} -> {grp['date'].max().date()}, {len(grp)} rows")
    out = pd.DataFrame({"date": both["date"], "tga_bn": both["tga"] / 1000})
    print(f"  TGA min {out['tga_bn'].min():.1f}bn on {out.loc[out['tga_bn'].idxmin(), 'date'].date()}, "
          f"max {out['tga_bn'].max():.1f}bn on {out.loc[out['tga_bn'].idxmax(), 'date'].date()}")
    return out.reset_index(drop=True)


def fetch_fred(series_id: str) -> pd.DataFrame:
    r = get(FRED_CSV, {"id": series_id}, tries=1, timeout=(10, 30))
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", series_id.lower()]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id.lower()] = pd.to_numeric(df[series_id.lower()], errors="coerce")
    return df.dropna().reset_index(drop=True)


def report(name: str, df: pd.DataFrame) -> None:
    gaps = df["date"].diff().dt.days.dropna()
    print(
        f"{name:<10} rows={len(df):>5}  "
        f"{df['date'].min().date()} -> {df['date'].max().date()}  "
        f"max gap={int(gaps.max()) if len(gaps) else 0}d"
    )


def run(name: str, job, refresh: bool) -> pd.DataFrame | None:
    path = RAW / f"{name}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["date"])
        print(f"{name:<10} cached", end="  ")
        report(name, df)
        return df
    try:
        df = job()
    except Exception as exc:  # report and continue, never fabricate
        print(f"{name:<10} FAILED  {exc!r}")
        return None
    df.to_csv(path, index=False)
    report(name, df)
    return df


def main() -> int:
    args = sys.argv[1:]
    if "--root" in args:
        i = args.index("--root")
        args = args[:i] + args[i + 2:]
    refresh_all = args == ["--refresh"]
    names = {a for a in args if not a.startswith("--")}  # e.g. --refresh tga rrp
    refresh = lambda n: refresh_all or ("--refresh" in args and n in names)
    RAW.mkdir(parents=True, exist_ok=True)

    required = {"sofr": fetch_sofr, "effr": fetch_effr, "rrp": fetch_rrp, "tga": fetch_tga,
                "rrp_rate": fetch_rrp_results}
    failed = [n for n, job in required.items() if run(n, job, refresh(n)) is None]

    print("\nOptional (FRED):")
    if os.environ.get("REPO_SKIP_FRED"):  # set in CI, where FRED is not needed
        print("  skipped: REPO_SKIP_FRED is set")
        skipped: list[str] = []
    else:
        skipped = [
            sid for sid in FRED_OPTIONAL
            if run(sid.lower(), (lambda s=sid: fetch_fred(s)), refresh(sid.lower())) is None
        ]

    if failed:
        print("\nRequired series failed:", ", ".join(failed))
        return 1
    if skipped:
        print("\nOptional series skipped:", ", ".join(skipped))
    print("\nRequired series are in", RAW)
    return 0


if __name__ == "__main__":
    sys.exit(main())
