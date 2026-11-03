"""Merge the downloaded series into one daily table, with no look-ahead.

Run from the project root, after fetch_data.py:
    python src/build_dataset.py

Output: data/processed/daily.csv, one row per SOFR date.

Timing rules
  rrp_bn_t       same-day ON RRP amount. Known when SOFR is set, so it is fine
                 to explain SOFR with it, but not to forecast with it.
  *_lag1         the last observation strictly before the SOFR date. Safe for
                 forecasting. The Treasury balance is a closing balance, so it
                 only ever appears lagged.
  Every lagged column has a *_date column and the script asserts that date is
  earlier than the SOFR date.

Spreads are in basis points (1 bp = 0.01 percentage point).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if "--root" in sys.argv:
    ROOT = Path(sys.argv[sys.argv.index("--root") + 1])
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

BASE_WIN, BASE_MIN = 60, 30  # trailing median window, and minimum to emit one
WINSOR_BP = 25.0  # fixed cap on the modelling target, stated not fitted

# Administered rates against the target range, read off the Federal Reserve's
# own implementation notes (date = effective date):
#   2018-12-20  range 2.25-2.50  IOER 2.40 (top-10)  ON RRP 2.25 (bottom)
#   2019-10-31  range 1.50-1.75  IOER 1.55 (top-20)  ON RRP 1.45 (bottom-5)
#   2021-06-17  range 0.00-0.25  IORB 0.15 (top-10)  ON RRP 0.05 (bottom+5)
#   2022-06-16  range 1.50-1.75  IORB 1.65 (top-10)  ON RRP 1.55 (bottom+5)
#   2023-07-27  range 5.25-5.50  IORB 5.40 (top-10)  ON RRP 5.30 (bottom+5)
#   2024-12-19  range 4.25-4.50  IORB 4.40 (top-10)  ON RRP 4.25 (bottom)
# From 2021-06-17 the offsets are stable and verified at three interior dates
# plus the 30 observed days to 2024-12-18, which show exactly +5.0bp. Before
# that they are not: October 2019 puts ON RRP *below* the range and IOER 20bp
# under the top. So nothing before 2021-06-17 is reconstructed here.
PLUS5_START, PLUS5_END = pd.Timestamp("2021-06-17"), pd.Timestamp("2024-12-18")
RULE_VERIFIED_FROM = PLUS5_START
IORB_OFFSET = 0.10  # below the top of the range, verified from 2021-06-17


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv", parse_dates=["date"]).sort_values("date")


def asof(left: pd.DataFrame, right: pd.DataFrame, name: str, cols: list[str], strict: bool,
         suffix: str, tolerance_days: int = 5) -> pd.DataFrame:
    """Attach the latest right-hand observation on or (strictly) before each date.

    Adds a column <name><suffix>_date with the date of the observation used.
    """
    r = right[["date", *cols]].copy()
    r[f"{name}{suffix}_date"] = r["date"]
    r = r.rename(columns={c: f"{c}{suffix}" for c in cols})
    return pd.merge_asof(
        left, r, on="date", direction="backward",
        allow_exact_matches=not strict, tolerance=pd.Timedelta(days=tolerance_days),
    )


def calendar_flags(dates: pd.Series) -> pd.DataFrame:
    """Month-end, quarter-end, year-end, mid-month settlement and month-start flags.

    Month-end is the last observed date on or before the last weekday of the
    month, and is only set for months that are complete in the data.
    """
    idx = pd.DatetimeIndex(dates)
    last = idx.max()
    flags = pd.DataFrame({"date": idx})
    flags["month_end"] = 0
    flags["month_start"] = 0
    flags["mid_settle"] = 0
    months = pd.period_range(idx.min(), idx.max(), freq="M")
    for m in months:
        in_m = idx[(idx.year == m.year) & (idx.month == m.month)]
        if len(in_m) == 0:
            continue
        flags.loc[flags["date"] == in_m.min(), "month_start"] = 1
        end_wkday = pd.Timestamp(m.year, m.month, 1) + pd.offsets.BMonthEnd(0)
        if end_wkday <= last:
            flags.loc[flags["date"] == in_m[in_m <= end_wkday].max(), "month_end"] = 1
        fifteenth = pd.Timestamp(m.year, m.month, 15)
        after = in_m[in_m >= fifteenth]
        if len(after):
            flags.loc[flags["date"] == after.min(), "mid_settle"] = 1
    flags["quarter_end"] = ((flags["month_end"] == 1) & flags["date"].dt.month.isin([3, 6, 9, 12])).astype(int)
    flags["year_end"] = ((flags["month_end"] == 1) & (flags["date"].dt.month == 12)).astype(int)
    return flags


def main() -> int:
    sofr, effr, rrp, tga = (load(n) for n in ("sofr", "effr", "rrp", "tga"))

    df = sofr.merge(effr, on="date", how="left")
    df["spread_hi_bp"] = (df["sofr"] - df["target_hi"]) * 100
    df["spread_effr_bp"] = (df["sofr"] - df["effr"]) * 100
    df["disp_bp"] = (df["p99"] - df["p01"]) * 100
    df["iqr_bp"] = (df["p75"] - df["p25"]) * 100
    df["log_volume"] = np.log(df["volume_bn"])

    rrp = rrp.assign(d_rrp_bn=rrp["rrp_bn"].diff())
    tga = tga.assign(d_tga_bn=tga["tga_bn"].diff())
    df = asof(df, rrp, "rrp", ["rrp_bn"], strict=False, suffix="_t", tolerance_days=4)
    df = asof(df, rrp, "rrp", ["rrp_bn", "d_rrp_bn"], strict=True, suffix="_lag1")
    df = asof(df, tga, "tga", ["tga_bn", "d_tga_bn"], strict=True, suffix="_lag1")

    # No look-ahead: every lagged observation is strictly earlier than the SOFR date.
    lag_dates = [c for c in df.columns if c.endswith("_lag1_date")]
    assert lag_dates == ["rrp_lag1_date", "tga_lag1_date"], lag_dates
    for src in lag_dates:
        ok = df[src].isna() | (df[src] < df["date"])
        assert ok.all(), f"look-ahead in {src}"
    # The same-day RRP date may equal the SOFR date but never exceed it.
    assert (df["rrp_t_date"].isna() | (df["rrp_t_date"] <= df["date"])).all()

    df = df.merge(calendar_flags(df["date"]), on="date")
    df["year"] = df["date"].dt.year

    # Slow component: a past-only trailing median. Removing it strips the
    # regime drift (yearly means ran from -4.5bp to -21.6bp on the raw spread)
    # so the model explains the fast, day-level variation instead of the drift.
    df["base_bp"] = df["spread_hi_bp"].shift(1).rolling(BASE_WIN, min_periods=BASE_MIN).median()
    df["dev_bp"] = df["spread_hi_bp"] - df["base_bp"]

    # Winsorised target. 2019-09-17 alone is 46% of the squared deviation of
    # the raw spread, so an unwinsorised least-squares fit is a fit of that one
    # day. The cap is a fixed, stated number rather than a sample percentile,
    # which would leak information from the future into a backtest.
    df["dev_w_bp"] = df["dev_bp"].clip(-WINSOR_BP, WINSOR_BP)
    n_cap = int((df["dev_bp"].abs() > WINSOR_BP).sum())

    # Carry on cash placed overnight: lend in repo at SOFR, or place at the Fed.
    # The ON RRP award rate is the alternative. Where it is unavailable the
    # bottom of the target range stands in, which the check below tests.
    if (RAW / "rrp_rate.csv").exists():
        rate = load("rrp_rate")[["date", "rrp_award_pct", "rrp_cpty"]]
        df = df.merge(rate, on="date", how="left")
        ok = df.dropna(subset=["rrp_award_pct"])
        if len(ok):
            diff = (ok["rrp_award_pct"] - ok["target_lo"]).abs()
            bad = ok[diff > 1e-9]
            print(f"\nON RRP award rate available on {len(ok)} days "
                  f"({ok['date'].min().date()} -> {ok['date'].max().date()})")
            print(f"  equals bottom of target range on {len(ok) - len(bad)}/{len(ok)} of them")
            if len(bad):
                print("  mismatches:")
                print(bad[["date", "rrp_award_pct", "target_lo"]].head(10).to_string(index=False))
    else:
        df["rrp_award_pct"], df["rrp_cpty"] = pd.NA, pd.NA
        print("\nrrp_rate.csv absent: award rate not checked")

    # Award rate: observed where the API reaches, otherwise reconstructed from
    # the published offset rules above. Never a bare copy of the range bottom.
    in_plus5 = df["date"].between(PLUS5_START, PLUS5_END)
    rule = df["target_lo"] + np.where(in_plus5, 0.05, 0.0)
    rule = pd.Series(rule, index=df.index).where(df["date"] >= RULE_VERIFIED_FROM)
    df["award_pct"] = df["rrp_award_pct"].fillna(rule)
    df["award_src"] = np.where(
        df["rrp_award_pct"].notna(), "observed",
        np.where(df["date"] >= RULE_VERIFIED_FROM, "rule", "unavailable"))
    df["carry_bp"] = (df["sofr"] - df["award_pct"]) * 100
    df["carry_usable"] = df["award_src"].ne("unavailable").astype(int)

    # IORB, the rate a bank earns instead of lending in repo. Reconstructed
    # only where the offset is verified; left empty before, where it is not.
    df["iorb_pct"] = (df["target_hi"] - IORB_OFFSET).where(df["date"] >= RULE_VERIFIED_FROM)
    df["spread_iorb_bp"] = (df["sofr"] - df["iorb_pct"]) * 100
    df["is_zirp"] = (df["target_hi"] <= 0.25).astype(int)

    # Target-range changes move the reference rate while the trailing median
    # still reflects the old level, so dev_bp picks up a mechanical jump. Six
    # of the 26 changes land within a day of a month-end, which would otherwise
    # inflate the month-end coefficient. Flagged so the model estimates it.
    chg = df["target_hi"].diff().fillna(0) != 0
    df["rate_change"] = chg.astype(int)
    df["rate_change_lag1"] = chg.shift(1).fillna(False).astype(int)

    stale = (df["date"] - df["tga_lag1_date"]).dt.days
    print(f"TGA lag: median {stale.median():.0f} days, max {stale.max():.0f} days")

    print(f"daily.csv  rows={len(df)}  {df['date'].min().date()} -> {df['date'].max().date()}")
    print("\nMissing values per column (only columns with any):")
    miss = df.isna().sum()
    print(miss[miss > 0].to_string() if (miss > 0).any() else "  none")
    print("\nFlag counts:", {c: int(df[c].sum()) for c in
                            ("month_end", "quarter_end", "year_end", "mid_settle", "month_start")})
    print("\nMonth-end days per year (should be about 12, 2026 fewer):")
    print(df.groupby("year")["month_end"].sum().to_string())

    print(f"\nTarget: dev_bp = spread - trailing {BASE_WIN}d median "
          f"({df['dev_bp'].notna().sum()} usable rows, first "
          f"{df.loc[df['dev_bp'].notna(), 'date'].min().date()})")
    print(f"  winsorised at +/-{WINSOR_BP:.0f}bp: {n_cap} rows capped")
    print("\nDrift check, by year (bp):")
    chk = df.groupby("year").agg(raw_mean=("spread_hi_bp", "mean"),
                                 dev_mean=("dev_bp", "mean"),
                                 dev_std=("dev_bp", "std"),
                                 carry_mean=("carry_bp", "mean")).round(2)
    print(chk.to_string())
    print("\nAward rate source:")
    print(df["award_src"].value_counts().to_string())
    use = df[df["carry_usable"] == 1]
    print(f"carry usable on {len(use)}/{len(df)} days "
          f"({use['date'].min().date()} -> {use['date'].max().date()}); "
          f"negative on {int((use['carry_bp'] < 0).sum())} of them "
          f"({100 * (use['carry_bp'] < 0).mean():.1f}%), median {use['carry_bp'].median():.1f}bp")
    bad = df.loc[(df["award_src"] == "observed")
                 & ((df["award_pct"] - df["rrp_award_pct"]).abs() > 1e-9)]
    assert bad.empty, "observed award rate overwritten"
    print(f"\nTarget-range changes: {int(df['rate_change'].sum())}; "
          f"of which within a day of month-end: "
          f"{int(((df['rate_change'] == 1) & (df['month_end'] == 1)).sum() + ((df['rate_change_lag1'] == 1) & (df['month_end'] == 1)).sum())}")
    print(f"ZIRP days: {int(df['is_zirp'].sum())}")

    # Write only once every check above has passed. Writing first meant a
    # failed assertion left a corrupted daily.csv on disk, which then fed the
    # next analysis silently.
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "daily.csv", index=False)
    print(f"\nwrote {OUT / 'daily.csv'}  rows={len(df)}  cols={df.shape[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
