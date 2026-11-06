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

# Administered rates come from data/fed_administered_rates.csv, one row per
# Federal Reserve implementation note, each carrying its source URL. The
# offsets are NOT stable, which is why the table exists rather than a rule:
# IORB sits 0, 5, 10, 15 or 20bp below the top of the range depending on the
# year, and ON RRP sits 5bp below the bottom in late 2019 and 5bp above it
# from June 2021 to December 2024.
#
# Between notes the last note's OFFSETS are carried forward and applied to that
# day's actual target range, so a range change with no note of its own is
# handled correctly. Every day records which note it came from.
RATES_CSV = "fed_administered_rates.csv"


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

    # Administered rates from the implementation-note table.
    notes = pd.read_csv(ROOT / "data" / RATES_CSV, parse_dates=["effective_date"])
    notes = notes.sort_values("effective_date").reset_index(drop=True)
    notes["iorb_off"] = notes["iorb_pct"] - notes["target_hi"]
    notes["onrrp_off"] = notes["onrrp_pct"] - notes["target_lo"]
    notes["note_date"] = notes["effective_date"]
    df = pd.merge_asof(
        df, notes[["effective_date", "iorb_off", "onrrp_off", "note_date",
                   "target_lo", "target_hi", "source_url"]]
        .rename(columns={"target_lo": "note_lo", "target_hi": "note_hi"}),
        left_on="date", right_on="effective_date", direction="backward")
    assert df["iorb_off"].notna().all(), "a date precedes the first note in the table"

    # Independent cross-check: on a note's own effective date, the range the
    # note states must match the range the NY Fed EFFR series carries.
    on_note = df[df["date"] == df["note_date"]]
    bad = on_note[(on_note["note_lo"] - on_note["target_lo"]).abs().gt(1e-9)
                  | (on_note["note_hi"] - on_note["target_hi"]).abs().gt(1e-9)]
    assert bad.empty, f"note range disagrees with NY Fed range:\n{bad[['date']]}"
    print(f"\n{len(notes)} implementation notes; range agrees with the NY Fed "
          f"series on all {len(on_note)} note dates present in the sample")

    df["iorb_pct"] = df["target_hi"] + df["iorb_off"]
    rule = df["target_lo"] + df["onrrp_off"]
    df["award_pct"] = df["rrp_award_pct"].fillna(rule)
    df["award_src"] = np.where(df["rrp_award_pct"].notna(), "observed", "note")
    # A day whose range differs from its note's range relies on the offset
    # having persisted across a rate change no note in the table covers.
    df["offset_carried"] = ((df["target_lo"] - df["note_lo"]).abs().gt(1e-9)
                            | (df["target_hi"] - df["note_hi"]).abs().gt(1e-9)).astype(int)

    # The table must reproduce every directly observed award rate.
    obs = df[df["rrp_award_pct"].notna()]
    mism = obs[(obs["rrp_award_pct"] - (obs["target_lo"] + obs["onrrp_off"])).abs() > 1e-9]
    print(f"table reproduces the observed award rate on "
          f"{len(obs) - len(mism)}/{len(obs)} observed days")
    assert mism.empty, f"table disagrees with observed award rate:\n{mism[['date']].head()}"

    df["carry_bp"] = (df["sofr"] - df["award_pct"]) * 100
    df["carry_usable"] = 1
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
    print(f"days whose offset is carried across an uncovered rate change: "
          f"{int(df['offset_carried'].sum())}")
    use = df[df["carry_usable"] == 1]
    print(f"carry usable on {len(use)}/{len(df)} days "
          f"({use['date'].min().date()} -> {use['date'].max().date()}); "
          f"negative on {int((use['carry_bp'] < 0).sum())} of them "
          f"({100 * (use['carry_bp'] < 0).mean():.1f}%), median {use['carry_bp'].median():.1f}bp")
    bad = df.loc[(df["award_src"] == "observed")
                 & ((df["award_pct"] - df["rrp_award_pct"]).abs() > 1e-9)]
    assert bad.empty, "observed award rate overwritten"
    print("\nIORB offset below the top of the range, by period (bp):")
    print((df.groupby(df["note_date"].dt.date)["iorb_off"].first() * 100)
          .round(0).astype(int).to_string())
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
