"""Desk view logic: the morning note, flags, staleness and live track record.

Pure functions, so tests can call them without Streamlit. Every sentence comes
from a fixed template in src/i18n.py; only numbers, dates and the listed flags
are filled in, so nothing is generated freely.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from i18n import STRINGS, num
from market_calendar import previous_business_day

ET = ZoneInfo("America/New_York")
FLAG_KEYS = ["month_end", "quarter_end", "year_end", "mid_settle", "month_start", "rate_change"]
MIN_SCORED = 20
FLAT_BP = 0.5  # below this the note says "in line with" rather than above/below


def long_date(ts, lang: str) -> str:
    t = STRINGS[lang]
    ts = pd.Timestamp(ts)
    return f"{t['weekdays'][ts.day_of_week]} {ts.day} {t['months'][ts.month - 1]} {ts.year}"


def run_time_et(run_utc: str, lang: str) -> str:
    ts = pd.Timestamp(run_utc).tz_convert(ET)
    return f"{long_date(ts, lang)} {ts:%H:%M}"


def flags_in_play(row: pd.Series, lang: str) -> list[str]:
    t = STRINGS[lang]
    return [t[f"flag_{k}"] for k in FLAG_KEYS if int(row.get(k, 0) or 0) == 1]


def today_et() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(ET).date())


def is_stale(target, today=None) -> bool:
    """Stale when the target is more than one business day before today."""
    today = pd.Timestamp(today) if today is not None else today_et()
    return pd.Timestamp(target) < previous_business_day(today)


def morning_note(row: pd.Series, lang: str) -> tuple[str, str]:
    t = STRINGS[lang]
    n = lambda x, dec=1, sign=False: num(x, dec, lang, sign)  # noqa: E731
    f = float(row["forecast_bp"])
    args = {"target": long_date(row["target_date"], lang),
            "f": n(f, 1, True), "abs": n(abs(f), 1),
            "lo": n(row["lower_bp"], 1, True), "hi": n(row["upper_bp"], 1, True),
            "sofr": n(row["implied_sofr_pct"], 2)}
    if abs(f) < FLAT_BP:
        line1 = t["note_flat"].format(**args)
    else:
        line1 = t["note_line1"].format(direction=t["dir_above" if f > 0 else "dir_below"], **args)
    lv = {"rrp": n(row["rrp_bn"], 1), "drrp": n(row["d_rrp_bn"], 1, True),
          "tga": n(row["tga_bn"], 0), "dtga": n(row["d_tga_bn"], 0, True)}
    flags = flags_in_play(row, lang)
    line2 = (t["note_line2_flags"].format(flags=", ".join(flags), **lv) if flags
             else t["note_line2_none"].format(**lv))
    return line1, line2


def track_record(live: pd.DataFrame) -> dict:
    """Scored days (latest run per target), interval hit rate and RMSE of the
    model and of AR(1) on exactly those days."""
    start = live["target_date"].min()
    s = live.dropna(subset=["realised_bp"])
    out = {"start": start, "n": len(s)}
    if len(s) >= MIN_SCORED:
        e, ea = s["realised_bp"] - s["forecast_bp"], s["realised_bp"] - s["ar1_bp"]
        out.update(hit=float(s["hit"].mean()),
                   rmse=float(np.sqrt(np.mean(e ** 2))),
                   rmse_ar1=float(np.sqrt(np.mean(ea ** 2))))
    return out
