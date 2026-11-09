"""US government securities settlement calendar, and the FOMC meeting dates.

SOFR is published only for business days of the US government securities
market, whose holidays follow SIFMA's full-close recommendations. This module
encodes those rules on top of pandas' holiday machinery. Where it differs from
pandas' USFederalHolidayCalendar, and why:

  Good Friday       closed here; not a federal holiday. No SOFR has been
                    published for any Good Friday since 2019, even in years
                    such as 2026 when SIFMA recommended only an early close.
  New Year's Day    a Saturday holiday is NOT moved to the Friday before (SIFMA
  and Veterans Day  keeps that Friday open: 31 Dec 2021, 10 Nov 2023, and the
                    year-end 2027 early close). pandas moves it to Friday.
  Juneteenth        closed from 2022 only; SIFMA did not adopt it in 2021,
                    and SOFR was published on 18 June 2021.
  Ad hoc closures   days of national mourning are listed in AD_HOC. Only
                    5 Dec 2018 closed the market; 9 Jan 2025 was an early
                    close and SOFR was published.

tests/test_live.py checks this calendar against every SOFR publication date
since April 2018 and requires an exact match. Dates after the data are
SIFMA's 2026 list and, for 2027, the rules above: SIFMA's own 2027 page could
not be read in full, so 2027 rests on the rules and a third-party copy of the
list (Good Friday 26 March, Juneteenth observed 18 June, Independence Day
observed 5 July, Christmas observed 24 December). Recheck it against
https://www.sifma.org/resources/guides-playbooks/holiday-schedule when SIFMA
next updates the page.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import (MO, TH, AbstractHolidayCalendar, GoodFriday, Holiday,
                                    nearest_workday, sunday_to_monday)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset

AD_HOC = [pd.Timestamp("2018-12-05")]  # national day of mourning, President G. H. W. Bush


class SettlementCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        Holiday("Martin Luther King Jr. Day", month=1, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday("Presidents Day", month=2, day=1, offset=DateOffset(weekday=MO(3))),
        GoodFriday,
        Holiday("Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Columbus Day", month=10, day=1, offset=DateOffset(weekday=MO(2))),
        Holiday("Veterans Day", month=11, day=11, observance=sunday_to_monday),
        Holiday("Thanksgiving", month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def holidays(start="2018-01-01", end="2028-12-31") -> pd.DatetimeIndex:
    h = SettlementCalendar().holidays(start, end)
    h = h[h.day_of_week < 5]  # a Saturday New Year's or Veterans Day closes nothing
    extra = pd.DatetimeIndex(AD_HOC)
    extra = extra[(extra >= pd.Timestamp(start)) & (extra <= pd.Timestamp(end))]
    return h.union(extra).sort_values()


BDAY = CustomBusinessDay(holidays=holidays())


def is_business_day(d) -> bool:
    d = pd.Timestamp(d).normalize()
    return d.day_of_week < 5 and d not in set(holidays())


def next_business_day(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize() + BDAY


def previous_business_day(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize() - BDAY


def business_days(start, end) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=BDAY)


# FOMC meetings come from data/fomc_meetings.csv, one row per scheduled
# meeting with the Fed calendar page as its source. The decision is announced
# at 2:00 p.m. ET on the meeting's last day and takes effect on the next
# business day. Add rows there when the Fed publishes a new year.
FOMC_CSV = Path(__file__).resolve().parent.parent / "data" / "fomc_meetings.csv"


def fomc_meetings() -> pd.DataFrame:
    return pd.read_csv(FOMC_CSV, parse_dates=["meeting_start", "meeting_end"])


FOMC_DECISION_DAYS = pd.DatetimeIndex(fomc_meetings()["meeting_end"])


def fomc_effective_day(d) -> bool:
    """True if d is the first business day after a scheduled FOMC decision."""
    d = pd.Timestamp(d).normalize()
    return any(next_business_day(f) == d for f in FOMC_DECISION_DAYS)
