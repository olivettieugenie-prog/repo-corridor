"""Every user-facing string in the app, in English and French.

STRINGS[lang][key] holds one string per key. app.py reads the active language
from st.session_state["lang"] and looks every label up here, so adding or
rewording text means editing this file only. Both languages must carry the
same keys; tests/test_i18n.py checks that.

French text is written with ordinary spaces and passed through _fr_typography
once at import, which puts the no-break spaces French typesetting needs before
: ; ? ! % and between a number and its unit, so a line never breaks there.

num() formats a number for the active language: comma thousands and a decimal
point in English, a narrow no-break space for thousands and a decimal comma in
French.
"""
from __future__ import annotations

import re

LANGS = ("en", "fr")
LANG_NAMES = {"en": "English", "fr": "Français"}

NBSP = " "         # no-break space
NNBSP = " "        # narrow no-break space


def num(x: float, dec: int = 0, lang: str = "en", sign: bool = False) -> str:
    """Format x with `dec` decimals, thousands grouping and an optional sign."""
    s = f"{x:{'+' if sign else ''},.{dec}f}"
    if lang == "fr":
        s = s.replace(",", NNBSP).replace(".", ",")
    return s


def date_short(ts, lang: str = "en") -> str:
    """17 Sep 2019 / 17 sept. 2019, without depending on the system locale."""
    return f"{ts.day} {STRINGS[lang]['months'][ts.month - 1]} {ts.year}"


EN: dict[str, object] = {
    "page_title": "Repo turn monitor",
    "lang_label": "Language",
    "title": "What the next repo turn will cost",
    "intro": (
        "Forecasts of where SOFR will print against the top of the Fed's "
        "target range on the coming month-ends and quarter-ends, up to three "
        "weeks ahead, with ranges tested on past turns. Ordinary days are "
        "shown for context only: beyond a few days the model adds nothing to a "
        "simple autoregression. Public data from the New York Fed and the US "
        "Treasury."
    ),
    "bp": "bp",
    "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "weekdays": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    # dashboard step 2: short captions, tables, method, footer
    "turns_raw_median": (
        "† Raw median: calibration did not reduce its bias at the {h}-day "
        "horizon, where {above}% of past {kinds} printed above it."
    ),
    "turns_state": (
        "State: ON RRP {rrp} bn, TGA {tga} bn, 60-day median {base} bp. "
        "Calibrated quantiles at the horizon closest to the days left."
    ),
    "calc_note_short": (
        "{n} days, {k} settlement days, {t} turns; act/360; the stress case "
        "puts every day at its 90th percentile at once."
    ),
    "cap_short_calendar": (
        "SOFR minus the top of the range on each month-end, bp; Q quarter-end, "
        "Y year-end."
    ),
    "ch_turnmap": "Month-end prints, bp over the top of the range",
    "ch_next40": "Next 40 settlement days, forecast bp",
    "tt_month": "month",
    "tt_value": "bp over top",
    "tt_mark": "turn",
    "next40_ye": "year-end: no model",
    "cap_short_corridor": (
        "Daily: rates in the corridor, SOFR's position above the facility "
        "floor, ON RRP usage and SRF take-up."
    ),
    "ch_rrp_daily": "ON RRP usage, daily",
    "cap_short_model": "All scores are out of sample and walk-forward; method notes are at the bottom.",
    "cap_short_carry": (
        "The whole opportunity is {prize} bp a year; a rule with no model takes "
        "two thirds and the LASSO adds {edge} bp over AR(1). The corridor is too "
        "tight for the overnight choice to carry money."
    ),
    "tb_rmse": "Out-of-sample RMSE by year, bp",
    "naive_label": "Naive (yesterday's value)",
    "c_forecaster": "forecaster",
    "c_benchmark": "model against",
    "c_r2_bench": "R² vs benchmark",
    "g_turn_days": "turn days",
    "turns_vs_naive": (
        "Against a naive range at the {h}-day horizon, over {n} past {kinds}, "
        "the calibrated range {verdict}: pinball loss at p90 {pm} against "
        "{pn} bp for naive, with {cm}% of turns at or below p90 against {cn}% "
        "(target 90%)."
    ),
    "vs_beats": "beats it",
    "vs_loses": "does not beat it on loss but covers more",
    "vs_loses_both": "does not beat it",
    "tb_dm": "Diebold-Mariano test: model against each benchmark",
    "tb_cov": "80% interval coverage",
    "tb_turns": "Turn quantiles, raw and calibrated",
    "tb_rival": "Rival stage-two specifications",
    "c_year": "year",
    "c_period": "period",
    "c_group": "group",
    "c_n": "n",
    "c_r2": "R² vs AR(1)",
    "c_rmse": "RMSE, bp",
    "c_rmse_ar1": "RMSE AR(1), bp",
    "c_hln": "DM (HLN)",
    "c_p": "p, two-sided",
    "c_cover": "coverage",
    "c_below": "below",
    "c_above": "above",
    "c_width": "width, bp",
    "c_type": "type",
    "c_h": "horizon",
    "c_version": "version",
    "c_cov3": "coverage p50/p90/p95",
    "c_pin90": "pinball p90",
    "c_model": "stage-two model",
    "c_r2_all": "R², all",
    "c_r2_24": "R², 2024+",
    "v_raw": "raw",
    "v_cal": "calibrated",
    "g_all": "all",
    "g_me": "month-ends",
    "g_other": "other days",
    "g_qe": "quarter-ends",
    "g_me24": "month-ends, 2024+",
    "g_other24": "other days, 2024+",
    "p_undefined": "undefined",
    "method_title": "Method",
    "m_turns": "Next turns and term versus roll",
    "m_corridor": "Corridor and administered rates",
    "m_calendar": "Calendar pressure",
    "m_model": "Model, significance and intervals",
    "m_importance": "What the model uses",
    "m_carry": "The carry rule",
    "m_timing": "Data sources and publication timing",
    "timing_table": (
        "| series | source | published | used as |\n"
        "|---|---|---|---|\n"
        "| SOFR, percentiles, volume | NY Fed | about 8:00 a.m. ET next business day, revisable about 2:30 p.m. | day t |\n"
        "| EFFR and target range | NY Fed | about 9:00 a.m. ET next business day | day t |\n"
        "| ON RRP take-up and award rate | NY Fed | same day, after the 1:15 p.m. ET close | t − 1 |\n"
        "| SRF take-up | NY Fed | same day, after each operation | display only |\n"
        "| TGA closing balance | Daily Treasury Statement | 4:00 p.m. ET next business day | t − 1 |\n"
        "| Administered rates, SRF rate | FOMC implementation notes | 2:00 p.m. ET on decision day, effective next day | table |"
    ),
    "timing_note": (
        "The forecast for day D therefore exists only on the morning of D, once "
        "SOFR for D − 1 is out: it is a same-day nowcast, not a call made the "
        "evening before."
    ),
    "footer_sources": (
        "Sources: [NY Fed markets API](https://markets.newyorkfed.org/static/docs/markets-api.html) "
        "(SOFR, EFFR, ON RRP, SRF) · [Daily Treasury Statement]"
        "(https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/) (TGA) · "
        "[FOMC implementation notes and calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) "
        "· code and data: [github.com/olivettieugenie-prog/repo-corridor]"
        "(https://github.com/olivettieugenie-prog/repo-corridor)"
    ),
    "dl_title": "Download the tables as CSV",
    "dl_strip": "Market strip",
    "dl_turns": "Next turns",
    "dl_analogs": "Analogs",
    "dl_ye": "Past year-ends",
    "dl_r2": "RMSE by year",
    "dl_dm": "Diebold-Mariano",
    "dl_cov": "Interval coverage",
    "dl_turnscores": "Turn quantile scores",
    "dl_rival": "Rival specifications",
    "dl_importance": "Permutation importance",
    "dl_rules": "Carry rules",
    "dl_next40": "Next 40 days path",
    "rivals": {
        "linear, main effects": "linear, main effects",
        "linear + interactions": "linear + interactions",
        "ridge + interactions": "ridge + interactions",
        "random forest": "random forest",
        "gradient boosting": "gradient boosting",
    },
    # client note under Next turns
    "note_kind_month-end": "Month-end",
    "note_kind_quarter-end": "Quarter-end",
    "note_kind_year-end": "Year-end",
    "client_line1": (
        "{Kind} is in {days} settlement days ({date}). Against the top of the "
        "target range the model expects about {q50} bp, with 90% of outcomes "
        "at or below {q90} bp."
    ),
    "client_line2": "Past {kinds} in similar conditions printed between {a} and {b} bp.",
    "client_line1_ye": (
        "{Kind} is in {days} settlement days ({date}). There is no model for "
        "year-ends."
    ),
    "client_line2_ye": (
        "Past year-ends in similar conditions printed between {a} and {b} bp "
        "against the top of the range: history, not a forecast."
    ),
    "client_title": "Client note",
    # forward pressure calendar and track record
    "nav_forward": "Forward",
    "fwd_title": "Forward pressure calendar",
    "fwd_none": "The forward path appears after the next run of the daily job.",
    "fwd_y_top": "bp vs top of range",
    "fwd_ctx": "context, AR decay",
    "fwd_turnq": "turn: p50–p95",
    "fwd_q_tt": "p50 / p90 / p95, bp",
    "fwd_caption": (
        "Turns show the calibrated turn quantiles because the pooled 80% range "
        "held only {tmin}–{tmax}% of past turns. Ordinary days are context: "
        "their R² against AR(1) is {r1} at 1 day, {r5} at 5 and {r10} at 10, "
        "so days beyond {days} are faded. ON RRP and TGA held at {date}."
    ),
    "tt_h": "day",
    "tt_range": "80% range, bp",
    "tt_sofr": "implied SOFR",
    "tt_cov": "tested coverage",
    "tt_status": "status",
    "track_title": "Track record",
    "track_call": "Same-day call, {date}",
    "track_hit": "Hit rate {hit}% · scored days: {n} · since {start}.",
    "track_none_scored": "No forecast scored yet; the record started on {start}.",
    "track_note_title": "Morning note",
    "tr_target": "target",
    "tr_fc": "forecast, bp",
    "tr_range": "80% range",
    "tr_real": "realised, bp",
    "tr_hit": "in range",
    "yes": "yes",
    "no": "no",
    "dl_track": "Track record",
    "dl_forward": "Forward path",
    # dashboard header, links and market strip
    "hdr_asof": ("Rates to {rates} · ON RRP to {rrp} · TGA to {tga} · forecast for {target}, "
                 "run {run} (New York time)"),
    "hdr_asof_nolive": "Rates to {rates} · ON RRP to {rrp} · TGA to {tga} · no live forecast yet",
    "hdr_delay": (
        "Data delay: SOFR and EFFR publish the next business day at about 8:00 "
        "and 9:00 ET, the TGA a day later; the forecast job runs at 14:37 UTC, "
        "with a backup at 15:47."
    ),
    "about_title": "About this page",
    "nav_label": "Sections",
    "nav_turns": "Turns",
    "nav_term_roll": "Term/Roll",
    "nav_corridor": "Corridor",
    "nav_calendar": "Calendar",
    "nav_model": "Model",
    "nav_carry": "Carry",
    "nav_method": "Method",
    "strip_sofr": "SOFR",
    "strip_effr": "EFFR",
    "strip_range": "Target range",
    "strip_award": "ON RRP award rate",
    "strip_srf_rate": "SRF rate",
    "strip_rrp": "ON RRP usage",
    "strip_tga": "TGA",
    "strip_srf_take": "SRF take-up",
    "strip_bn": "bn",
    "strip_tip": "{label}: last observation {date}, change since the previous one",
    "tt_col_date": "Date",
    "tt_col_type": "Type",
    "tt_col_days": "Days",
    "tt_col_p50": "p50, bp",
    "tt_col_p90": "p90, bp",
    "tt_col_p95": "p95, bp",
    "tt_col_sofr95": "SOFR at p95",
    "tt_col_cov": "Cov. %",
    "tt_col_n": "n",
    "tt_col_h": "Horizon",
    "turns_ye_cell": "history only",
    "turns_table_note": (
        "Calibrated quantiles of SOFR minus the top of the range at the horizon "
        "closest to the settlement days left; coverage is that horizon's "
        "walk-forward record (p50/p90/p95) over n past turns."
    ),
    "analog_pick": "Analogs for",
    # next turns
    "turns_title": "Next turns",
    "turns_intro": (
        "SOFR minus the top of the target range on each turn to year-end, "
        "from a quantile model of past month-ends and quarter-ends, calibrated "
        "on its own past errors, on today's scarcity state: ON RRP {rrp} bn, "
        "TGA {tga} bn, 60-day median spread {base} bp."
    ),
    "turn_month-end": "month-end",
    "turn_quarter-end": "quarter-end",
    "turn_year-end": "year-end",
    "turns_month-end": "month-ends",
    "turns_quarter-end": "quarter-ends",
    "turns_year-end": "year-ends",
    "turn_head": "**{date}**, {kind}",
    "turn_q50": "Median",
    "turn_q90": "90th percentile",
    "turn_q95": "95th percentile",
    "turn_level": "SOFR {v}%",
    "turn_cover": (
        "Calibrated at the {h}-day horizon, the closest to the {left} "
        "settlement days left. At that horizon, on {n} past {kinds}, the "
        "median was at or above the print on {c50}% of them, the 90th "
        "percentile on {c90}% and the 95th on {c95}%."
    ),
    "turn_none": "No turn left before year-end 2026.",
    "turn_ye": (
        "No model: only {n} year-ends since 2018, too few to fit. The analogs "
        "below are history only."
    ),
    "analogs_title": "Closest past {kinds} by scarcity state",
    "an_date": "date",
    "an_rrp": "ON RRP, bn",
    "an_tga": "TGA, bn",
    "an_base": "median spread, bp",
    "an_dist": "distance",
    "an_print": "printed, bp",
    "an_p99": "99th pct, bp",
    "analogs_note": (
        "Printed is SOFR minus the top of the range that day; 99th pct is the "
        "99th percentile of that day's SOFR trades, same units. Distance is in "
        "standard deviations of past daily data."
    ),

    # term versus roll
    "calc_title": "Term or roll?",
    "calc_start": "Start",
    "calc_end": "End",
    "calc_quote": "Term repo quote, %",
    "calc_central": "Expected roll",
    "calc_p90": "Stress case (p90)",
    "calc_be": "Roll minus quote",
    "calc_be_p90": "Stress case minus quote",
    "calc_term": (
        "Terming out at {q}% is cheaper than the expected roll by {be} bp, and "
        "{be90} bp cheaper than the stress case."
    ),
    "calc_insure": (
        "Term at {q}% costs {be} bp more than the expected roll but {be90} bp "
        "less than the stress case: it is insurance against the turn."
    ),
    "calc_roll": "Rolling is cheaper even in the stress case, by {be90} bp.",
    "calc_fomc": (
        "The window touches the FOMC decision of {dates}. It cannot be priced "
        "without a policy path, so no verdict is given."
    ),
    "calc_ye_title": "Year-end of {date}: history, not a forecast",
    "calc_ye_stats": (
        "Past year-ends printed from {min} to {max} bp over the top of the "
        "range, median {med} bp ({n} years)."
    ),
    "calc_ye_be": (
        "With the year-end's {d} days included, the quote breaks even if the "
        "year-end prints at {be} bp over the top of the range."
    ),
    "calc_ye_rest": (
        "The figures below price the rest of the window, without the "
        "year-end; no verdict is given."
    ),
    "ye_year": "year",
    "ye_print": "printed, bp",
    "calc_early": "The first day that can be priced is {date}; move the start there or later.",
    "calc_empty": "The end must come after the start.",
    "calc_note": (
        "{n} calendar days over {k} settlement days (turns among them: {t}), act/360 "
        "on both legs. Ordinary days follow the two-stage forecast and its "
        "AR(1) decay; turns use the median and 90th percentile above. The "
        "stress case puts every day at its 90th percentile at once, as if "
        "every day were bad together: an upper reference, not the 90th "
        "percentile of the average."
    ),

    # desk view
    "desk_target": "Target date: **{target}**",
    "desk_forecast": "Deviation forecast",
    "desk_range": "80% range: {lo} to {hi} bp",
    "desk_sofr": "Implied SOFR",
    "desk_sofr_range": "80% range: {lo} to {hi}%",
    "desk_flags": "Calendar flags in play: {flags}",
    "desk_no_flags": "none",
    "desk_level": "{v} bn",
    "desk_level_help": "Level on {date}, with its change from the previous day",
    "flag_month_end": "month-end",
    "flag_quarter_end": "quarter-end",
    "flag_year_end": "year-end",
    "flag_mid_settle": "mid-month settlement",
    "flag_month_start": "month start",
    "flag_rate_change": "target-range change",
    "desk_asof": (
        "Run {run} New York time, on data to {asof}. Two-stage LASSO on the "
        "latest 500 days; the range comes from the last 250 forecast errors."
    ),
    "desk_stale": (
        "This forecast is out of date: it targets {target}, more than one "
        "business day ago. The daily job may have failed."
    ),
    "desk_unconfirmed": (
        "{target} follows an FOMC decision that is not yet in the "
        "administered-rate table, so the implied SOFR assumes an unchanged "
        "target range."
    ),
    "note_line1": (
        "For {target}, the model puts SOFR {abs} bp {direction} its 60-day "
        "median position in the corridor, with an 80% range of {lo} to {hi} bp; "
        "that implies SOFR at {sofr}%."
    ),
    "note_flat": (
        "For {target}, the model puts SOFR in line with its 60-day median "
        "position in the corridor ({f} bp, 80% range {lo} to {hi} bp); that "
        "implies SOFR at {sofr}%."
    ),
    "dir_above": "above",
    "dir_below": "below",
    "note_line2_flags": (
        "In play: {flags}. ON RRP stands at {rrp} bn ({drrp} bn on the day) "
        "and the TGA at {tga} bn ({dtga} bn)."
    ),
    "note_line2_none": (
        "No calendar effect in play. ON RRP stands at {rrp} bn ({drrp} bn on "
        "the day) and the TGA at {tga} bn ({dtga} bn)."
    ),
    "desk_track": (
        "Live track record, {n} scored days since {start}: the 80% range held "
        "the outcome on {hit}% of them, and the RMSE is {rmse} bp against "
        "{rmse_ar1} bp for AR(1) on the same days."
    ),
    "desk_track_pending": (
        "Live track record started on {start}; {n} of 20 scored days so far. "
        "The hit rate and the RMSE against AR(1) appear from 20 days."
    ),
    "desk_src_local": "The data branch on GitHub could not be reached; showing the local copy.",
    "desk_none": (
        "No live forecast is available yet. It appears after the first run of "
        "the daily workflow."
    ),

    # stat tiles
    "tile_days": "SOFR days covered",
    "tile_days_help": "2 April 2018 to {last}, from the NY Fed API",
    "tile_skill": "Model skill vs AR(1), 2024+",
    "tile_skill_help": (
        "Out-of-sample R squared of the two-stage LASSO against an "
        "autoregression, rolling 500-day training window. Diebold-Mariano "
        "test: significant since 2024 (p = {p24}), not over the full period "
        "from May 2020 (R² {r2all}, p = {pall})."
    ),
    "tile_carry": "Carry from always lending",
    "tile_carry_help": (
        "Mean annualised pickup of overnight repo over the ON RRP facility, "
        "on the {days} out-of-sample decision days from May 2020"
    ),
    "tile_peak": "Carry on 17 Sep 2019",
    "tile_peak_help": (
        "The day the plumbing broke: SOFR printed 5.25% while the Fed's "
        "facility paid 2.00%. Normal range is about -10 to +30 bp."
    ),

    # 1. corridor
    "s1": "The corridor",
    "date_range": "Date range",
    "slider_format": "MMM YYYY",
    "series_sofr": "SOFR",
    "series_award": "ON RRP award rate",
    "series_srf": "SRF rate",
    "ch_srf": "Standing Repo Facility take-up",
    "ax_srf": "USD bn",
    "tt_srf": "SRF take-up, bn",
    "cap_srf": (
        "The dashed line is the Standing Repo Facility (SRF) rate, from the "
        "Fed's implementation notes, 43 since July 2021, each linked in "
        "data/srf_rates.csv: a minimum bid rate until October 2025 and a fixed "
        "full-allotment rate since December 2025, always at the top of the "
        "target range. The bars are daily take-up from the New York Fed "
        "markets API."
    ),
    "ax_percent": "percent",
    "ax_position": "SOFR − award, bp (clip {clip})",
    "tt_date": "date",
    "tt_series": "series",
    "tt_percent": "percent",
    "tt_carry": "carry, bp",
    "ch_band": "Shaded band is the FOMC target range",
    "ch_position": "Where SOFR sits above the facility floor",
    "cap1": (
        "The administered rates come from a table of 15 Federal Reserve "
        "implementation notes, each cited in the repository, because the "
        "offsets are not stable. ON RRP sat at the bottom of the range until "
        "September 2019, 5 bp below it through to March 2020, 5 bp above it "
        "from June 2021, and back at the bottom from 19 December 2024. The "
        "table reproduces all 491 directly observed award rates."
    ),

    # 2. calendar
    "s2": "Calendar pressure only bites when reserves are scarce",
    "ax_bp": "basis points",
    "ax_rrp": "USD bn, daily average",
    "tt_year": "year",
    "tt_premium": "premium, bp",
    "tt_rrp": "ON RRP, $bn",
    "ch_premium": "Month-end premium on the spread",
    "ch_rrp": "ON RRP usage",
    "cap2": (
        "Two charts rather than two axes on one. The month-end premium is "
        "about 9 to 13 bp in 2018 and 2019, essentially zero from 2020 to 2022 "
        "when ON RRP usage ran near $2.0tn, and returns as the facility "
        "drains. With the system awash with cash, month-end balance-sheet "
        "constraints had nothing to push against."
    ),

    # 3. skill
    "s3": "Does the model add anything to an autoregression?",
    "ax_r2": "out-of-sample R² vs AR(1)",
    "tt_r2": "R² vs AR(1)",
    "ch_r2": "Positive means the LASSO beat the autoregression",
    "cap3": (
        "Zero is a weak benchmark for this target and is not used. The "
        "trailing median removes the level of the spread but not a slow trend "
        "in it, so the deviation stays autocorrelated at 0.81 and an "
        "autoregression alone reaches R² 0.64 against zero. The bar direction "
        "carries the sign, so colour is not the only cue. The model loses in "
        "2020 and 2021 because the rolling window has to pass through the "
        "regime break before it adapts."
    ),
    "ch_fan": "Forecast, 80% interval and outcome",
    "ax_dev": "deviation, bp",
    "fan_inside": "inside the interval",
    "fan_outside": "outside",
    "tt_outcome": "outcome, bp",
    "tt_forecast": "forecast, bp",
    "tt_band": "80% interval, bp",
    "cap_fan": (
        "The last 250 days. The line is the two-stage forecast of the next day's deviation; the "
        "band runs from the 10th to the 90th percentile of the previous 250 "
        "forecast errors only, so it never sees the day it covers. Since May "
        "2021 it holds the outcome on {all}% of days, {other}% outside "
        "month-ends, but only {me}% on month-ends and {qe}% on quarter-ends, "
        "where errors are larger than the pooled errors suggest."
    ),
    "cap3_dm": (
        "Is the gain significant? A Diebold-Mariano test on squared errors, "
        "with a HAC variance, finds it significant from 2024 onwards "
        "(p = {p24}) but not over the whole period from May 2020 (p = {pall}): "
        "the losses of 2020 offset the later gains."
    ),

    # 3b. importance
    "s3b": "What is the model actually using?",
    "kind_one": "one at a time",
    "kind_group": "shuffled as a group",
    "ax_rmse": "increase in out-of-sample RMSE, bp",
    "tt_predictor": "predictor",
    "tt_rmse": "RMSE increase, bp",
    "tt_p10": "10th percentile, bp",
    "tt_p90": "90th percentile, bp",
    "ch_importance": "Scramble a predictor and see how much worse the forecast gets",
    # Group names from permutation_importance.csv. Single predictors keep
    # their variable names, which are the same in both languages.
    "groups": {
        "calendar, all five": "calendar, all five",
        "month/quarter/year nest": "month/quarter/year nest",
        "dispersion pair": "dispersion pair",
        "rate-change pair": "rate-change pair",
        "balance sheet, all four": "balance sheet, all four",
    },
    "cap3b": (
        "Shuffled in blocks of 21 days so the serial structure survives, 30 "
        "times each; whiskers span the 10th to 90th percentile of those "
        "repeats. Correlated predictors are also shuffled together, because "
        "the calendar flags are nested: every quarter-end is a month-end and "
        "every year-end is a quarter-end, so one at a time they cover for each "
        "other. The four balance-sheet variables contribute nothing even "
        "jointly, at −0.005 bp. Negative means the shuffle happened to help, "
        "which is noise."
    ),

    # 4. money
    "s4": "Is the skill worth any money? No",
    "s4_intro": (
        "Place tomorrow's overnight cash in repo at SOFR or at the facility at "
        "the award rate, decided the day before. The carry splits exactly into "
        "`base + deviation + corridor`, and base and corridor are both known a "
        "day ahead. So the benchmark that matters is a rule given only the "
        "known part, not passive always-lend, which would credit the model for "
        "the corridor."
    ),
    "col_rule": "rule",
    "col_net": "net bp",
    "col_vs_always": "vs always-lend",
    "col_vs_known": "vs known-only",
    "rule_always": "always lend",
    "rule_known": "known only, no model",
    "rule_ar1": "AR(1)",
    "rule_lasso": "two-stage LASSO",
    "rule_perfect": "perfect foresight",
    "conclusion": (
        "The entire opportunity is **{prize} bp** of annualised yield. A rule "
        "with no model captures two thirds of it, and the LASSO's margin over "
        "an autoregression is {edge} bp, which is nothing. At a 1 bp switching "
        "cost the LASSO falls behind AR(1), {lasso1} against {ar1_1}. The Fed's "
        "two facilities bound the corridor so tightly that the overnight "
        "placement decision has almost no money in it. In 2022 always-lend "
        "gave up 1.55 bp against the facility and every smarter rule merely "
        "returns to zero, which is why about $2.0tn sat at ON RRP: not "
        "chasing carry, but because the facility simply paid more."
    ),
    "sources": (
        "Sources: SOFR, EFFR with the FOMC target range, and ON RRP operations "
        "from the Federal Reserve Bank of New York markets API. Treasury "
        "General Account from the US Treasury Fiscal Data API. "
        "Administered-rate offsets read from Federal Reserve implementation "
        "notes. Built with pandas and scikit-learn; no paid data. Limitations "
        "and the extensions left out are in the repository README."
    ),
}

FR: dict[str, object] = {
    "page_title": "Moniteur des fins de mois repo",
    "lang_label": "Langue",
    "title": "Ce que coûtera le prochain passage de fin de mois sur le repo",
    "intro": (
        "Prévisions du niveau du SOFR par rapport au haut de la fourchette "
        "cible de la Fed pour les prochaines fins de mois et de trimestre, "
        "jusqu'à trois semaines à l'avance, avec des fourchettes testées sur "
        "les passages passés. Les jours ordinaires sont affichés pour "
        "contexte : au-delà de quelques jours, le modèle n'apporte rien de plus "
        "qu'une simple autorégression. Données publiques de la Fed de New York "
        "et du Trésor américain."
    ),
    "bp": "pb",
    "months": ["janv.", "févr.", "mars", "avr.", "mai", "juin",
               "juil.", "août", "sept.", "oct.", "nov.", "déc."],
    "weekdays": ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."],
    "turns_raw_median": (
        "† Médiane brute : la calibration n'a pas réduit son biais à l'horizon "
        "de {h} jours, où {above} % des {kinds} passées se sont fixées au-dessus."
    ),
    "turns_state": (
        "État : ON RRP {rrp} Md, TGA {tga} Md, médiane sur 60 jours {base} pb. "
        "Quantiles calibrés à l'horizon le plus proche des jours restants."
    ),
    "calc_note_short": (
        "{n} jours, {k} jours de règlement, {t} échéances ; exact/360 ; le "
        "stress place chaque jour à son 90e centile en même temps."
    ),
    "cap_short_calendar": (
        "SOFR moins le haut de la fourchette à chaque fin de mois, en pb ; Q fin "
        "de trimestre, Y fin d'année."
    ),
    "ch_turnmap": "Fins de mois, pb au-dessus du haut de la fourchette",
    "ch_next40": "40 prochains jours de règlement, prévision en pb",
    "tt_month": "mois",
    "tt_value": "pb vs haut",
    "tt_mark": "échéance",
    "next40_ye": "fin d'année : pas de modèle",
    "cap_short_corridor": (
        "Quotidien : taux dans le corridor, position du SOFR au-dessus du "
        "plancher, recours à l'ON RRP et à la SRF."
    ),
    "ch_rrp_daily": "Recours à l'ON RRP, quotidien",
    "cap_short_model": "Tous les scores sont hors échantillon et glissants ; les notes de méthode figurent en bas de page.",
    "cap_short_carry": (
        "L'opportunité totale est de {prize} pb par an ; une règle sans modèle en "
        "capte les deux tiers et le LASSO ajoute {edge} pb à l'AR(1). Le corridor "
        "est trop étroit pour que le choix au jour le jour rapporte."
    ),
    "tb_rmse": "RMSE hors échantillon par année, pb",
    "naive_label": "Naïf (valeur de la veille)",
    "c_forecaster": "prévision",
    "c_benchmark": "modèle face à",
    "c_r2_bench": "R² vs référence",
    "g_turn_days": "jours d'échéance",
    "turns_vs_naive": (
        "Face à un intervalle naïf à l'horizon de {h} jours, sur {n} {kinds} "
        "passées, l'intervalle calibré {verdict} : perte pinball au p90 de {pm} "
        "contre {pn} pb pour le naïf, avec {cm} % des échéances au niveau du p90 "
        "ou en dessous contre {cn} % (cible 90 %)."
    ),
    "vs_beats": "fait mieux",
    "vs_loses": "fait moins bien en perte mais couvre davantage",
    "vs_loses_both": "fait moins bien",
    "tb_dm": "Test de Diebold-Mariano : modèle face à chaque référence",
    "tb_cov": "Couverture de l'intervalle à 80 %",
    "tb_turns": "Quantiles d'échéance, bruts et calibrés",
    "tb_rival": "Spécifications concurrentes de la deuxième étape",
    "c_year": "année",
    "c_period": "période",
    "c_group": "groupe",
    "c_n": "n",
    "c_r2": "R² face à l'AR(1)",
    "c_rmse": "RMSE, pb",
    "c_rmse_ar1": "RMSE AR(1), pb",
    "c_hln": "DM (HLN)",
    "c_p": "p, bilatéral",
    "c_cover": "couverture",
    "c_below": "en dessous",
    "c_above": "au-dessus",
    "c_width": "largeur, pb",
    "c_type": "type",
    "c_h": "horizon",
    "c_version": "version",
    "c_cov3": "couverture p50/p90/p95",
    "c_pin90": "perte pinball p90",
    "c_model": "modèle de deuxième étape",
    "c_r2_all": "R², ensemble",
    "c_r2_24": "R², 2024+",
    "v_raw": "brut",
    "v_cal": "calibré",
    "g_all": "ensemble",
    "g_me": "fins de mois",
    "g_other": "autres jours",
    "g_qe": "fins de trimestre",
    "g_me24": "fins de mois, 2024+",
    "g_other24": "autres jours, 2024+",
    "p_undefined": "non défini",
    "method_title": "Méthode",
    "m_turns": "Prochaines échéances et terme ou renouvellement",
    "m_corridor": "Corridor et taux administrés",
    "m_calendar": "Effets de calendrier",
    "m_model": "Modèle, significativité et intervalles",
    "m_importance": "Ce qu'utilise le modèle",
    "m_carry": "La règle de portage",
    "m_timing": "Sources et horaires de publication",
    "timing_table": (
        "| série | source | publication | utilisée comme |\n"
        "|---|---|---|---|\n"
        "| SOFR, centiles, volume | Fed de New York | vers 8 h (New York) le jour ouvré suivant, révisable vers 14 h 30 | jour t |\n"
        "| EFFR et fourchette cible | Fed de New York | vers 9 h (New York) le jour ouvré suivant | jour t |\n"
        "| Recours et taux servi ON RRP | Fed de New York | le jour même, après la clôture de 13 h 15 | t − 1 |\n"
        "| Recours à la SRF | Fed de New York | le jour même, après chaque opération | affichage seul |\n"
        "| Solde de clôture du TGA | Daily Treasury Statement | 16 h (New York) le jour ouvré suivant | t − 1 |\n"
        "| Taux administrés, taux SRF | notes de mise en œuvre du FOMC | 14 h le jour de la décision, effectif le lendemain | tableau |"
    ),
    "timing_note": (
        "La prévision du jour D n'existe donc que le matin de D, une fois le "
        "SOFR de D − 1 publié : c'est une estimation du jour même, pas une "
        "décision prise la veille au soir."
    ),
    "footer_sources": (
        "Sources : [API des marchés de la Fed de New York](https://markets.newyorkfed.org/static/docs/markets-api.html) "
        "(SOFR, EFFR, ON RRP, SRF) · [Daily Treasury Statement]"
        "(https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/) (TGA) · "
        "[notes de mise en œuvre et calendrier du FOMC](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) "
        "· code et données : [github.com/olivettieugenie-prog/repo-corridor]"
        "(https://github.com/olivettieugenie-prog/repo-corridor)"
    ),
    "dl_title": "Télécharger les tableaux en CSV",
    "dl_strip": "Bandeau de marché",
    "dl_turns": "Prochaines échéances",
    "dl_analogs": "Analogues",
    "dl_ye": "Fins d'année passées",
    "dl_r2": "RMSE par année",
    "dl_dm": "Diebold-Mariano",
    "dl_cov": "Couverture de l'intervalle",
    "dl_turnscores": "Scores des quantiles d'échéance",
    "dl_rival": "Spécifications concurrentes",
    "dl_importance": "Importance par permutation",
    "dl_rules": "Règles de portage",
    "dl_next40": "Trajectoire des 40 jours",
    "rivals": {
        "linear, main effects": "linéaire, effets principaux",
        "linear + interactions": "linéaire + interactions",
        "ridge + interactions": "ridge + interactions",
        "random forest": "forêt aléatoire",
        "gradient boosting": "gradient boosting",
    },
    "note_kind_month-end": "La fin de mois",
    "note_kind_quarter-end": "La fin de trimestre",
    "note_kind_year-end": "La fin d'année",
    "client_line1": (
        "{Kind} tombe dans {days} jours de règlement ({date}). Par rapport au "
        "haut de la fourchette cible, le modèle anticipe environ {q50} pb, avec "
        "90 % des réalisations attendues à {q90} pb ou moins."
    ),
    "client_line2": (
        "Les {kinds} passées dans des conditions comparables se sont fixées "
        "entre {a} et {b} pb."
    ),
    "client_line1_ye": (
        "{Kind} tombe dans {days} jours de règlement ({date}). Il n'existe pas "
        "de modèle pour les fins d'année."
    ),
    "client_line2_ye": (
        "Les fins d'année passées dans des conditions comparables se sont "
        "fixées entre {a} et {b} pb par rapport au haut de la fourchette : "
        "historique, pas une prévision."
    ),
    "client_title": "Note client",
    "nav_forward": "À venir",
    "fwd_title": "Calendrier de pression à venir",
    "fwd_none": "La trajectoire apparaîtra après la prochaine exécution de la tâche quotidienne.",
    "fwd_y_top": "pb vs haut de la fourchette",
    "fwd_ctx": "contexte, décroissance AR",
    "fwd_turnq": "échéance : p50–p95",
    "fwd_q_tt": "p50 / p90 / p95, pb",
    "fwd_caption": (
        "Les échéances portent les quantiles calibrés, car l'intervalle commun à "
        "80 % n'a contenu que {tmin}–{tmax} % des échéances passées. Les jours "
        "ordinaires sont du contexte : leur R² face à l'AR(1) vaut {r1} à "
        "1 jour, {r5} à 5 et {r10} à 10, d'où l'estompage au-delà de {days} "
        "jours. ON RRP et TGA figés au {date}."
    ),
    "tt_h": "jour",
    "tt_range": "intervalle à 80 %, pb",
    "tt_sofr": "SOFR implicite",
    "tt_cov": "couverture testée",
    "tt_status": "statut",
    "track_title": "Historique",
    "track_call": "Prévision du jour, {date}",
    "track_hit": "Taux de réussite {hit} % · jours évalués : {n} · depuis le {start}.",
    "track_none_scored": "Aucune prévision évaluée pour l'instant ; l'historique a commencé le {start}.",
    "track_note_title": "Note du matin",
    "tr_target": "cible",
    "tr_fc": "prévision, pb",
    "tr_range": "intervalle à 80 %",
    "tr_real": "réalisé, pb",
    "tr_hit": "dans l'intervalle",
    "yes": "oui",
    "no": "non",
    "dl_track": "Historique",
    "dl_forward": "Trajectoire à venir",
    "hdr_asof": ("Taux au {rates} · ON RRP au {rrp} · TGA au {tga} · prévision pour le {target}, "
                 "calculée le {run} (heure de New York)"),
    "hdr_asof_nolive": ("Taux au {rates} · ON RRP au {rrp} · TGA au {tga} · pas encore de "
                        "prévision en conditions réelles"),
    "hdr_delay": (
        "Délai des données : SOFR et EFFR sont publiés le jour ouvré suivant "
        "vers 8 h et 9 h (heure de New York), le TGA un jour plus tard ; la "
        "tâche de prévision tourne à 14 h 37 UTC, avec une reprise à 15 h 47."
    ),
    "about_title": "À propos de cette page",
    "nav_label": "Sections",
    "nav_turns": "Échéances",
    "nav_term_roll": "Terme/JJ",
    "nav_corridor": "Corridor",
    "nav_calendar": "Calendrier",
    "nav_model": "Modèle",
    "nav_carry": "Portage",
    "nav_method": "Méthode",
    "strip_sofr": "SOFR",
    "strip_effr": "EFFR",
    "strip_range": "Fourchette cible",
    "strip_award": "Taux servi ON RRP",
    "strip_srf_rate": "Taux SRF",
    "strip_rrp": "Recours ON RRP",
    "strip_tga": "TGA",
    "strip_srf_take": "Recours SRF",
    "strip_bn": "Md",
    "strip_tip": "{label} : dernière observation le {date}, variation sur la précédente",
    "tt_col_date": "Date",
    "tt_col_type": "Type",
    "tt_col_days": "Jours",
    "tt_col_p50": "p50, pb",
    "tt_col_p90": "p90, pb",
    "tt_col_p95": "p95, pb",
    "tt_col_sofr95": "SOFR au p95",
    "tt_col_cov": "Couv. %",
    "tt_col_n": "n",
    "tt_col_h": "Horizon",
    "turns_ye_cell": "historique seul",
    "turns_table_note": (
        "Quantiles calibrés de SOFR moins le haut de la fourchette, à l'horizon "
        "le plus proche des jours de règlement restants ; la couverture est "
        "l'historique hors échantillon de cet horizon (p50/p90/p95) sur n échéances passées."
    ),
    "analog_pick": "Analogues pour",
    "turns_title": "Prochaines échéances",
    "turns_intro": (
        "SOFR moins le haut de la fourchette cible à chaque échéance d'ici la "
        "fin d'année, selon un modèle de quantiles estimé sur les fins de mois "
        "et de trimestre passées, calibré sur ses propres erreurs passées, à "
        "l'état de rareté actuel : ON RRP {rrp} Md USD, TGA {tga} Md USD, "
        "écart médian sur 60 jours {base} pb."
    ),
    "turn_month-end": "fin de mois",
    "turn_quarter-end": "fin de trimestre",
    "turn_year-end": "fin d'année",
    "turns_month-end": "fins de mois",
    "turns_quarter-end": "fins de trimestre",
    "turns_year-end": "fins d'année",
    "turn_head": "**{date}**, {kind}",
    "turn_q50": "Médiane",
    "turn_q90": "90e centile",
    "turn_q95": "95e centile",
    "turn_level": "SOFR {v} %",
    "turn_cover": (
        "Calibré à l'horizon de {h} jours, le plus proche des {left} jours de "
        "règlement restants. À cet horizon, sur {n} {kinds} passées, la "
        "médiane était au-dessus ou au niveau du taux fixé dans {c50} % des "
        "cas, le 90e centile dans {c90} % et le 95e dans {c95} %."
    ),
    "turn_none": "Plus aucune échéance avant la fin de l'année 2026.",
    "turn_ye": (
        "Pas de modèle : seulement {n} fins d'année depuis 2018, trop peu pour "
        "une estimation. Les analogues ci-dessous sont donnés à titre "
        "historique uniquement."
    ),
    "analogs_title": "Analogues historiques les plus proches par état de rareté ({kinds})",
    "an_date": "date",
    "an_rrp": "ON RRP, Md",
    "an_tga": "TGA, Md",
    "an_base": "écart médian, pb",
    "an_dist": "distance",
    "an_print": "fixé, pb",
    "an_p99": "99e centile, pb",
    "analogs_note": (
        "Fixé : SOFR moins le haut de la fourchette ce jour-là ; 99e centile : "
        "celui des transactions SOFR du jour, dans la même unité. La distance "
        "est exprimée en écarts-types des données quotidiennes passées."
    ),

    "calc_title": "Terme ou renouvellement au jour le jour ?",
    "calc_start": "Début",
    "calc_end": "Fin",
    "calc_quote": "Cotation de pension à terme, %",
    "calc_central": "Renouvellement attendu",
    "calc_p90": "Stress (90e centile)",
    "calc_be": "Écart à la cotation",
    "calc_be_p90": "Écart en stress",
    "calc_term": (
        "Emprunter à terme à {q} % coûte {be} pb de moins que le "
        "renouvellement attendu, et {be90} pb de moins que le scénario de "
        "stress."
    ),
    "calc_insure": (
        "Le terme à {q} % coûte {be} pb de plus que le renouvellement attendu, "
        "mais {be90} pb de moins que le scénario de stress : c'est une "
        "assurance contre l'échéance."
    ),
    "calc_roll": "Le renouvellement au jour le jour reste moins cher même en stress, de {be90} pb.",
    "calc_fomc": (
        "La fenêtre touche la décision du FOMC du {dates}. Elle ne peut pas "
        "être évaluée sans trajectoire de politique monétaire ; aucune "
        "conclusion n'est donnée."
    ),
    "calc_ye_title": "Fin d'année du {date} : historique, pas une prévision",
    "calc_ye_stats": (
        "Les fins d'année passées se sont fixées entre {min} et {max} pb "
        "au-dessus du haut de la fourchette, médiane {med} pb ({n} années)."
    ),
    "calc_ye_be": (
        "En incluant les {d} jours de la fin d'année, la cotation est à "
        "l'équilibre si la fin d'année se fixe à {be} pb au-dessus du haut de "
        "la fourchette."
    ),
    "calc_ye_rest": (
        "Les chiffres ci-dessous évaluent le reste de la fenêtre, hors fin "
        "d'année ; aucune conclusion n'est donnée."
    ),
    "ye_year": "année",
    "ye_print": "fixé, pb",
    "calc_early": "Le premier jour évaluable est le {date} ; décaler le début à cette date ou après.",
    "calc_empty": "La fin doit suivre le début.",
    "calc_note": (
        "{n} jours calendaires sur {k} jours de règlement (échéances : {t}), "
        "base exact/360 des deux côtés. Les jours ordinaires "
        "suivent la prévision en deux étapes et la décroissance de son AR(1) ; "
        "les échéances utilisent la médiane et le 90e centile ci-dessus. Le "
        "scénario de stress place chaque jour à son 90e centile en même temps, "
        "comme si tous les jours étaient défavorables à la fois : c'est une "
        "borne de référence, pas le 90e centile de la moyenne."
    ),

    "desk_target": "Date cible : **{target}**",
    "desk_forecast": "Déviation prévue",
    "desk_range": "Intervalle à 80 % : {lo} à {hi} pb",
    "desk_sofr": "SOFR implicite",
    "desk_sofr_range": "Intervalle à 80 % : {lo} à {hi} %",
    "desk_flags": "Effets de calendrier en jeu : {flags}",
    "desk_no_flags": "aucun",
    "desk_level": "{v} Md USD",
    "desk_level_help": "Niveau au {date}, avec sa variation sur la veille",
    "flag_month_end": "fin de mois",
    "flag_quarter_end": "fin de trimestre",
    "flag_year_end": "fin d'année",
    "flag_mid_settle": "règlement de mi-mois",
    "flag_month_start": "début de mois",
    "flag_rate_change": "changement de fourchette cible",
    "desk_asof": (
        "Calcul du {run}, heure de New York, sur les données au {asof}. LASSO "
        "en deux étapes sur les 500 derniers jours ; l'intervalle provient des "
        "250 dernières erreurs de prévision."
    ),
    "desk_stale": (
        "Cette prévision n'est plus à jour : elle porte sur le {target}, il y a "
        "plus d'un jour ouvré. La tâche quotidienne a peut-être échoué."
    ),
    "desk_unconfirmed": (
        "Le {target} suit une décision du FOMC qui ne figure pas encore dans "
        "le tableau des taux administrés ; le SOFR implicite suppose une "
        "fourchette cible inchangée."
    ),
    "note_line1": (
        "Pour le {target}, le modèle situe le SOFR {abs} pb {direction} de sa "
        "position médiane sur 60 jours dans le corridor, avec un intervalle à "
        "80 % de {lo} à {hi} pb, soit un SOFR implicite de {sofr} %."
    ),
    "note_flat": (
        "Pour le {target}, le modèle situe le SOFR en ligne avec sa position "
        "médiane sur 60 jours dans le corridor ({f} pb, intervalle à 80 % de "
        "{lo} à {hi} pb), soit un SOFR implicite de {sofr} %."
    ),
    "dir_above": "au-dessus",
    "dir_below": "en dessous",
    "note_line2_flags": (
        "En jeu : {flags}. L'ON RRP s'établit à {rrp} Md USD ({drrp} Md sur la "
        "journée) et le TGA à {tga} Md USD ({dtga} Md)."
    ),
    "note_line2_none": (
        "Aucun effet de calendrier en jeu. L'ON RRP s'établit à {rrp} Md USD "
        "({drrp} Md sur la journée) et le TGA à {tga} Md USD ({dtga} Md)."
    ),
    "desk_track": (
        "Historique en conditions réelles, {n} jours évalués depuis le "
        "{start} : l'intervalle à 80 % a contenu la réalisation {hit} % du "
        "temps, pour une RMSE de {rmse} pb contre {rmse_ar1} pb pour l'AR(1) "
        "sur les mêmes jours."
    ),
    "desk_track_pending": (
        "Historique en conditions réelles commencé le {start} ; {n} jours "
        "évalués sur 20. Le taux de couverture et la RMSE face à l'AR(1) "
        "s'afficheront à partir de 20 jours."
    ),
    "desk_src_local": "La branche data sur GitHub est injoignable ; affichage de la copie locale.",
    "desk_none": (
        "Aucune prévision en conditions réelles n'est encore disponible. Elle "
        "apparaîtra après la première exécution de la tâche quotidienne."
    ),

    "tile_days": "Jours de SOFR couverts",
    "tile_days_help": "Du 2 avril 2018 au {last}, via l'API de la Fed de New York",
    "tile_skill": "Apport du modèle face à l'AR(1), 2024+",
    "tile_skill_help": (
        "R² hors échantillon du LASSO en deux étapes par rapport à un modèle "
        "autorégressif, fenêtre d'estimation glissante de 500 jours. Test de "
        "Diebold-Mariano : gain significatif depuis 2024 (p = {p24}), mais pas "
        "sur l'ensemble de la période depuis mai 2020 (R² {r2all}, p = {pall})."
    ),
    "tile_carry": "Portage du prêt systématique",
    "tile_carry_help": (
        "Surplus annualisé moyen du repo au jour le jour sur la facilité "
        "ON RRP, sur les {days} jours de décision hors échantillon depuis mai 2020"
    ),
    "tile_peak": "Portage le 17 sept. 2019",
    "tile_peak_help": (
        "Le jour où la mécanique du marché a cédé : le SOFR s'est fixé à "
        "5,25 % alors que la facilité de la Fed rémunérait à 2,00 %. La plage "
        "habituelle va d'environ -10 à +30 pb."
    ),

    "s1": "Le corridor",
    "date_range": "Période",
    "slider_format": "MM/YYYY",
    "series_sofr": "SOFR",
    "series_award": "Taux servi ON RRP",
    "series_srf": "Taux SRF",
    "ch_srf": "Recours à la facilité permanente de pension",
    "ax_srf": "Md USD",
    "tt_srf": "recours à la SRF, Md",
    "cap_srf": (
        "La ligne en tirets est le taux de la facilité permanente de prise en "
        "pension (Standing Repo Facility, SRF), tiré des notes de mise en œuvre "
        "de la Fed, 43 depuis juillet 2021, chacune référencée dans "
        "data/srf_rates.csv : un taux minimal de soumission jusqu'en octobre "
        "2025, puis un taux fixe à allocation intégrale depuis décembre 2025, "
        "toujours égal au haut de la fourchette cible. Les barres donnent le "
        "recours quotidien, issu de l'API des marchés de la Fed de New York."
    ),
    "ax_percent": "en %",
    "ax_position": "SOFR − taux servi, pb (écrêté {clip})",
    "tt_date": "date",
    "tt_series": "série",
    "tt_percent": "en %",
    "tt_carry": "portage, pb",
    "ch_band": "Zone ombrée : fourchette cible du FOMC",
    "ch_position": "Le SOFR au-dessus du plancher de la facilité",
    "cap1": (
        "Les taux administrés proviennent d'un tableau de 15 notes de mise en "
        "œuvre de la Réserve fédérale, chacune citée dans le dépôt, car les "
        "écarts ne sont pas stables. L'ON RRP se situait au bas de la "
        "fourchette jusqu'en septembre 2019, 5 pb en dessous jusqu'en mars "
        "2020, 5 pb au-dessus à partir de juin 2021, puis de nouveau au bas de "
        "la fourchette depuis le 19 décembre 2024. Le tableau reproduit les "
        "491 taux servis observés directement."
    ),

    "s2": "Les effets de calendrier ne jouent que lorsque les réserves sont rares",
    "ax_bp": "points de base",
    "ax_rrp": "Md USD, moyenne quotidienne",
    "tt_year": "année",
    "tt_premium": "prime, pb",
    "tt_rrp": "ON RRP, Md USD",
    "ch_premium": "Prime de fin de mois sur l'écart",
    "ch_rrp": "Recours à l'ON RRP",
    "cap2": (
        "Deux graphiques plutôt que deux axes sur un seul. La prime de fin de "
        "mois avoisine 9 à 13 pb en 2018 et 2019, elle est quasi nulle de 2020 "
        "à 2022, lorsque le recours à l'ON RRP approchait "
        f"2{NNBSP}000 milliards de dollars, et réapparaît à mesure que la "
        "facilité se vide. Dans un système saturé de liquidités, les "
        "contraintes de bilan de fin de mois n'avaient aucune prise."
    ),

    "s3": "Le modèle apporte-t-il quelque chose à un modèle autorégressif ?",
    "ax_r2": "R² hors échantillon face à l'AR(1)",
    "tt_r2": "R² face à l'AR(1)",
    "ch_r2": "Valeur positive : le LASSO bat l'AR(1)",
    "cap3": (
        "Zéro est une référence trop faible pour cette variable et n'est pas "
        "retenu. La médiane glissante retire le niveau de l'écart mais pas sa "
        "tendance lente, si bien que la déviation reste autocorrélée à 0,81 et "
        "qu'un modèle autorégressif seul atteint un R² de 0,64 face à zéro. Le "
        "sens des barres porte le signe, la couleur n'est donc pas le seul "
        "repère. Le modèle perd en 2020 et 2021, car la fenêtre glissante doit "
        "traverser la rupture de régime avant de s'adapter."
    ),
    "ch_fan": "Prévision, intervalle à 80 % et réalisation",
    "ax_dev": "déviation, pb",
    "fan_inside": "dans l'intervalle",
    "fan_outside": "hors intervalle",
    "tt_outcome": "réalisation, pb",
    "tt_forecast": "prévision, pb",
    "tt_band": "intervalle à 80 %, pb",
    "cap_fan": (
        "Les 250 derniers jours. La courbe est la prévision en deux étapes de la déviation du "
        "lendemain ; la bande va du 10e au 90e centile des 250 erreurs de "
        "prévision précédentes uniquement, sans jamais voir le jour qu'elle "
        "couvre. Depuis mai 2021, elle contient la réalisation {all} % des "
        "jours, {other} % hors fins de mois, mais seulement {me} % les fins "
        "de mois et {qe} % les fins de trimestre, où les erreurs dépassent ce "
        "que suggère l'ensemble des erreurs."
    ),
    "cap3_dm": (
        "Le gain est-il significatif ? Un test de Diebold-Mariano sur les "
        "erreurs quadratiques, avec variance HAC, le juge significatif depuis "
        "2024 (p = {p24}), mais pas sur l'ensemble de la période depuis mai "
        "2020 (p = {pall}) : les pertes de 2020 compensent les gains ultérieurs."
    ),

    "s3b": "Sur quoi le modèle s'appuie-t-il réellement ?",
    "kind_one": "un par un",
    "kind_group": "permutés en bloc",
    "ax_rmse": "hausse de la RMSE, pb",
    "tt_predictor": "prédicteur",
    "tt_rmse": "hausse de la RMSE, pb",
    "tt_p10": "10e centile, pb",
    "tt_p90": "90e centile, pb",
    "ch_importance": "Dégradation de la prévision après permutation",
    "groups": {
        "calendar, all five": "calendrier, les cinq",
        "month/quarter/year nest": "imbrication mois/trimestre/année",
        "dispersion pair": "paire de dispersion",
        "rate-change pair": "paire de variation de taux",
        "balance sheet, all four": "bilan, les quatre",
    },
    "cap3b": (
        "Permutation par blocs de 21 jours pour préserver la structure "
        "sérielle, répétée 30 fois ; les moustaches couvrent du 10e au 90e "
        "centile de ces répétitions. Les prédicteurs corrélés sont aussi "
        "permutés ensemble, car les indicatrices de calendrier sont "
        "imbriquées : toute fin de trimestre est une fin de mois et toute fin "
        "d'année une fin de trimestre, de sorte qu'un par un ils se "
        "compensent. Les quatre variables de bilan n'apportent rien, même "
        "conjointement, à −0,005 pb. Une valeur négative signifie que la "
        "permutation a amélioré la prévision par hasard, ce qui relève du "
        "bruit."
    ),

    "s4": "Cette capacité prédictive a-t-elle une valeur économique ? Non",
    "s4_intro": (
        "La trésorerie au jour le jour du lendemain est placée soit en repo au "
        "SOFR, soit auprès de la facilité au taux servi, sur décision prise la "
        "veille. Le portage se décompose exactement en "
        "`base + déviation + corridor`, et la base comme le corridor sont "
        "connus un jour à l'avance. La référence pertinente est donc une règle "
        "qui ne dispose que de la partie connue, et non le prêt systématique "
        "passif, qui attribuerait au modèle le mérite du corridor."
    ),
    "col_rule": "règle",
    "col_net": "pb nets",
    "col_vs_always": "écart au prêt systématique",
    "col_vs_known": "écart à la partie connue",
    "rule_always": "prêt systématique",
    "rule_known": "partie connue seule, sans modèle",
    "rule_ar1": "AR(1)",
    "rule_lasso": "LASSO en deux étapes",
    "rule_perfect": "prévision parfaite",
    "conclusion": (
        "L'opportunité totale représente **{prize} pb** de rendement annualisé. "
        "Une règle sans modèle en capte les deux tiers, et l'avantage du LASSO "
        "sur un modèle autorégressif est de {edge} pb, c'est-à-dire rien. Avec "
        "un coût de bascule de 1 pb, le LASSO passe derrière l'AR(1), {lasso1} "
        "contre {ar1_1}. Les deux facilités de la Fed encadrent le corridor si "
        "étroitement que la décision de placement au jour le jour ne recèle "
        "presque aucune valeur. En 2022, le prêt systématique cédait 1,55 pb "
        "par rapport à la facilité et toutes les règles plus élaborées ne font "
        f"que revenir à zéro. C'est pourquoi près de 2{NNBSP}000 milliards de "
        "dollars étaient placés à l'ON RRP : non pour rechercher du portage, "
        "mais simplement parce que la facilité rémunérait davantage."
    ),
    "sources": (
        "Sources : SOFR, EFFR avec la fourchette cible du FOMC, et opérations "
        "ON RRP issues de l'API des marchés de la Federal Reserve Bank of New "
        "York. Treasury General Account (TGA) issu de l'API Fiscal Data du "
        "Trésor américain. Écarts des taux administrés tirés des notes de mise "
        "en œuvre de la Réserve fédérale. Réalisé avec pandas et scikit-learn ; "
        "aucune donnée payante. Les limites et les extensions écartées "
        "figurent dans le README du dépôt."
    ),
}


def _fr_typography(s: str) -> str:
    s = re.sub(r" ([:;?!%])", NNBSP + r"\1", s)
    return re.sub(r"(\d) (pb\b)", r"\1" + NBSP + r"\2", s)


def _apply(v, f):
    if isinstance(v, str):
        return f(v)
    if isinstance(v, dict):
        return {k: _apply(x, f) for k, x in v.items()}
    return v


STRINGS: dict[str, dict[str, object]] = {
    "en": EN,
    "fr": {k: (v if k in ("months", "weekdays") else _apply(v, _fr_typography))
           for k, v in FR.items()},
}
