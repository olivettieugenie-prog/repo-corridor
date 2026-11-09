"""Where repo prints inside the Fed's corridor.

Streamlit app for the Machine Learning with R project (emlyon, 1BLAML).
Reads the precomputed files in data/processed so the page loads without
refitting anything.

Design and language. The colour scheme comes from src/design.py, which holds
four validated options; pick one with the REPO_DESIGN environment variable or
by editing ACTIVE there, then run `python src/design.py <name>` to regenerate
.streamlit/config.toml. Light and dark palettes are both declared, so the
viewer switches theme through Streamlit's own Settings menu and the charts
follow via st.context.theme. There is no custom theme toggle because
Streamlit has no API to set its own theme at runtime, and faking it with
injected CSS leaves the page header in the wrong colours.

Every user-facing string lives in src/i18n.py, keyed by language. The
English/French switch at the top right stores its choice in
st.session_state["lang"]; numbers go through i18n.num so French gets a decimal
comma and narrow no-break spaces. Altair's own axis tick labels (dates and
numbers) stay in Vega's default English format: st.altair_chart exposes no
locale setting, and the only alternative is rewriting every label by
expression. Axis titles, tooltips and in-chart labels are translated.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from design import DESIGNS, active_name  # noqa: E402
from i18n import LANG_NAMES, LANGS, STRINGS, date_short, num  # noqa: E402
import desk  # noqa: E402
import live_feed  # noqa: E402

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"

DESIGN = DESIGNS[active_name()]


def theme_mode() -> str:
    """light or dark, as the viewer's browser and Streamlit settled it."""
    try:
        t = st.context.theme
        return getattr(t, "type", None) or "light"
    except Exception:
        return "light"


MODE = theme_mode()
C = DESIGN[MODE]
BLUE, ORANGE, RED = C["blue"], C["orange"], C["red"]
INK, MUTED, GRID, BAND = C["ink"], C["muted"], C["grid"], C["band"]
FONT = DESIGN["body_font"]
HEAD = DESIGN["head_font"]

if st.session_state.get("lang") not in LANGS:
    st.session_state["lang"] = "en"
LANG = st.session_state["lang"]
T = STRINGS[LANG]
BP = T["bp"]


def n(x: float, dec: int = 0, sign: bool = False) -> str:
    return num(x, dec, LANG, sign)


def fmt_col(s: pd.Series, dec: int, sign: bool = False) -> pd.Series:
    return s.map(lambda x: n(x, dec, sign))


def date_col(s: pd.Series) -> pd.Series:
    return s.map(lambda t: date_short(t, LANG))


st.set_page_config(page_title=T["page_title"], layout="wide")

st.html(
    """<style>
  /* Fonts come from .streamlit/config.toml (theme.font / theme.headingFont),
     which is the supported route. Only numeral alignment is set here. */
  [data-testid="stMetricValue"], [data-testid="stTable"], table {
      font-variant-numeric: tabular-nums;
  }
</style>"""
)


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(PROC / "daily.csv", parse_dates=["date"])
    oos = pd.read_csv(PROC / "oos_predictions.csv", parse_dates=["date"])
    carry = pd.read_csv(PROC / "carry_backtest.csv", parse_dates=["date"])
    return daily, oos, carry


def style(chart: alt.Chart) -> alt.Chart:
    return (chart
            .configure_view(strokeWidth=0)
            .configure(background="transparent")
            .configure_axis(labelFont=FONT, titleFont=FONT, labelColor=MUTED,
                            titleColor=MUTED, gridColor=GRID, domainColor=GRID,
                            tickColor=GRID, labelFontSize=12, titleFontSize=12)
            .configure_legend(labelFont=FONT, titleFont=FONT, labelColor=INK,
                              titleColor=MUTED, labelFontSize=12, titleFontSize=12)
            .configure_title(font=FONT, color=INK, fontSize=14, anchor="start"))


def r2(y, p, base) -> float:
    y, p, base = np.asarray(y), np.asarray(p), np.asarray(base)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - base) ** 2))


daily, oos, carry = load()

head, switch = st.columns([5, 1], vertical_alignment="center")
with switch, st.container(horizontal=True, horizontal_alignment="right"):
    st.segmented_control(T["lang_label"], LANGS, format_func=LANG_NAMES.get,
                         key="lang", required=True, label_visibility="collapsed")
head.title(T["title"])
st.markdown(T["intro"])


# ---------------------------------------------------------------- desk view
@st.cache_data(ttl=3600, show_spinner=False)
def load_live(url: str, local: str) -> tuple[pd.DataFrame | None, str]:
    return live_feed.load(url, Path(local))


def desk_view() -> None:
    import os
    live, source = load_live(os.environ.get("REPO_LIVE_URL", live_feed.RAW_URL),
                             os.environ.get("REPO_LIVE_LOCAL", str(live_feed.LOCAL)))
    with st.container(border=True):
        st.subheader(T["desk_title"])
        if live is None:
            st.info(T["desk_none"])
            return
        if source == "local":
            st.caption(T["desk_src_local"])
        live = live_feed.latest_per_target(live)
        row = live.iloc[-1]
        target = desk.long_date(row["target_date"], LANG)
        if desk.is_stale(row["target_date"], os.environ.get("REPO_TODAY")):
            st.warning(T["desk_stale"].format(target=target))
        if int(row.get("range_unconfirmed", 0) or 0):
            st.warning(T["desk_unconfirmed"].format(target=target))

        st.markdown(T["desk_target"].format(target=target))
        k2, k3, k4, k5 = st.columns(4)
        k2.metric(T["desk_forecast"], f"{n(row['forecast_bp'], 1, sign=True)}\u00a0{BP}")
        k2.caption(T["desk_range"].format(lo=n(row["lower_bp"], 1, sign=True),
                                          hi=n(row["upper_bp"], 1, sign=True)))
        if pd.notna(row.get("implied_sofr_pct")):
            k3.metric(T["desk_sofr"], f"{n(row['implied_sofr_pct'], 2)}\u00a0%")
            k3.caption(T["desk_sofr_range"].format(lo=n(row["implied_sofr_lo_pct"], 2),
                                                   hi=n(row["implied_sofr_hi_pct"], 2)))
        for col, name, lvl, chg, dt, dec in (
                (k4, "ON RRP", row["rrp_bn"], row["d_rrp_bn"], row["rrp_date"], 1),
                (k5, "TGA", row["tga_bn"], row["d_tga_bn"], row["tga_date"], 0)):
            col.metric(name, T["desk_level"].format(v=n(lvl, dec)),
                       delta=n(chg, dec, sign=True), delta_color="off",
                       help=T["desk_level_help"].format(date=desk.long_date(dt, LANG)))

        flags = desk.flags_in_play(row, LANG)
        st.markdown(T["desk_flags"].format(flags=", ".join(flags) or T["desk_no_flags"]))
        line1, line2 = desk.morning_note(row, LANG)
        st.markdown(f"> {line1}  \n> {line2}")
        st.caption(T["desk_asof"].format(run=desk.run_time_et(row["run_time_utc"], LANG),
                                         asof=desk.long_date(row["as_of_date"], LANG)))
        tr = desk.track_record(live)
        start = desk.long_date(tr["start"], LANG)
        if "hit" in tr:
            st.caption(T["desk_track"].format(n=tr["n"], start=start, hit=n(100 * tr["hit"], 0),
                                              rmse=n(tr["rmse"], 2), rmse_ar1=n(tr["rmse_ar1"], 2)))
        else:
            st.caption(T["desk_track_pending"].format(start=start, n=tr["n"]))


desk_view()

# ---------------------------------------------------------------- stat tiles
usable = carry
always = usable["carry_bp"].mean()
perfect = np.maximum(usable["carry_bp"], 0).mean()
rec = oos[oos["date"].dt.year >= 2024]
peak = daily["carry_bp"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric(T["tile_days"], n(len(daily)), help=T["tile_days_help"])
# Significance of the headline, from src/dm_test.py.
dmt = pd.read_csv(PROC / "dm_test.csv").set_index("period")
DM_ARGS = {"p24": n(dmt.loc["2024+", "p_two_sided"], 4),
           "pall": n(dmt.loc["all", "p_two_sided"], 2),
           "r2all": n(dmt.loc["all", "r2_vs_ar1"], 3, sign=True)}
c2.metric(T["tile_skill"], n(r2(rec["y"], rec["two_stage"], rec["ar1"]), 3, sign=True),
          help=T["tile_skill_help"].format(**DM_ARGS))
c3.metric(T["tile_carry"], f"{n(always, 2)}\u00a0{BP}", help=T["tile_carry_help"])
c4.metric(T["tile_peak"], f"{n(peak)}\u00a0{BP}", help=T["tile_peak_help"])

st.divider()

# ------------------------------------------------------------- the corridor
st.subheader(T["s1"])
lo, hi = daily["date"].min().to_pydatetime(), daily["date"].max().to_pydatetime()
# A fixed key keeps the chosen range when the language, and so the label,
# changes.
start, end = st.slider(T["date_range"], min_value=lo, max_value=hi, value=(lo, hi),
                       format=T["slider_format"], key="range")
d = daily[(daily["date"] >= start) & (daily["date"] <= end)]

# The two corridor panels are drawn as separate charts, because a vconcat
# ignores width="stretch" and overflowed its column. To keep their dates
# aligned they share one explicit x domain, the same y-axis gutter, and the
# legend sits above the top panel rather than taking width at its side.
# Drawn alone, each height includes its title, legend and axes, hence taller
# values than the old 260 and 150 plot heights.
X_DATE = alt.X("date:T", title=None,
               scale=alt.Scale(domain=[pd.Timestamp(start).isoformat(),
                                       pd.Timestamp(end).isoformat()]))
Y_GUTTER = 56

band = (alt.Chart(d).mark_area(opacity=0.55, color=BAND)
        .encode(x=X_DATE,
                y=alt.Y("target_lo:Q", title=T["ax_percent"],
                        scale=alt.Scale(zero=False),
                        axis=alt.Axis(minExtent=Y_GUTTER, maxExtent=Y_GUTTER)),
                y2="target_hi:Q"))
# Standing Repo Facility rate from data/srf_rates.csv, one row per FOMC
# implementation note since the facility opened on 29 July 2021.
srf_rates = pd.read_csv(ROOT / "data" / "srf_rates.csv", parse_dates=["effective_date"])
d = pd.merge_asof(d.sort_values("date"),
                  srf_rates[["effective_date", "srf_rate_pct"]].rename(
                      columns={"effective_date": "date"}).sort_values("date"),
                  on="date", direction="backward")
rates = d.melt(id_vars="date", value_vars=["sofr", "award_pct", "srf_rate_pct"],
               var_name="series", value_name="pct").dropna()
SERIES = [T["series_sofr"], T["series_award"], T["series_srf"]]
rates["series"] = rates["series"].map(dict(zip(["sofr", "award_pct", "srf_rate_pct"], SERIES)))
# Tooltip values are formatted here rather than by Vega, so they follow the
# language.
rates["date_s"] = date_col(rates["date"])
rates["pct_s"] = fmt_col(rates["pct"], 2)
lines = (alt.Chart(rates).mark_line(strokeWidth=2)
         .encode(x=X_DATE,
                 y=alt.Y("pct:Q", title=T["ax_percent"], scale=alt.Scale(zero=False)),
                 color=alt.Color("series:N", title=None,
                                 scale=alt.Scale(domain=SERIES, range=[BLUE, ORANGE, MUTED]),
                                 legend=alt.Legend(orient="top")),
                 strokeDash=alt.StrokeDash("series:N", legend=None,
                                           scale=alt.Scale(domain=SERIES,
                                                           range=[[1, 0], [1, 0], [5, 3]])),
                 tooltip=[alt.Tooltip("date_s:N", title=T["tt_date"]),
                          alt.Tooltip("series:N", title=T["tt_series"]),
                          alt.Tooltip("pct_s:N", title=T["tt_percent"])]))
# The corridor is 25bp wide on an axis spanning five percentage points, so the
# band is a sliver and the position inside it cannot be read off the rates. The
# second panel plots that position directly, in basis points, on the same
# dates. Two panels rather than a second y-scale on one.
# 17 September 2019 reaches 325 bp against a normal range of about -10 to +30,
# so the axis is clipped to keep the rest readable and the spike is labelled
# rather than dropped.
inside = d.dropna(subset=["carry_bp"]).copy()
inside["date_s"] = date_col(inside["date"])
inside["carry_s"] = fmt_col(inside["carry_bp"], 1)
CLIP_HI = 70
pos = (alt.Chart(inside).mark_line(strokeWidth=2, color=BLUE, clip=True)
       .encode(x=X_DATE,
               y=alt.Y("carry_bp:Q",
                       title=T["ax_position"].format(clip=CLIP_HI),
                       scale=alt.Scale(domain=[-15, CLIP_HI]),
                       axis=alt.Axis(minExtent=Y_GUTTER, maxExtent=Y_GUTTER)),
               tooltip=[alt.Tooltip("date_s:N", title=T["tt_date"]),
                        alt.Tooltip("carry_s:N", title=T["tt_carry"])]))
# Only the single largest day is labelled. Annotating every clipped spike put
# three labels on top of each other around 2019.
top = inside.loc[[inside["carry_bp"].idxmax()]].copy()
top["lab"] = fmt_col(top["carry_bp"], 0) + "\u00a0" + BP
top["y"] = CLIP_HI - 7
pos = pos + (alt.Chart(top).mark_text(align="left", dx=6, font=FONT,
                                      fontSize=11, color=RED)
             .encode(x="date:T", y=alt.Y("y:Q"), text="lab:N"))
floor = (alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeWidth=1)
         .encode(y="y:Q"))
st.altair_chart(style((band + lines).properties(height=340, title=T["ch_band"])),
                width="stretch")
st.altair_chart(style((pos + floor).properties(height=320, title=T["ch_position"])),
                width="stretch")
# Daily Standing Repo Facility take-up, from src/fetch_srf.py.
srf = pd.read_csv(PROC / "srf_takeup.csv", parse_dates=["date"])
srf = srf[(srf["date"] >= start) & (srf["date"] <= end)].copy()
srf["date_s"] = date_col(srf["date"])
srf["srf_s"] = fmt_col(srf["srf_bn"], 1)
take = (alt.Chart(srf).mark_bar(color=RED, width=2)
        .encode(x=X_DATE,
                y=alt.Y("srf_bn:Q", title=T["ax_srf"],
                        axis=alt.Axis(minExtent=Y_GUTTER, maxExtent=Y_GUTTER)),
                tooltip=[alt.Tooltip("date_s:N", title=T["tt_date"]),
                         alt.Tooltip("srf_s:N", title=T["tt_srf"])]))
st.altair_chart(style(take.properties(height=200, title=T["ch_srf"])), width="stretch")
st.caption(T["cap1"])
st.caption(T["cap_srf"])

st.divider()

# ------------------------------------------- the regime-dependent month-end
st.subheader(T["s2"])
dd = daily.dropna(subset=["dev_bp"]).copy()
dd["dev_c"] = dd["dev_bp"].clip(-25, 25)
prem = (dd.groupby("year")
        .apply(lambda g: g.loc[g.month_end == 1, "dev_c"].mean()
               - g.loc[g.month_end == 0, "dev_c"].mean(), include_groups=False)
        .rename("premium_bp").reset_index())
rrp = dd.groupby("year")["rrp_bn_t"].mean().rename("rrp_bn").reset_index()
prem["premium_s"] = fmt_col(prem["premium_bp"], 2)
prem["label_s"] = fmt_col(prem["premium_bp"], 1)
rrp["rrp_s"] = fmt_col(rrp["rrp_bn"], 0)

left, right = st.columns(2)
bars = (alt.Chart(prem).mark_bar(color=BLUE, cornerRadiusTopLeft=4,
                                 cornerRadiusTopRight=4, size=26)
        .encode(x=alt.X("year:O", title=None),
                y=alt.Y("premium_bp:Q", title=T["ax_bp"]),
                tooltip=[alt.Tooltip("year:O", title=T["tt_year"]),
                         alt.Tooltip("premium_s:N", title=T["tt_premium"])]))
labels = (alt.Chart(prem).mark_text(dy=-8, font=FONT, fontSize=11, color=INK)
          .encode(x="year:O", y="premium_bp:Q",
                  text="label_s:N"))
left.altair_chart(style((bars + labels).properties(
    height=300, title=T["ch_premium"])), width="stretch")

ubars = (alt.Chart(rrp).mark_bar(color=ORANGE, cornerRadiusTopLeft=4,
                                 cornerRadiusTopRight=4, size=26)
         .encode(x=alt.X("year:O", title=None),
                 y=alt.Y("rrp_bn:Q", title=T["ax_rrp"]),
                 tooltip=[alt.Tooltip("year:O", title=T["tt_year"]),
                          alt.Tooltip("rrp_s:N", title=T["tt_rrp"])]))
right.altair_chart(style(ubars.properties(
    height=300, title=T["ch_rrp"])), width="stretch")
st.caption(T["cap2"])

st.divider()

# -------------------------------------------------------- does the model add
st.subheader(T["s3"])
o = oos.copy()
o["year"] = o["date"].dt.year
sk = (o.groupby("year")
      .apply(lambda g: r2(g["y"], g["two_stage"], g["ar1"]), include_groups=False)
      .rename("r2").reset_index())
sk["sign"] = np.where(sk["r2"] >= 0, "adds", "hurts")   # internal, not shown
sk["r2_s"] = fmt_col(sk["r2"], 3, sign=True)
sk["label_s"] = fmt_col(sk["r2"], 2, sign=True)

dv = (alt.Chart(sk).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                             cornerRadiusBottomLeft=4, cornerRadiusBottomRight=4,
                             size=26)
      .encode(x=alt.X("year:O", title=None),
              y=alt.Y("r2:Q", title=T["ax_r2"]),
              color=alt.Color("sign:N", title=None,
                              scale=alt.Scale(domain=["adds", "hurts"],
                                              range=[BLUE, RED]),
                              legend=None),
              tooltip=[alt.Tooltip("year:O", title=T["tt_year"]),
                       alt.Tooltip("r2_s:N", title=T["tt_r2"])]))
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeWidth=1).encode(y="y:Q")
# dy is a mark property, not an encoding channel, so the label sits above a
# positive bar and below a negative one via two layers rather than a condition.
def dlabel(frame: pd.DataFrame, dy: int) -> alt.Chart:
    return (alt.Chart(frame).mark_text(dy=dy, font=FONT, fontSize=11, color=INK)
            .encode(x="year:O", y="r2:Q", text="label_s:N"))


dlab = dlabel(sk[sk["r2"] >= 0], -8) + dlabel(sk[sk["r2"] < 0], 14)
st.altair_chart(style((dv + zero + dlab).properties(
    height=320, title=T["ch_r2"])),
    width="stretch")
st.caption(T["cap3"])
st.caption(T["cap3_dm"].format(**DM_ARGS))

# Forecast against outcome with the 80% interval from src/intervals.py, which
# uses only errors of forecasts whose target date came before.
fan = o.dropna(subset=["lo80"]).tail(250).copy()
fan["inside"] = np.where((fan["y"] >= fan["lo80"]) & (fan["y"] <= fan["hi80"]),
                         T["fan_inside"], T["fan_outside"])
fan["date_s"] = date_col(fan["date"])
fan["y_s"] = fmt_col(fan["y"], 1)
fan["f_s"] = fmt_col(fan["two_stage"], 1)
fan["band_s"] = fmt_col(fan["lo80"], 1) + " ; " + fmt_col(fan["hi80"], 1)
fan_tt = [alt.Tooltip("date_s:N", title=T["tt_date"]),
          alt.Tooltip("y_s:N", title=T["tt_outcome"]),
          alt.Tooltip("f_s:N", title=T["tt_forecast"]),
          alt.Tooltip("band_s:N", title=T["tt_band"])]
fx = alt.X("date:T", title=None)
fband = (alt.Chart(fan).mark_area(opacity=0.7, color=BAND)
         .encode(x=fx, y=alt.Y("lo80:Q", title=T["ax_dev"]), y2="hi80:Q"))
fline = (alt.Chart(fan).mark_line(strokeWidth=1.5, color=BLUE)
         .encode(x=fx, y="two_stage:Q", tooltip=fan_tt))
fdots = (alt.Chart(fan).mark_point(filled=True, size=18)
         .encode(x=fx, y="y:Q",
                 color=alt.Color("inside:N", title=None,
                                 scale=alt.Scale(domain=[T["fan_inside"], T["fan_outside"]],
                                                 range=[MUTED, RED]),
                                 legend=alt.Legend(orient="top")),
                 tooltip=fan_tt))
st.altair_chart(style((fband + fline + fdots).properties(height=300, title=T["ch_fan"])),
                width="stretch")
cov = pd.read_csv(PROC / "interval_coverage.csv").set_index("group")["coverage"]
st.caption(T["cap_fan"].format(all=n(100 * cov["all"], 1),
                               me=n(100 * cov["month-end"], 1),
                               qe=n(100 * cov["quarter-end"], 1),
                               other=n(100 * cov["other days"], 1)))

st.subheader(T["s3b"])
imp = pd.read_csv(PROC / "permutation_importance.csv")
KINDS = [T["kind_one"], T["kind_group"]]
imp["kind"] = np.where(imp["n_cols"] > 1, KINDS[1], KINDS[0])
# Group rows get a translated name; single predictors keep their variable name.
imp["what"] = imp["what"].map(lambda w: T["groups"].get(w, w))
for col, dec in (("mean_bp", 4), ("p10", 4), ("p90", 4)):
    imp[col + "_s"] = fmt_col(imp[col], dec, sign=True)
order = imp.sort_values("mean_bp", ascending=False)["what"].tolist()
# 19 categories: Altair drops labels to avoid overlap unless told not to, and
# the chart needs the height to show them all or the dots cannot be read.
enc_y = alt.Y("what:N", sort=order, title=None,
              axis=alt.Axis(labelOverlap=False, labelLimit=220))
dots = (alt.Chart(imp).mark_point(filled=True, size=90)
        .encode(x=alt.X("mean_bp:Q", title=T["ax_rmse"]),
                y=enc_y,
                color=alt.Color("kind:N", title=None,
                                scale=alt.Scale(domain=KINDS, range=[BLUE, ORANGE])),
                tooltip=[alt.Tooltip("what:N", title=T["tt_predictor"]),
                         alt.Tooltip("mean_bp_s:N", title=T["tt_rmse"]),
                         alt.Tooltip("p10_s:N", title=T["tt_p10"]),
                         alt.Tooltip("p90_s:N", title=T["tt_p90"])]))
bars = (alt.Chart(imp).mark_rule(strokeWidth=2, opacity=0.65)
        .encode(x="p10:Q", x2="p90:Q", y=enc_y,
                color=alt.Color("kind:N", legend=None,
                                scale=alt.Scale(domain=KINDS, range=[BLUE, ORANGE]))))
zline = (alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED, strokeWidth=1)
         .encode(x="x:Q"))
st.altair_chart(style((bars + dots + zline).properties(
    height=28 * len(imp) + 40,
    title=T["ch_importance"])),
    width="stretch")
st.caption(T["cap3b"])

st.divider()

# ------------------------------------------------------------ is it worth it
st.subheader(T["s4"])
st.markdown(T["s4_intro"])
# None marks the rule's own column, shown as a dash.
NET = [4.829, 5.030, 5.114, 5.117, 5.132]
VS_ALWAYS = [None, 0.201, 0.284, 0.287, 0.303]
VS_KNOWN = [-0.201, None, 0.084, 0.087, 0.103]
dash = lambda v: "—" if v is None else n(v, 3, sign=True)  # noqa: E731
rules = pd.DataFrame({
    T["col_rule"]: [T[k] for k in ("rule_always", "rule_known", "rule_ar1",
                                   "rule_lasso", "rule_perfect")],
    T["col_net"]: [n(v, 3) for v in NET],
    T["col_vs_always"]: [dash(v) for v in VS_ALWAYS],
    T["col_vs_known"]: [dash(v) for v in VS_KNOWN],
})
st.dataframe(rules, hide_index=True, width="stretch")
st.markdown(T["conclusion"])

st.divider()
st.caption(T["sources"])
