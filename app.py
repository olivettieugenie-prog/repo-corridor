"""Where repo prints inside the Fed's corridor.

Streamlit app for the Machine Learning with R project (emlyon, 1BLAML).
Reads the precomputed files in data/processed so the page loads without
refitting anything:
    daily.csv           one row per SOFR date
    oos_predictions.csv walk-forward forecasts and benchmarks
    carry_backtest.csv  the cash placement rule

Run locally:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"

# Validated against the light chart surface #fcfcfb with the dataviz palette
# validator: all six checks pass in light and dark. The app pins the light
# theme in .streamlit/config.toml so these are the colours that render.
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BAND = "#e1e0d9"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

st.set_page_config(page_title="Repo in the Fed's corridor", layout="wide")


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(PROC / "daily.csv", parse_dates=["date"])
    oos = pd.read_csv(PROC / "oos_predictions.csv", parse_dates=["date"])
    carry = pd.read_csv(PROC / "carry_backtest.csv", parse_dates=["date"])
    return daily, oos, carry


def style(chart: alt.Chart) -> alt.Chart:
    return (chart
            .configure_view(strokeWidth=0)
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

st.title("Where repo prints inside the Fed's corridor")
st.markdown(
    "Overnight repo trades between two administered rates: the ON RRP facility "
    "below and interest on reserves above. This page asks what moves SOFR "
    "inside that corridor, whether a penalised regression predicts it better "
    "than an autoregression, and whether that prediction is worth any money. "
    "Public data only, from the New York Fed and the US Treasury."
)

# ---------------------------------------------------------------- stat tiles
usable = carry
always = usable["carry_bp"].mean()
perfect = np.maximum(usable["carry_bp"], 0).mean()
rec = oos[oos["date"].dt.year >= 2024]
peak = daily["carry_bp"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("SOFR days covered", f"{len(daily):,}",
          help="2 April 2018 to 26 October 2026, from the NY Fed API")
c2.metric("Model skill vs AR(1), 2024+",
          f"{r2(rec['y'], rec['two_stage'], rec['ar1']):+.3f}",
          help="Out-of-sample R squared of the two-stage LASSO against an "
               "autoregression, rolling 500-day training window")
c3.metric("Carry from always lending", f"{always:.2f} bp",
          help="Mean annualised pickup of overnight repo over the ON RRP "
               "facility, on the 1,336 days where the award rate is known")
c4.metric("Carry on 17 Sep 2019", f"{peak:,.0f} bp",
          help="The day the plumbing broke: SOFR printed 5.25% while the Fed's "
               "facility paid 2.00%. Normal range is about -10 to +30 bp.")

st.divider()

# ------------------------------------------------------------- the corridor
st.subheader("1. The corridor")
lo, hi = daily["date"].min().to_pydatetime(), daily["date"].max().to_pydatetime()
start, end = st.slider("Date range", min_value=lo, max_value=hi, value=(lo, hi),
                       format="MMM YYYY")
d = daily[(daily["date"] >= start) & (daily["date"] <= end)]

band = (alt.Chart(d).mark_area(opacity=0.55, color=BAND)
        .encode(x=alt.X("date:T", title=None),
                y=alt.Y("target_lo:Q", title="percent",
                        scale=alt.Scale(zero=False)),
                y2="target_hi:Q"))
rates = d.melt(id_vars="date", value_vars=["sofr", "award_pct"],
               var_name="series", value_name="pct").dropna()
rates["series"] = rates["series"].map({"sofr": "SOFR",
                                       "award_pct": "ON RRP award rate"})
lines = (alt.Chart(rates).mark_line(strokeWidth=2)
         .encode(x=alt.X("date:T", title=None),
                 y=alt.Y("pct:Q", title="percent", scale=alt.Scale(zero=False)),
                 color=alt.Color("series:N", title=None,
                                 scale=alt.Scale(domain=["SOFR", "ON RRP award rate"],
                                                 range=[BLUE, ORANGE])),
                 tooltip=[alt.Tooltip("date:T", title="date"),
                          alt.Tooltip("series:N", title=None),
                          alt.Tooltip("pct:Q", title="percent", format=".2f")]))
# The corridor is 25bp wide on an axis spanning five percentage points, so the
# band is a sliver and the position inside it cannot be read off the rates. The
# second panel plots that position directly, in basis points, sharing the
# x-axis. Two panels rather than a second y-scale on one.
# 17 September 2019 reaches 325 bp against a normal range of about -10 to +30,
# so the axis is clipped to keep the rest readable and the spike is labelled
# rather than dropped.
inside = d.dropna(subset=["carry_bp"])
CLIP_HI = 70
pos = (alt.Chart(inside).mark_line(strokeWidth=2, color=BLUE, clip=True)
       .encode(x=alt.X("date:T", title=None),
               y=alt.Y("carry_bp:Q",
                       title=f"SOFR over award rate, bp (clipped at {CLIP_HI})",
                       scale=alt.Scale(domain=[-15, CLIP_HI])),
               tooltip=[alt.Tooltip("date:T", title="date"),
                        alt.Tooltip("carry_bp:Q", title="carry, bp", format=".1f")]))
# Only the single largest day is labelled. Annotating every clipped spike put
# three labels on top of each other around 2019.
top = inside.loc[[inside["carry_bp"].idxmax()]].copy()
top["lab"] = top["carry_bp"].round().astype(int).astype(str) + " bp"
top["y"] = CLIP_HI - 7
pos = pos + (alt.Chart(top).mark_text(align="left", dx=6, font=FONT,
                                      fontSize=11, color=RED)
             .encode(x="date:T", y=alt.Y("y:Q"), text="lab:N"))
floor = (alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeWidth=1)
         .encode(y="y:Q"))
st.altair_chart(style(alt.vconcat(
    (band + lines).properties(height=260,
                              title="Shaded band is the FOMC target range"),
    (pos + floor).properties(height=150,
                             title="Where SOFR sits above the facility floor"),
).resolve_scale(x="shared")), width="stretch")
st.caption(
    "The administered rates come from a table of 15 Federal Reserve "
    "implementation notes, each cited in the repository, because the offsets "
    "are not stable. ON RRP sat at the bottom of the range until September "
    "2019, 5 bp below it through to March 2020, 5 bp above it from June 2021, "
    "and back at the bottom from 19 December 2024. The table reproduces all "
    "491 directly observed award rates."
)

st.divider()

# ------------------------------------------- the regime-dependent month-end
st.subheader("2. Calendar pressure only bites when reserves are scarce")
dd = daily.dropna(subset=["dev_bp"]).copy()
dd["dev_c"] = dd["dev_bp"].clip(-25, 25)
prem = (dd.groupby("year")
        .apply(lambda g: g.loc[g.month_end == 1, "dev_c"].mean()
               - g.loc[g.month_end == 0, "dev_c"].mean(), include_groups=False)
        .rename("premium_bp").reset_index())
rrp = dd.groupby("year")["rrp_bn_t"].mean().rename("rrp_bn").reset_index()

left, right = st.columns(2)
bars = (alt.Chart(prem).mark_bar(color=BLUE, cornerRadiusTopLeft=4,
                                 cornerRadiusTopRight=4, size=26)
        .encode(x=alt.X("year:O", title=None),
                y=alt.Y("premium_bp:Q", title="basis points"),
                tooltip=[alt.Tooltip("year:O", title="year"),
                         alt.Tooltip("premium_bp:Q", title="premium, bp", format=".2f")]))
labels = (alt.Chart(prem).mark_text(dy=-8, font=FONT, fontSize=11, color=INK)
          .encode(x="year:O", y="premium_bp:Q",
                  text=alt.Text("premium_bp:Q", format=".1f")))
left.altair_chart(style((bars + labels).properties(
    height=300, title="Month-end premium on the spread")), width="stretch")

ubars = (alt.Chart(rrp).mark_bar(color=ORANGE, cornerRadiusTopLeft=4,
                                 cornerRadiusTopRight=4, size=26)
         .encode(x=alt.X("year:O", title=None),
                 y=alt.Y("rrp_bn:Q", title="USD bn, daily average"),
                 tooltip=[alt.Tooltip("year:O", title="year"),
                          alt.Tooltip("rrp_bn:Q", title="ON RRP, $bn", format=",.0f")]))
right.altair_chart(style(ubars.properties(
    height=300, title="ON RRP usage")), width="stretch")
st.caption(
    "Two charts rather than two axes on one. The month-end premium is about "
    "9 to 13 bp in 2018 and 2019, essentially zero from 2020 to 2022 when ON "
    "RRP usage ran near $2.0tn, and returns as the facility drains. With the "
    "system awash with cash, month-end balance-sheet constraints had nothing "
    "to push against."
)

st.divider()

# -------------------------------------------------------- does the model add
st.subheader("3. Does the model add anything to an autoregression?")
o = oos.copy()
o["year"] = o["date"].dt.year
sk = (o.groupby("year")
      .apply(lambda g: r2(g["y"], g["two_stage"], g["ar1"]), include_groups=False)
      .rename("r2").reset_index())
sk["sign"] = np.where(sk["r2"] >= 0, "adds", "hurts")

dv = (alt.Chart(sk).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                             cornerRadiusBottomLeft=4, cornerRadiusBottomRight=4,
                             size=26)
      .encode(x=alt.X("year:O", title=None),
              y=alt.Y("r2:Q", title="out-of-sample R² vs AR(1)"),
              color=alt.Color("sign:N", title=None,
                              scale=alt.Scale(domain=["adds", "hurts"],
                                              range=[BLUE, RED]),
                              legend=None),
              tooltip=[alt.Tooltip("year:O", title="year"),
                       alt.Tooltip("r2:Q", title="R² vs AR(1)", format="+.3f")]))
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeWidth=1).encode(y="y:Q")
# dy is a mark property, not an encoding channel, so the label sits above a
# positive bar and below a negative one via two layers rather than a condition.
def dlabel(frame: pd.DataFrame, dy: int) -> alt.Chart:
    return (alt.Chart(frame).mark_text(dy=dy, font=FONT, fontSize=11, color=INK)
            .encode(x="year:O", y="r2:Q", text=alt.Text("r2:Q", format="+.2f")))


dlab = dlabel(sk[sk["r2"] >= 0], -8) + dlabel(sk[sk["r2"] < 0], 14)
st.altair_chart(style((dv + zero + dlab).properties(
    height=320, title="Positive means the LASSO beat the autoregression")),
    width="stretch")
st.caption(
    "Zero is a weak benchmark for this target and is not used. The trailing "
    "median removes the level of the spread but not a slow trend in it, so the "
    "deviation stays autocorrelated at 0.81 and an autoregression alone reaches "
    "R² 0.64 against zero. The bar direction carries the sign, so colour is not "
    "the only cue. The model loses in 2020 and 2021 because the rolling window "
    "has to pass through the regime break before it adapts."
)

st.divider()

# ------------------------------------------------------------ is it worth it
st.subheader("4. Is the skill worth any money? No")
st.markdown(
    "Place tomorrow's overnight cash in repo at SOFR or at the facility at the "
    "award rate, decided the day before. The carry splits exactly into "
    "`base + deviation + corridor`, and base and corridor are both known a day "
    "ahead. So the benchmark that matters is a rule given only the known part, "
    "not passive always-lend, which would credit the model for the corridor."
)
rules = pd.DataFrame({
    "rule": ["always lend", "known only, no model", "AR(1)", "two-stage LASSO",
             "perfect foresight"],
    "net bp": [4.829, 5.030, 5.114, 5.117, 5.132],
    "vs always-lend": ["—", "+0.201", "+0.284", "+0.287", "+0.303"],
    "vs known-only": ["-0.201", "—", "+0.084", "+0.087", "+0.103"],
})
st.dataframe(rules, hide_index=True, width="stretch")
st.markdown(
    "The entire opportunity is **0.303 bp** of annualised yield. A rule with no "
    "model captures two thirds of it, and the LASSO's margin over an "
    "autoregression is 0.003 bp, which is nothing. At a 1 bp switching cost the "
    "LASSO falls behind AR(1), +0.155 against +0.186. The Fed's two facilities "
    "bound the corridor so tightly that the overnight placement decision has "
    "almost no money in it. In 2022 always-lend gave up 1.55 bp against the "
    "facility and every smarter rule merely returns to zero, which is why about "
    "$2.0tn sat at ON RRP: not chasing carry, but because the facility simply "
    "paid more."
)

st.divider()
st.caption(
    "Sources: SOFR, EFFR with the FOMC target range, and ON RRP operations from "
    "the Federal Reserve Bank of New York markets API. Treasury General Account "
    "from the US Treasury Fiscal Data API. Administered-rate offsets read from "
    "Federal Reserve implementation notes. Built with pandas and scikit-learn; "
    "no paid data. Limitations and the extensions left out are in the repository "
    "README."
)
