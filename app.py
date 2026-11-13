"""Repo turn monitor: what the next repo turn will cost.

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

import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from design import DESIGNS, active_name  # noqa: E402
from i18n import LANG_NAMES, LANGS, STRINGS, date_short, num  # noqa: E402
import analogs  # noqa: E402
import dashboard  # noqa: E402
import desk  # noqa: E402
import market_calendar  # noqa: E402
import roll_cost  # noqa: E402
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

# One scoped stylesheet (src/style.css) for density, the sticky section links
# and the market strip; its colours are CSS variables from the active palette.
st.html(dashboard.stylesheet(C))


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
                            tickColor=GRID, labelFontSize=11, titleFontSize=11)
            .configure_legend(labelFont=FONT, titleFont=FONT, labelColor=INK,
                              titleColor=MUTED, labelFontSize=11, titleFontSize=11)
            .configure_title(font=FONT, color=INK, fontSize=12, anchor="start"))


def r2(y, p, base) -> float:
    y, p, base = np.asarray(y), np.asarray(p), np.asarray(base)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - base) ** 2))


# daily_bt is main's frozen backtest snapshot (data/processed), kept for the
# backtest panels. daily is the table the job rebuilt on its last run, for
# everything shown as current: header, strip, charts, analogs, turns.
daily_bt, oos, carry = load()


@st.cache_data(ttl=3600, show_spinner=False)
def load_daily(url: str, local: str) -> tuple[pd.DataFrame, str]:
    return live_feed.load_daily(url, Path(local))


daily, DAILY_SOURCE = load_daily(os.environ.get("REPO_DAILY_URL", live_feed.DAILY_URL),
                                 os.environ.get("REPO_DAILY_LOCAL", str(live_feed.DAILY_LOCAL)))


@st.cache_data(ttl=3600, show_spinner=False)
def load_live(url: str, local: str) -> tuple[pd.DataFrame | None, str]:
    return live_feed.load(url, Path(local))


def today() -> pd.Timestamp:
    return pd.Timestamp(os.environ.get("REPO_TODAY") or desk.today_et())


LIVE, LIVE_SOURCE = load_live(os.environ.get("REPO_LIVE_URL", live_feed.RAW_URL),
                              os.environ.get("REPO_LIVE_LOCAL", str(live_feed.LOCAL)))
LIVE = live_feed.latest_per_target(LIVE) if LIVE is not None else None
LIVE_ROW = LIVE.iloc[-1] if LIVE is not None else None


@st.cache_data(ttl=3600, show_spinner=False)
def load_forward(url: str, local: str) -> tuple[pd.DataFrame | None, str]:
    return live_feed.load_forward(url, Path(local))


FWD, FWD_SOURCE = load_forward(os.environ.get("REPO_FORWARD_URL", live_feed.FORWARD_URL),
                               os.environ.get("REPO_FORWARD_LOCAL", str(live_feed.FORWARD_LOCAL)))
SRF_RATES = pd.read_csv(ROOT / "data" / "srf_rates.csv", parse_dates=["effective_date"])
SRF_TAKE = pd.read_csv(PROC / "srf_takeup.csv", parse_dates=["date"])

# ------------------------------------------------------------------ header
TILES = dashboard.strip_tiles(daily, LIVE_ROW, SRF_RATES, SRF_TAKE)
bal_date = max(t["date"] for t in TILES if t["key"] in ("rrp", "tga"))
head, switch = st.columns([5, 1], vertical_alignment="center")
with switch, st.container(horizontal=True, horizontal_alignment="right"):
    st.segmented_control(T["lang_label"], LANGS, format_func=LANG_NAMES.get,
                         key="lang", required=True, label_visibility="collapsed")
with head:
    st.title(T["title"])
    asof = dict(rates=desk.long_date(daily["date"].max(), LANG),
                bal=desk.long_date(bal_date, LANG))
    st.caption(T["hdr_asof"].format(run=desk.run_time_et(LIVE_ROW["run_time_utc"], LANG), **asof)
               if LIVE_ROW is not None else T["hdr_asof_nolive"].format(**asof))
    st.caption(T["hdr_delay"])
st.html(dashboard.navbar(LANG))
st.html(dashboard.strip_html(TILES, LANG))

# ---------------------------------------------------------------- desk row
STATE = roll_cost.build_state(daily, LIVE_ROW, oos)
# Calibrated turn quantiles, one fit per forecast horizon, and their
# walk-forward record at each horizon (src/turn_quantiles.py).
COEFS = pd.read_csv(PROC / "turn_quantile_coefs_horizon.csv")
TSCORE = (pd.read_csv(PROC / "turn_quantile_horizons.csv")
          .query("version == 'calibrated'").set_index(["type", "horizon"]))
# Where the conformal shift does not bring the median closer to 50% out of
# sample, the raw median is kept and its bias is stated next to the table.
POLICY = roll_cost.median_policy()


def signed_style(v: str) -> str:
    """Colour for a signed, formatted number; the sign stays in the text."""
    v = str(v).strip()
    if v.startswith("+"):
        return f"color: {C['blue']}"
    if v.startswith("-") or v.startswith("\u2212"):
        return f"color: {C['red']}"
    return ""


# Every table on the page goes through show_table, which renders the formatted
# view and registers the raw numbers for the CSV downloads in the footer.
TABLES: dict[str, pd.DataFrame] = {"dl_strip": pd.DataFrame(
    [{"series": t["key"], "value": (f"{t['value'][0]}-{t['value'][1]}" if t["kind"] == "range"
                                    else t["value"]),
      "change": t["change"], "unit_change": "bn" if t["kind"] == "bn" else "bp",
      "date": pd.Timestamp(t["date"]).date()} for t in TILES])}


def show_table(key: str, view: pd.DataFrame, raw: pd.DataFrame, signed=(), **kw) -> None:
    TABLES[key] = raw
    styled = view.style.map(signed_style, subset=list(signed)) if signed else view
    st.dataframe(styled, hide_index=True, width="stretch", row_height=kw.pop("row_height", 28),
                 **kw)


CONTEXT_DAYS = 5   # beyond this the forward path's gain over AR(1) is about zero


def forward_panel() -> None:
    """The next 20 settlement days against the top of the target range.

    Turn days carry the calibrated turn quantiles at the matching horizon
    (median dot, 90th percentile tick, bar to the 95th): the pooled 80% range
    covers only 56 to 63% of turns in forward_backtest.csv. Ordinary days are
    a muted context line from the AR decay, faded beyond five days where the
    model's gain over AR(1) is about zero."""
    with st.container(border=True):
        st.subheader(T["fwd_title"], anchor="forward")
        if FWD is None:
            st.info(T["fwd_none"])
            return
        f = FWD.copy()
        f["lab"] = f["date"].map(lambda x: f"{x.day} {T['months'][x.month - 1]}")
        f["kind"] = f["kind"].fillna("")
        f["mark"] = f["kind"].map({"month-end": "M", "quarter-end": "Q", "year-end": "Y", "": ""})
        f["date_s"] = date_col(f["date"])
        # Everything on one axis: SOFR minus the top of the range, bp. The
        # ordinary path is a deviation from the 60-day median, so add it back.
        f["y"] = f["base_bp"] + f["forecast_bp"]
        f["y_s"] = fmt_col(f["y"], 1, sign=True)
        f["faded"] = (f["h"] > CONTEXT_DAYS).astype(int)
        f["cat"] = np.where(f["kind"] != "", T["fwd_turnq"], T["fwd_ctx"])
        turns = []
        for _, r in f[f["kind"].isin(["month-end", "quarter-end"])].iterrows():
            q = roll_cost.turn_quantiles(COEFS, r["kind"], STATE.scarcity, STATE.base_bp,
                                         roll_cost.days_until(STATE, r["date"]), POLICY)
            sc = TSCORE.loc[(r["kind"], q["horizon"])]
            turns.append({"lab": r["lab"], "date_s": r["date_s"], "h": r["h"], "mark": r["mark"],
                          "q50": q["q50"], "q90": q["q90"], "q95": q["q95"],
                          "q_s": " / ".join(n(q[k], 1, sign=True) for k in ("q50", "q90", "q95")),
                          "hz": q["horizon"],
                          "cov_s": "/".join(n(100 * sc[c], 0) for c in
                                            ("cover_q50", "cover_q90", "cover_q95")) + "\u00a0%",
                          "cat": T["fwd_turnq"]})
        tq = pd.DataFrame(turns)
        TABLES["dl_forward"] = FWD.assign(
            sofr_minus_top_bp=f["y"], context_only=(f["kind"] == "").astype(int),
            faded=f["faded"]).merge(
            tq[["h", "q50", "q90", "q95", "hz"]].rename(columns={"hz": "turn_horizon"})
            if len(tq) else pd.DataFrame(columns=["h"]), on="h", how="left")

        order = f["lab"].tolist()
        x = alt.X("lab:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45))
        y = alt.Y("y:Q", title=T["fwd_y_top"])
        col = alt.Color("cat:N", title=None, legend=alt.Legend(orient="top", labelLimit=0, columns=1),
                        scale=alt.Scale(domain=[T["fwd_ctx"], T["fwd_turnq"]], range=[MUTED, RED]))
        tt_ctx = [alt.Tooltip("date_s:N", title=T["tt_date"]), alt.Tooltip("h:Q", title=T["tt_h"]),
                  alt.Tooltip("y_s:N", title=T["fwd_y_top"]), alt.Tooltip("cat:N", title=T["tt_status"])]
        ctx = f[f["kind"] == ""]
        line = (alt.Chart(ctx).mark_line(strokeWidth=1.5, strokeDash=[4, 3], color=MUTED)
                .encode(x=x, y=y, opacity=alt.value(0.7)))
        dots = (alt.Chart(ctx).mark_point(filled=True, size=40)
                .encode(x=x, y=y, color=col, tooltip=tt_ctx,
                        opacity=alt.condition("datum.faded == 1", alt.value(0.3), alt.value(0.9))))
        layers = line + dots
        if len(tq):
            tt_turn = [alt.Tooltip("date_s:N", title=T["tt_date"]), alt.Tooltip("h:Q", title=T["tt_h"]),
                       alt.Tooltip("q_s:N", title=T["fwd_q_tt"]), alt.Tooltip("hz:Q", title=T["c_h"]),
                       alt.Tooltip("cov_s:N", title=T["tt_cov"])]
            bar = (alt.Chart(tq).mark_rule(strokeWidth=6, opacity=0.35, color=RED)
                   .encode(x=x, y=alt.Y("q50:Q"), y2="q95:Q", tooltip=tt_turn))
            tick = (alt.Chart(tq).mark_tick(thickness=2, size=16, color=RED)
                    .encode(x=x, y="q90:Q", tooltip=tt_turn))
            med = (alt.Chart(tq).mark_point(filled=True, size=80)
                   .encode(x=x, y="q50:Q", color=col, tooltip=tt_turn))
            lab = (alt.Chart(tq).mark_text(dy=-10, font=FONT, fontSize=11, fontWeight="bold", color=RED)
                   .encode(x=x, y="q95:Q", text="mark:N"))
            layers = layers + bar + tick + med + lab
        top = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeWidth=1).encode(y="y:Q")
        st.altair_chart(style((top + layers).properties(height=250)), width="stretch")

        sc = pd.read_csv(PROC / "forward_backtest.csv")
        turn = sc[sc["days"] == "turn days"]
        other = sc[sc["days"] == "other days"].set_index("h")["r2_vs_ar1"]
        held = desk.long_date(LIVE_ROW["tga_date"], LANG) if LIVE_ROW is not None else "—"
        st.caption(T["fwd_caption"].format(
            tmin=n(100 * turn["coverage80"].min(), 0), tmax=n(100 * turn["coverage80"].max(), 0),
            r1=n(other[1], 3, sign=True), r5=n(other[5], 3, sign=True),
            r10=n(other[10], 3, sign=True), days=CONTEXT_DAYS, date=held))


def track_tile() -> None:
    """The same-day call, the latest scored forecasts and the running hit rate."""
    with st.container(border=True):
        st.subheader(T["track_title"])
        if LIVE_ROW is None:
            st.info(T["desk_none"])
            return
        row = LIVE_ROW
        if LIVE_SOURCE == "local":
            st.caption(T["desk_src_local"])
        target = desk.long_date(row["target_date"], LANG)
        if desk.is_stale(row["target_date"], os.environ.get("REPO_TODAY")):
            st.warning(T["desk_stale"].format(target=target))
        if int(row.get("range_unconfirmed", 0) or 0):
            st.warning(T["desk_unconfirmed"].format(target=target))
        st.markdown(T["track_call"].format(date=target))
        k1, k2 = st.columns(2)
        k1.metric(T["desk_forecast"], f"{n(row['forecast_bp'], 1, sign=True)}\u00a0{BP}")
        k1.caption(T["desk_range"].format(lo=n(row["lower_bp"], 1, sign=True),
                                          hi=n(row["upper_bp"], 1, sign=True)))
        if pd.notna(row.get("implied_sofr_pct")):
            k2.metric(T["desk_sofr"], f"{n(row['implied_sofr_pct'], 2)}\u00a0%")
        scored = LIVE.dropna(subset=["realised_bp"]).tail(5).iloc[::-1]
        start = desk.long_date(LIVE["target_date"].min(), LANG)
        if len(scored):
            view = pd.DataFrame({
                T["tr_target"]: date_col(scored["target_date"]),
                T["tr_fc"]: fmt_col(scored["forecast_bp"], 1, sign=True),
                T["tr_range"]: fmt_col(scored["lower_bp"], 1, sign=True) + " ; "
                + fmt_col(scored["upper_bp"], 1, sign=True),
                T["tr_real"]: fmt_col(scored["realised_bp"], 1, sign=True),
                T["tr_hit"]: [T["yes"] if h == 1 else T["no"] for h in scored["hit"]]})
            show_table("dl_track", view, LIVE.dropna(subset=["realised_bp"]),
                       signed=[T["tr_fc"], T["tr_real"]])
            all_scored = LIVE.dropna(subset=["realised_bp"])
            st.caption(T["track_hit"].format(hit=n(100 * all_scored["hit"].mean(), 0),
                                             n=len(all_scored), start=start))
        else:
            st.caption(T["track_none_scored"].format(start=start))
        tr = desk.track_record(LIVE)
        if "hit" in tr:   # RMSE against AR(1) once 20 days are scored
            st.caption(T["desk_track"].format(n=tr["n"], start=start, hit=n(100 * tr["hit"], 0),
                                              rmse=n(tr["rmse"], 2), rmse_ar1=n(tr["rmse_ar1"], 2)))
        with st.expander(T["track_note_title"]):
            flags = desk.flags_in_play(row, LANG)
            st.markdown(T["desk_flags"].format(flags=", ".join(flags) or T["desk_no_flags"]))
            line1, line2 = desk.morning_note(row, LANG)
            st.markdown(f"> {line1}  \n> {line2}")
            st.caption(T["desk_asof"].format(run=desk.run_time_et(row["run_time_utc"], LANG),
                                             asof=desk.long_date(row["as_of_date"], LANG)))


def analog_table(kind: str) -> None:
    a = analogs.analogs(daily, {**STATE.scarcity, "base_bp": STATE.base_bp}, kind, today())
    view = pd.DataFrame({
        T["an_date"]: date_col(a["date"]),
        T["an_rrp"]: fmt_col(a["rrp_bn_lag1"], 1),
        T["an_tga"]: fmt_col(a["tga_bn_lag1"], 0),
        T["an_base"]: fmt_col(a["base_bp"], 1, sign=True),
        T["an_dist"]: fmt_col(a["distance"], 2),
        T["an_print"]: fmt_col(a["printed_bp"], 0, sign=True),
        T["an_p99"]: fmt_col(a["p99_bp"], 0, sign=True),
    })
    show_table("dl_analogs", view, a, signed=[T["an_base"], T["an_print"], T["an_p99"]])


RAW_MEDIANS: list[tuple[str, int, float]] = []   # (type, horizon, share at or below)


def turns_table(turns) -> pd.DataFrame:
    """One row per upcoming turn, numbers formatted for the active language;
    the raw numbers go to TABLES for the CSV download."""
    rows, raw = [], []
    dash = "—"
    for d, kind in turns:
        left = roll_cost.days_until(STATE, d)
        row = {T["tt_col_date"]: date_short(d, LANG), T["tt_col_type"]: T[f"turn_{kind}"],
               T["tt_col_days"]: str(left)}
        r = {"date": pd.Timestamp(d).date(), "type": kind, "days_left": left}
        if kind == "year-end":
            r.update(note="no model, history only")
            row.update({T["tt_col_p50"]: dash, T["tt_col_p90"]: dash, T["tt_col_p95"]: dash,
                        T["tt_col_sofr95"]: dash, T["tt_col_cov"]: T["turns_ye_cell"],
                        T["tt_col_n"]: str(int((daily["year_end"] == 1).sum()))})
        else:
            q = roll_cost.turn_quantiles(COEFS, kind, STATE.scarcity, STATE.base_bp, left, POLICY)
            sc = TSCORE.loc[(kind, q["horizon"])]
            if not q["median_calibrated"]:
                RAW_MEDIANS.append((kind, q["horizon"], q["median_at_or_below"]))
            r.update(horizon=q["horizon"], q50_bp=q["q50"], q90_bp=q["q90"], q95_bp=q["q95"],
                     median="raw" if not q["median_calibrated"] else "calibrated",
                     median_at_or_below=q["median_at_or_below"],
                     sofr_at_q95_pct=STATE.target_hi + q["q95"] / 100,
                     cover_q50=sc["cover_q50"], cover_q90=sc["cover_q90"],
                     cover_q95=sc["cover_q95"], n=int(sc["n"]))
            row.update({
                T["tt_col_p50"]: n(q["q50"], 1, sign=True) + ("\u00a0†" if not q["median_calibrated"]
                                                               else ""),
                T["tt_col_p90"]: n(q["q90"], 1, sign=True),
                T["tt_col_p95"]: n(q["q95"], 1, sign=True),
                T["tt_col_sofr95"]: f"{n(STATE.target_hi + q['q95'] / 100, 2)}\u00a0%",
                T["tt_col_cov"]: "/".join(n(100 * sc[c], 0) for c in
                                          ("cover_q50", "cover_q90", "cover_q95")),
                T["tt_col_n"]: str(int(sc["n"])),
            })
        rows.append(row)
        raw.append(r)
    TABLES["dl_turns"] = pd.DataFrame(raw)
    return pd.DataFrame(rows)


def turns_panel() -> None:
    with st.container(border=True):
        st.subheader(T["turns_title"], anchor="turns")
        turns = roll_cost.upcoming_turns(today(), "2026-12-31")
        if not turns:
            st.info(T["turn_none"])
            return
        tab = turns_table(turns)
        widths = {"tt_col_date": 92, "tt_col_type": 84, "tt_col_days": 46, "tt_col_p50": 60,
                  "tt_col_p90": 60, "tt_col_p95": 60, "tt_col_sofr95": 88, "tt_col_cov": 100,
                  "tt_col_n": 36}
        show_table("dl_turns", tab, TABLES["dl_turns"],
                   signed=[T["tt_col_p50"], T["tt_col_p90"], T["tt_col_p95"]], row_height=30,
                   column_config={T[k]: st.column_config.TextColumn(width=w)
                                  for k, w in widths.items()})
        # Does the calibrated range beat the naive range at this turn's horizon?
        cmp_turn = next(((d, k) for d, k in turns if k in ("month-end", "quarter-end")), None)
        if cmp_turn is not None:
            d1, k1 = cmp_turn
            h1 = roll_cost.turn_quantiles(COEFS, k1, STATE.scarcity, STATE.base_bp,
                                          roll_cost.days_until(STATE, d1), POLICY)["horizon"]
            nrg = pd.read_csv(PROC / "turn_naive_ranges.csv").set_index(["type", "horizon", "method"])
            mod, nai = nrg.loc[(k1, h1, "calibrated model")], nrg.loc[(k1, h1, "naive range")]
            if mod["pinball_q90"] < nai["pinball_q90"]:
                verdict = T["vs_beats"]
            elif abs(mod["at_or_below_q90"] - 0.9) < abs(nai["at_or_below_q90"] - 0.9):
                verdict = T["vs_loses"]
            else:
                verdict = T["vs_loses_both"]
            st.caption(T["turns_vs_naive"].format(
                h=h1, n=int(mod["n"]), kinds=T[f"turns_{k1}"], verdict=verdict,
                pm=n(mod["pinball_q90"], 2), pn=n(nai["pinball_q90"], 2),
                cm=n(100 * mod["at_or_below_q90"], 0), cn=n(100 * nai["at_or_below_q90"], 0)))
        for kind, h, share in RAW_MEDIANS:
            st.caption(T["turns_raw_median"].format(kinds=T[f"turns_{kind}"], h=h,
                                                    above=n(100 * (1 - share), 0)))
        # Client note on the first upcoming turn: the numbers come from the
        # table above and the analog table, the words from a fixed template.
        d0, k0 = turns[0]
        left0 = roll_cost.days_until(STATE, d0)
        q0 = (None if k0 == "year-end" else
              roll_cost.turn_quantiles(COEFS, k0, STATE.scarcity, STATE.base_bp, left0, POLICY))
        an0 = analogs.analogs(daily, {**STATE.scarcity, "base_bp": STATE.base_bp}, k0, today())
        note = desk.client_note(k0, d0, left0, q0, an0["printed_bp"], LANG)
        st.markdown(f"**{T['client_title']}**")
        st.code(note, language=None, wrap_lines=True)
        st.caption(T["turns_state"].format(rrp=n(STATE.scarcity["rrp_bn_lag1"], 1),
                                           tga=n(STATE.scarcity["tga_bn_lag1"], 0),
                                           base=n(STATE.base_bp, 1, sign=True)))
        labels = [f"{desk.long_date(d, LANG)} · {T[f'turn_{k}']}" for d, k in turns]
        with st.expander(T["analogs_title"].format(kinds=T["turns_" + turns[0][1]])
                         if len(turns) == 1 else T["analog_pick"]):
            pick = st.selectbox(T["analog_pick"], range(len(turns)), format_func=labels.__getitem__,
                                key="analog_turn", label_visibility="collapsed")
            kind = turns[pick][1]
            if kind == "year-end":
                st.caption(T["turn_ye"].format(n=int((daily["year_end"] == 1).sum())))
            analog_table(kind)


def year_end_block(res: dict) -> None:
    """Past year-end prints, labelled as history, and the break-even print."""
    hist = roll_cost.year_end_history(daily)
    p_ = hist["printed_bp"]
    for d in res["year_end"]:
        st.markdown(f"**{T['calc_ye_title'].format(date=desk.long_date(d, LANG))}**")
    tab = pd.DataFrame({T["ye_year"]: hist["date"].dt.year.astype(str),
                        T["an_date"]: date_col(hist["date"]),
                        T["ye_print"]: fmt_col(p_, 0, sign=True)})
    show_table("dl_ye", tab, hist, signed=[T["ye_print"]])
    st.caption(T["calc_ye_stats"].format(min=n(p_.min(), 0, sign=True),
                                         max=n(p_.max(), 0, sign=True),
                                         med=n(p_.median(), 0, sign=True), n=len(p_)))
    if "ye_breakeven_bp" in res:
        st.caption(T["calc_ye_be"].format(d=res["ye_days"],
                                          be=n(res["ye_breakeven_bp"], 0, sign=True)))
    if "central_pct" in res:
        st.caption(T["calc_ye_rest"])


CALC_NOTE_ARGS: dict = {}   # the full calculator note goes to Method


def calc_panel() -> None:
    first = max(STATE.anchor_date, today())
    first = first if market_calendar.is_business_day(first) else market_calendar.next_business_day(first)
    with st.container(border=True):
        st.subheader(T["calc_title"], anchor="term-roll")
        fmt = "DD/MM/YYYY" if LANG == "fr" else "YYYY/MM/DD"
        c1, c2 = st.columns(2)
        s_ = c1.date_input(T["calc_start"], value=first.date(), format=fmt, key="calc_start")
        e_ = c2.date_input(T["calc_end"], value=(first + pd.Timedelta(days=28)).date(),
                           format=fmt, key="calc_end")
        q_ = st.number_input(T["calc_quote"], value=round(STATE.target_hi + STATE.base_bp / 100, 2),
                             step=0.01, format="%.2f", key="calc_quote")
        res = roll_cost.price(s_, e_, q_, STATE, COEFS, forward=FWD)
        if res["days"].empty:
            st.warning(T["calc_empty"])
            return
        if res["too_early"]:
            st.warning(T["calc_early"].format(date=desk.long_date(STATE.anchor_date, LANG)))
        if res["fomc"]:
            st.warning(T["calc_fomc"].format(
                dates=", ".join(desk.long_date(x, LANG) for x in res["fomc"])))
        if res["year_end"] and not res["fomc"] and not res["too_early"]:
            year_end_block(res)
        if "central_pct" not in res:
            return
        m1, m2 = st.columns(2)
        m1.metric(T["calc_central"], f"{n(res['central_pct'], 3)}\u00a0%")
        m2.metric(T["calc_p90"], f"{n(res['p90_pct'], 3)}\u00a0%")
        m3, m4 = st.columns(2)
        m3.metric(T["calc_be"], f"{n(res['breakeven_bp'], 1, sign=True)}\u00a0{BP}")
        m4.metric(T["calc_be_p90"], f"{n(res['breakeven_p90_bp'], 1, sign=True)}\u00a0{BP}")
        v = roll_cost.verdict(res)
        if v:
            args = {"q": n(q_, 2), "be": n(abs(res["breakeven_bp"]), 1),
                    "be90": n(abs(res["breakeven_p90_bp"]), 1)}
            st.markdown(f"**{T['calc_' + v].format(**args)}**")
        r = res["rates"]
        CALC_NOTE_ARGS.update(n=res["n_days"], k=len(r), t=int(r["kind"].notna().sum()))
        st.caption(T["calc_note_short"].format(**CALC_NOTE_ARGS))


# The forward calendar leads; then Next turns, the calculator and the small
# track-record tile. On a phone they stack in that order.
forward_panel()
col_turns, col_calc, col_track = st.columns([1.75, 1.05, 0.95])
with col_turns:
    turns_panel()
with col_calc:
    calc_panel()
with col_track:
    track_tile()

# ------------------------------------------------------- calendar: turn map
rec = oos[oos["date"].dt.year >= 2024]
dmt = pd.read_csv(PROC / "dm_test.csv").set_index("period")
DM_ARGS = {"p24": n(dmt.loc["2024+", "p_two_sided"], 4),
           "pall": n(dmt.loc["all", "p_two_sided"], 2),
           "r2all": n(dmt.loc["all", "r2_vs_ar1"], 3, sign=True)}

st.subheader(T["s2"], anchor="calendar")
st.caption(T["cap_short_calendar"])

# Diverging scale centred at 0 from the design's validated colours: blue
# below the top of the range, the neutral band colour at 0, red above.
LIM = 25
DIV = alt.Scale(domain=[-LIM, 0, LIM], range=[BLUE, BAND, RED], clamp=True)
TEXT_ON = "#FFFFFF" if MODE == "light" else C["bg"]   # text on saturated tiles
TT_TILE = [alt.Tooltip("date_s:N", title=T["tt_date"]),
           alt.Tooltip("v_s:N", title=T["tt_value"]),
           alt.Tooltip("mark_s:N", title=T["tt_mark"])]


def tile_text(data, x, y) -> alt.Chart:
    return (alt.Chart(data).mark_text(font=FONT, fontSize=10)
            .encode(x=x, y=y, text="lab:N", tooltip=TT_TILE,
                    color=alt.condition(f"abs(datum.v) > {LIM * 0.55}",
                                        alt.value(TEXT_ON), alt.value(INK))))


TURN_MARK = {"year-end": "Y", "quarter-end": "Q", "month-end": "M"}
hm = daily[daily["month_end"] == 1].copy()
hm["year"] = hm["date"].dt.year.astype(str)
hm["mon"] = hm["date"].dt.month.map(lambda m: T["months"][m - 1])
hm["kind"] = np.where(hm["year_end"] == 1, "year-end",
                      np.where(hm["quarter_end"] == 1, "quarter-end", "month-end"))
hm["mark"] = np.where(hm["kind"] == "month-end", "", hm["kind"].map(TURN_MARK))
hm["mark_s"] = hm["kind"].map(lambda k: T[f"turn_{k}"])
hm["v"] = hm["spread_hi_bp"]
hm["v_s"] = fmt_col(hm["v"], 0, sign=True)
hm["lab"] = hm["v_s"] + np.where(hm["mark"] != "", " " + hm["mark"], "")
hm["date_s"] = date_col(hm["date"])
mx = alt.X("mon:O", sort=T["months"], title=None, axis=alt.Axis(labelAngle=0, orient="top"))
my = alt.Y("year:O", title=None)
turn_map = (alt.Chart(hm).mark_rect(cornerRadius=3, stroke=C["bg"], strokeWidth=1.5)
            .encode(x=mx, y=my, tooltip=TT_TILE,
                    color=alt.Color("v:Q", scale=DIV,
                                    legend=alt.Legend(title=BP, orient="bottom",
                                                      gradientLength=140, direction="horizontal"))))

# The forward calendar at the top of the page replaces the old 40-day strip.
st.altair_chart(style((turn_map + tile_text(hm, mx, my))
                      .properties(height=30 * hm["year"].nunique() + 120, title=T["ch_turnmap"])),
                width="stretch")

# Month-end premium by year and annual ON RRP usage, compact.
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
bars = (alt.Chart(prem).mark_bar(color=BLUE, cornerRadiusTopLeft=3,
                                 cornerRadiusTopRight=3, size=20)
        .encode(x=alt.X("year:O", title=None),
                y=alt.Y("premium_bp:Q", title=T["ax_bp"]),
                tooltip=[alt.Tooltip("year:O", title=T["tt_year"]),
                         alt.Tooltip("premium_s:N", title=T["tt_premium"])]))
labels = (alt.Chart(prem).mark_text(dy=-7, font=FONT, fontSize=10, color=INK)
          .encode(x="year:O", y="premium_bp:Q", text="label_s:N"))
left.altair_chart(style((bars + labels).properties(height=220, title=T["ch_premium"])),
                  width="stretch")
ubars = (alt.Chart(rrp).mark_bar(color=ORANGE, cornerRadiusTopLeft=3,
                                 cornerRadiusTopRight=3, size=20)
         .encode(x=alt.X("year:O", title=None),
                 y=alt.Y("rrp_bn:Q", title=T["ax_rrp"]),
                 tooltip=[alt.Tooltip("year:O", title=T["tt_year"]),
                          alt.Tooltip("rrp_s:N", title=T["tt_rrp"])]))
right.altair_chart(style(ubars.properties(height=220, title=T["ch_rrp"])), width="stretch")

st.divider()

# ------------------------------------------------------------- the corridor
st.subheader(T["s1"], anchor="corridor")
st.caption(T["cap_short_corridor"])
lo, hi = daily["date"].min().to_pydatetime(), daily["date"].max().to_pydatetime()
# A fixed key keeps the chosen range when the language, and so the label,
# changes.
start, end = st.slider(T["date_range"], min_value=lo, max_value=hi, value=(lo, hi),
                       format=T["slider_format"], key="range")
d = daily[(daily["date"] >= start) & (daily["date"] <= end)]

# Four small multiples on one explicit date domain and one y-axis gutter, so
# the panels line up by date in each row.
X_DATE = alt.X("date:T", title=None,
               scale=alt.Scale(domain=[pd.Timestamp(start).isoformat(),
                                       pd.Timestamp(end).isoformat()]))
Y_GUTTER = 48
GUTTER = alt.Axis(minExtent=Y_GUTTER, maxExtent=Y_GUTTER)

band = (alt.Chart(d).mark_area(opacity=0.55, color=BAND)
        .encode(x=X_DATE,
                y=alt.Y("target_lo:Q", title=T["ax_percent"], scale=alt.Scale(zero=False),
                        axis=GUTTER),
                y2="target_hi:Q"))
# Standing Repo Facility rate from data/srf_rates.csv, one row per FOMC
# implementation note since the facility opened on 29 July 2021.
d = pd.merge_asof(d.sort_values("date"),
                  SRF_RATES[["effective_date", "srf_rate_pct"]].rename(
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
lines = (alt.Chart(rates).mark_line(strokeWidth=1.6)
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
# 17 September 2019 reaches 325 bp against a normal range of about -10 to +30,
# so the position axis is clipped and the spike is labelled rather than dropped.
inside = d.dropna(subset=["carry_bp"]).copy()
inside["date_s"] = date_col(inside["date"])
inside["carry_s"] = fmt_col(inside["carry_bp"], 1)
CLIP_HI = 70
pos = (alt.Chart(inside).mark_line(strokeWidth=1.6, color=BLUE, clip=True)
       .encode(x=X_DATE,
               y=alt.Y("carry_bp:Q", title=T["ax_position"].format(clip=CLIP_HI),
                       scale=alt.Scale(domain=[-15, CLIP_HI]), axis=GUTTER),
               tooltip=[alt.Tooltip("date_s:N", title=T["tt_date"]),
                        alt.Tooltip("carry_s:N", title=T["tt_carry"])]))
if len(inside):
    top = inside.loc[[inside["carry_bp"].idxmax()]].copy()
    top["lab"] = fmt_col(top["carry_bp"], 0) + " " + BP
    top["y"] = CLIP_HI - 7
    pos = pos + (alt.Chart(top).mark_text(align="left", dx=6, font=FONT, fontSize=10, color=RED)
                 .encode(x="date:T", y=alt.Y("y:Q"), text="lab:N"))
floor = (alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeWidth=1)
         .encode(y="y:Q"))

usage = d.dropna(subset=["rrp_bn_t"]).copy()
usage["date_s"] = date_col(usage["date"])
usage["rrp_s"] = fmt_col(usage["rrp_bn_t"], 0)
rrp_daily = (alt.Chart(usage).mark_area(color=ORANGE, opacity=0.75)
             .encode(x=X_DATE, y=alt.Y("rrp_bn_t:Q", title=T["ax_srf"], axis=GUTTER),
                     tooltip=[alt.Tooltip("date_s:N", title=T["tt_date"]),
                              alt.Tooltip("rrp_s:N", title=T["tt_rrp"])]))
srf = SRF_TAKE[(SRF_TAKE["date"] >= start) & (SRF_TAKE["date"] <= end)].copy()
srf["date_s"] = date_col(srf["date"])
srf["srf_s"] = fmt_col(srf["srf_bn"], 1)
take = (alt.Chart(srf).mark_bar(color=RED, width=2)
        .encode(x=X_DATE, y=alt.Y("srf_bn:Q", title=T["ax_srf"], axis=GUTTER),
                tooltip=[alt.Tooltip("date_s:N", title=T["tt_date"]),
                         alt.Tooltip("srf_s:N", title=T["tt_srf"])]))

g1, g2 = st.columns(2)
g1.altair_chart(style((band + lines).properties(height=260, title=T["ch_band"])), width="stretch")
g2.altair_chart(style((pos + floor).properties(height=260, title=T["ch_position"])),
                width="stretch")
g3, g4 = st.columns(2)
g3.altair_chart(style(rrp_daily.properties(height=220, title=T["ch_rrp_daily"])), width="stretch")
g4.altair_chart(style(take.properties(height=220, title=T["ch_srf"])), width="stretch")

st.divider()

# -------------------------------------------------------------- the model
st.subheader(T["s3"], anchor="model")
st.caption(T["cap_short_model"])
k1, k2 = st.columns(2)
k1.metric(T["tile_days"], n(len(daily)),
          help=T["tile_days_help"].format(last=date_short(daily["date"].max(), LANG)))
k2.metric(T["tile_skill"], n(r2(rec["y"], rec["two_stage"], rec["ar1"]), 3, sign=True),
          help=T["tile_skill_help"].format(**DM_ARGS))


def pval(p) -> str:
    return T["p_undefined"] if pd.isna(p) else n(p, 4)


# Every comparison table has the naive benchmark (yesterday's value carried
# forward, src/dm_test.py) as its first row.
dm = pd.read_csv(PROC / "dm_test.csv")
dmn = pd.read_csv(PROC / "dm_test_naive.csv")
PERIOD = {"all": T["g_all"], "turn days": T["g_turn_days"], "other days": T["g_other"]}
mn = dmn[(dmn["forecaster"] == "model") & (dmn["benchmark"] == "naive")].set_index("group")
periods = ["all", "2024+"] + [p_ for p_ in dm["period"] if p_.isdigit()]
rmse_raw = pd.DataFrame({
    "forecaster": ["naive", "AR(1)", "two-stage LASSO"],
    **{p_: [mn.at[p_, "rmse_benchmark_bp"], dm.set_index("period").at[p_, "rmse_ar1_bp"],
            dm.set_index("period").at[p_, "rmse_model_bp"]] for p_ in periods}})
r2_view = pd.DataFrame({T["c_forecaster"]: [T["naive_label"], T["rule_ar1"], T["rule_lasso"]],
                        **{PERIOD.get(p_, p_): fmt_col(rmse_raw[p_], 2) for p_ in periods}})
# Diebold-Mariano: the model against each benchmark, naive first.
dm_long = pd.concat([
    mn.reset_index().rename(columns={"group": "period", "r2_vs_benchmark": "r2"})
      .assign(benchmark="naive"),
    dm.rename(columns={"r2_vs_ar1": "r2"}).assign(benchmark="AR(1)"),
], ignore_index=True)[["benchmark", "period", "n", "r2", "hln_stat", "p_two_sided"]]
dm_view = pd.DataFrame({
    T["c_benchmark"]: dm_long["benchmark"].map({"naive": T["naive_label"], "AR(1)": T["rule_ar1"]}),
    T["c_period"]: dm_long["period"].map(lambda x: PERIOD.get(x, x)),
    T["c_n"]: dm_long["n"].astype(str),
    T["c_r2_bench"]: fmt_col(dm_long["r2"], 3, sign=True),
    T["c_hln"]: [T["p_undefined"] if pd.isna(v) else n(v, 2, sign=True) for v in dm_long["hln_stat"]],
    T["c_p"]: [pval(v) for v in dm_long["p_two_sided"]]})
cov_raw = pd.read_csv(PROC / "interval_coverage.csv")
GROUP = {"all": T["g_all"], "month-end": T["g_me"], "other days": T["g_other"],
         "quarter-end": T["g_qe"], "month-end, 2024+": T["g_me24"],
         "other days, 2024+": T["g_other24"]}
cov_view = pd.DataFrame({T["c_group"]: cov_raw["group"].map(lambda g: GROUP.get(g, g)),
                         T["c_n"]: cov_raw["n"].astype(str),
                         T["c_cover"]: fmt_col(100 * cov_raw["coverage"], 1) + " %",
                         T["c_below"]: fmt_col(100 * cov_raw["below"], 1) + " %",
                         T["c_above"]: fmt_col(100 * cov_raw["above"], 1) + " %",
                         T["c_width"]: fmt_col(cov_raw["mean_width_bp"], 2)})
hz = pd.read_csv(PROC / "turn_quantile_horizons.csv")
hz = hz[hz["type"].isin(["month-end", "quarter-end"]) & hz["version"].isin(["raw", "calibrated"])]
# The naive range (src/turn_quantiles.py) leads each type and horizon.
nr = pd.read_csv(PROC / "turn_naive_ranges.csv")
nr = nr[(nr["method"] == "naive range") & nr["type"].isin(["month-end", "quarter-end"])]
nr = nr.rename(columns={"at_or_below_q50": "cover_q50", "at_or_below_q90": "cover_q90",
                        "at_or_below_q95": "cover_q95"}).assign(version="naive")
hz = pd.concat([nr[["type", "horizon", "version", "n", "cover_q50", "cover_q90", "cover_q95",
                    "pinball_q90"]], hz], ignore_index=True)
hz["order"] = hz["version"].map({"naive": 0, "raw": 1, "calibrated": 2})
hz = hz.sort_values(["type", "horizon", "order"]).drop(columns="order").reset_index(drop=True)
hz_view = pd.DataFrame({
    T["c_type"]: hz["type"].map(lambda k: T[f"turn_{k}"]),
    T["c_h"]: hz["horizon"].astype(str),
    T["c_version"]: hz["version"].map({"naive": T["naive_label"], "raw": T["v_raw"],
                                       "calibrated": T["v_cal"]}),
    T["c_n"]: hz["n"].astype(str),
    T["c_cov3"]: ["/".join(n(100 * r[c], 0) for c in ("cover_q50", "cover_q90", "cover_q95"))
                  for _, r in hz.iterrows()],
    T["c_pin90"]: fmt_col(hz["pinball_q90"], 2)})

t1, t2 = st.columns(2)
with t1:
    st.markdown(f"**{T['tb_rmse']}**")
    show_table("dl_r2", r2_view, rmse_raw)
with t2:
    st.markdown(f"**{T['tb_dm']}**")
    # Only R² is coloured: a negative DM statistic means the model wins, so a
    # red/blue sign colour would read backwards there.
    show_table("dl_dm", dm_view, dm_long, signed=[T["c_r2_bench"]], height=320)
t3, t4 = st.columns(2)
with t3:
    st.markdown(f"**{T['tb_cov']}**")
    show_table("dl_cov", cov_view, cov_raw)
with t4:
    st.markdown(f"**{T['tb_turns']}**")
    show_table("dl_turnscores", hz_view, hz, height=260)

# R² by year as bars, and the forecast with its 80% interval.
o = oos.copy()
o["year"] = o["date"].dt.year
sk = (o.groupby("year")
      .apply(lambda g: r2(g["y"], g["two_stage"], g["ar1"]), include_groups=False)
      .rename("r2").reset_index())
sk["sign"] = np.where(sk["r2"] >= 0, "adds", "hurts")   # internal, not shown
sk["r2_s"] = fmt_col(sk["r2"], 3, sign=True)
sk["label_s"] = fmt_col(sk["r2"], 2, sign=True)
dv = (alt.Chart(sk).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                             cornerRadiusBottomLeft=3, cornerRadiusBottomRight=3, size=20)
      .encode(x=alt.X("year:O", title=None),
              y=alt.Y("r2:Q", title=T["ax_r2"]),
              color=alt.Color("sign:N", title=None,
                              scale=alt.Scale(domain=["adds", "hurts"], range=[BLUE, RED]),
                              legend=None),
              tooltip=[alt.Tooltip("year:O", title=T["tt_year"]),
                       alt.Tooltip("r2_s:N", title=T["tt_r2"])]))
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeWidth=1).encode(y="y:Q")


# dy is a mark property, not an encoding channel, so the label sits above a
# positive bar and below a negative one via two layers rather than a condition.
def dlabel(frame: pd.DataFrame, dy: int) -> alt.Chart:
    return (alt.Chart(frame).mark_text(dy=dy, font=FONT, fontSize=10, color=INK)
            .encode(x="year:O", y="r2:Q", text="label_s:N"))


dlab = dlabel(sk[sk["r2"] >= 0], -7) + dlabel(sk[sk["r2"] < 0], 12)
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
fline = (alt.Chart(fan).mark_line(strokeWidth=1.4, color=BLUE)
         .encode(x=fx, y="two_stage:Q", tooltip=fan_tt))
fdots = (alt.Chart(fan).mark_point(filled=True, size=14)
         .encode(x=fx, y="y:Q",
                 color=alt.Color("inside:N", title=None,
                                 scale=alt.Scale(domain=[T["fan_inside"], T["fan_outside"]],
                                                 range=[MUTED, RED]),
                                 legend=alt.Legend(orient="top")),
                 tooltip=fan_tt))
c1, c2 = st.columns(2)
c1.altair_chart(style((dv + zero + dlab).properties(height=240, title=T["ch_r2"])),
                width="stretch")
c2.altair_chart(style((fband + fline + fdots).properties(height=240, title=T["ch_fan"])),
                width="stretch")

st.markdown(f"**{T['s3b']}**")
imp = pd.read_csv(PROC / "permutation_importance.csv")
KINDS = [T["kind_one"], T["kind_group"]]
imp["kind"] = np.where(imp["n_cols"] > 1, KINDS[1], KINDS[0])
# Group rows get a translated name; single predictors keep their variable name.
imp["what"] = imp["what"].map(lambda w: T["groups"].get(w, w))
for col, dec in (("mean_bp", 4), ("p10", 4), ("p90", 4)):
    imp[col + "_s"] = fmt_col(imp[col], dec, sign=True)
TABLES["dl_importance"] = imp[["what", "n_cols", "mean_bp", "p10", "p90"]]
order = imp.sort_values("mean_bp", ascending=False)["what"].tolist()
# 19 categories: Altair drops labels to avoid overlap unless told not to.
enc_y = alt.Y("what:N", sort=order, title=None,
              axis=alt.Axis(labelOverlap=False, labelLimit=200))
dots = (alt.Chart(imp).mark_point(filled=True, size=60)
        .encode(x=alt.X("mean_bp:Q", title=T["ax_rmse"]),
                y=enc_y,
                color=alt.Color("kind:N", title=None,
                                scale=alt.Scale(domain=KINDS, range=[BLUE, ORANGE]),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("what:N", title=T["tt_predictor"]),
                         alt.Tooltip("mean_bp_s:N", title=T["tt_rmse"]),
                         alt.Tooltip("p10_s:N", title=T["tt_p10"]),
                         alt.Tooltip("p90_s:N", title=T["tt_p90"])]))
whisk = (alt.Chart(imp).mark_rule(strokeWidth=2, opacity=0.65)
         .encode(x="p10:Q", x2="p90:Q", y=enc_y,
                 color=alt.Color("kind:N", legend=None,
                                 scale=alt.Scale(domain=KINDS, range=[BLUE, ORANGE]))))
zline = (alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED, strokeWidth=1)
         .encode(x="x:Q"))

comp = pd.read_csv(PROC / "model_comparison.csv", parse_dates=["date"])
# Naive first: yesterday's deviation carried forward, on the same dates.
first = comp[comp["model"] == comp["model"].iloc[0]].merge(oos[["date", "naive"]], on="date")
f24 = first[first["date"].dt.year >= 2024]
rival = [{"model": "naive", "r2_vs_ar1": r2(first["y"], first["naive"], first["ar1"]),
          "r2_vs_ar1_2024": r2(f24["y"], f24["naive"], f24["ar1"]),
          "rmse_bp": float(np.sqrt(np.mean((first["y"] - first["naive"]) ** 2)))}]
for name, g in comp.groupby("model", sort=False):
    g24 = g[g["date"].dt.year >= 2024]
    rival.append({"model": name, "r2_vs_ar1": r2(g["y"], g["pred"], g["ar1"]),
                  "r2_vs_ar1_2024": r2(g24["y"], g24["pred"], g24["ar1"]),
                  "rmse_bp": float(np.sqrt(np.mean((g["y"] - g["pred"]) ** 2)))})
rival = pd.DataFrame(rival)
rival_view = pd.DataFrame({T["c_model"]: rival["model"].map(
    lambda m: T["naive_label"] if m == "naive" else T["rivals"].get(m, m)),
                           T["c_r2_all"]: fmt_col(rival["r2_vs_ar1"], 3, sign=True),
                           T["c_r2_24"]: fmt_col(rival["r2_vs_ar1_2024"], 3, sign=True),
                           T["c_rmse"]: fmt_col(rival["rmse_bp"], 3)})
i1, i2 = st.columns([1.3, 1])
i1.altair_chart(style((whisk + dots + zline).properties(height=21 * len(imp) + 80)),
                width="stretch")
with i2:
    st.markdown(f"**{T['tb_rival']}**")
    show_table("dl_rival", rival_view, rival, signed=[T["c_r2_all"], T["c_r2_24"]])

st.divider()

# ------------------------------------------------------------ is it worth it
st.subheader(T["s4"], anchor="carry")
st.caption(T["cap_short_carry"])
always = carry["carry_bp"].mean()
peak = daily_bt["carry_bp"].max()
r1, r2c, r3 = st.columns([1, 1, 3])
r1.metric(T["tile_carry"], f"{n(always, 2)} {BP}", help=T["tile_carry_help"])
r2c.metric(T["tile_peak"], f"{n(peak)} {BP}", help=T["tile_peak_help"])
# None marks the rule's own column, shown as a dash.
NET = [4.829, 5.030, 5.114, 5.117, 5.132]
VS_ALWAYS = [None, 0.201, 0.284, 0.287, 0.303]
VS_KNOWN = [-0.201, None, 0.084, 0.087, 0.103]
RULES = ["always lend", "known only, no model", "AR(1)", "two-stage LASSO", "perfect foresight"]
dash = lambda v: "—" if v is None else n(v, 3, sign=True)  # noqa: E731
rules = pd.DataFrame({
    T["col_rule"]: [T[k] for k in ("rule_always", "rule_known", "rule_ar1",
                                   "rule_lasso", "rule_perfect")],
    T["col_net"]: [n(v, 3) for v in NET],
    T["col_vs_always"]: [dash(v) for v in VS_ALWAYS],
    T["col_vs_known"]: [dash(v) for v in VS_KNOWN],
})
with r3:
    show_table("dl_rules", rules, pd.DataFrame({"rule": RULES, "net_bp": NET,
                                                "vs_always_lend_bp": VS_ALWAYS,
                                                "vs_known_only_bp": VS_KNOWN}),
               signed=[T["col_vs_always"], T["col_vs_known"]])

st.divider()

# ----------------------------------------------------------------- method
st.subheader(T["method_title"], anchor="method")
with st.expander(T["about_title"]):
    st.markdown(T["intro"])
with st.expander(T["m_turns"]):
    st.markdown(T["turns_intro"].format(rrp=n(STATE.scarcity["rrp_bn_lag1"], 1),
                                        tga=n(STATE.scarcity["tga_bn_lag1"], 0),
                                        base=n(STATE.base_bp, 1, sign=True)))
    st.markdown(T["turns_table_note"])
    st.markdown(T["analogs_note"])
    st.markdown(T["calc_note"].format(**CALC_NOTE_ARGS) if CALC_NOTE_ARGS
                else T["calc_note"].split(". ", 1)[-1])
with st.expander(T["m_corridor"]):
    st.markdown(T["cap1"])
    st.markdown(T["cap_srf"])
with st.expander(T["m_calendar"]):
    st.markdown(T["cap2"])
with st.expander(T["m_model"]):
    cov = cov_raw.set_index("group")["coverage"]
    st.markdown(T["cap3"])
    st.markdown(T["cap3_dm"].format(**DM_ARGS))
    st.markdown(T["cap_fan"].format(all=n(100 * cov["all"], 1), me=n(100 * cov["month-end"], 1),
                                    qe=n(100 * cov["quarter-end"], 1),
                                    other=n(100 * cov["other days"], 1)))
with st.expander(T["m_importance"]):
    st.markdown(T["cap3b"])
with st.expander(T["m_carry"]):
    st.markdown(T["s4_intro"])
    st.markdown(T["conclusion"])
with st.expander(T["m_timing"]):
    st.markdown(T["timing_table"])
    st.markdown(T["timing_note"])
    st.markdown(T["sources"])

# ----------------------------------------------------------------- footer
st.divider()
st.markdown(T["footer_sources"])
st.markdown(f"**{T['dl_title']}**")
dl_cols = st.columns(4)
for i, (key, frame) in enumerate(TABLES.items()):
    dl_cols[i % 4].download_button(T.get(key, key), frame.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{key[3:]}.csv", mime="text/csv",
                                   key=f"download_{key}", width="stretch")
