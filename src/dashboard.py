"""Header, section links and market strip for the dashboard layout.

Pure functions returning data or HTML strings, so tests can check them
without a browser. Colours are CSS variables set by app.py; every change is
also signed, so colour never carries the meaning on its own.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from i18n import STRINGS, num

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ["forward", "turns", "term-roll", "corridor", "calendar", "model", "carry", "value", "method"]


def css_vars(c: dict) -> str:
    """CSS variables for style.css from the active design's palette.
    Up is the palette's blue and down its red: both read at 4.5:1 or more as
    text on the page and tile backgrounds in light and dark."""
    return (f":root{{--ink:{c['ink']};--muted:{c['muted']};--grid:{c['grid']};"
            f"--surface:{c['surface']};--bg:{c['bg']};--up:{c['blue']};--down:{c['red']};}}")


def stylesheet(c: dict) -> str:
    css = (ROOT / "src" / "style.css").read_text()
    return f"<style>{css_vars(c)}\n{css}</style>"


def navbar(lang: str) -> str:
    t = STRINGS[lang]
    links = "".join(f'<a href="#{s}">{html.escape(t["nav_" + s.replace("-", "_")])}</a>'
                    for s in SECTIONS)
    return f'<nav class="navbar" aria-label="{html.escape(t["nav_label"])}">{links}</nav>'


# ---------------------------------------------------------------- strip
def _short(ts, lang: str) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {STRINGS[lang]['months'][ts.month - 1]}"


def strip_tiles(daily: pd.DataFrame, live_row: pd.Series | None, srf_rates: pd.DataFrame,
                srf_takeup: pd.DataFrame) -> list[dict]:
    """Eight tiles: value, change since the previous observation, and the
    date of the observation. Rates in percent with changes in bp; balances
    in USD bn with changes in bn."""
    d = daily.sort_values("date")
    last, prev = d.iloc[-1], d.iloc[-2]
    tiles = []

    def rate(key, col):
        tiles.append({"key": key, "kind": "rate", "value": float(last[col]),
                      "change": (float(last[col]) - float(prev[col])) * 100, "date": last["date"]})

    rate("sofr", "sofr")
    rate("effr", "effr")
    tiles.append({"key": "range", "kind": "range", "value": (float(last["target_lo"]),
                                                             float(last["target_hi"])),
                  "change": (float(last["target_hi"]) - float(prev["target_hi"])) * 100,
                  "date": last["date"]})
    rate("award", "award_pct")

    r = srf_rates.sort_values("effective_date").set_index("effective_date")["srf_rate_pct"]
    now, before = r.asof(last["date"]), r.asof(prev["date"])
    tiles.append({"key": "srf_rate", "kind": "rate", "value": float(now),
                  "change": (float(now) - float(before)) * 100, "date": last["date"]})

    if live_row is not None:
        tiles.append({"key": "rrp", "kind": "bn", "value": float(live_row["rrp_bn"]),
                      "change": float(live_row["d_rrp_bn"]), "date": pd.Timestamp(live_row["rrp_date"]),
                      "dec": 1})
        tiles.append({"key": "tga", "kind": "bn", "value": float(live_row["tga_bn"]),
                      "change": float(live_row["d_tga_bn"]), "date": pd.Timestamp(live_row["tga_date"]),
                      "dec": 0})
    else:
        tiles.append({"key": "rrp", "kind": "bn", "value": float(last["rrp_bn_t"]),
                      "change": float(last["rrp_bn_t"] - prev["rrp_bn_t"]), "date": last["date"],
                      "dec": 1})
        tiles.append({"key": "tga", "kind": "bn", "value": float(last["tga_bn_lag1"]),
                      "change": float(last["d_tga_bn_lag1"]),
                      "date": pd.Timestamp(last["tga_lag1_date"]), "dec": 0})

    s = srf_takeup.sort_values("date")
    tiles.append({"key": "srf_take", "kind": "bn", "value": float(s["srf_bn"].iloc[-1]),
                  "change": float(s["srf_bn"].iloc[-1] - s["srf_bn"].iloc[-2]),
                  "date": s["date"].iloc[-1], "dec": 1})
    return tiles


def direction(change: float, eps: float = 1e-9) -> str:
    return "up" if change > eps else ("down" if change < -eps else "flat")


def format_tile(t: dict, lang: str) -> dict:
    """Localised strings for one tile: label, value, signed change, date."""
    s = STRINGS[lang]
    if t["kind"] == "range":
        lo, hi = t["value"]
        value = f"{num(lo, 2, lang)}–{num(hi, 2, lang)} %"
    elif t["kind"] == "rate":
        value = f"{num(t['value'], 2, lang)} %"
    else:
        value = s["desk_level"].format(v=num(t["value"], t["dec"], lang))
    if t["kind"] == "bn":
        chg = num(t["change"], t["dec"], lang, sign=True)
        chg = f"{chg} {s['strip_bn']}"
    else:
        chg = f"{num(t['change'], 0, lang, sign=True)} {s['bp']}"
    # Direction follows the change as displayed, so a change that rounds to
    # zero reads "±0" and is not coloured.
    dirn = direction(round(t["change"], 0 if t["kind"] != "bn" else t["dec"]))
    if dirn == "flat":
        chg = f"±0 {s['bp'] if t['kind'] != 'bn' else s['strip_bn']}"
    return {"label": s[f"strip_{t['key']}"], "value": value, "change": chg,
            "dir": dirn, "date": _short(t["date"], lang)}


def strip_html(tiles: list[dict], lang: str) -> str:
    s = STRINGS[lang]
    cells = []
    for t in tiles:
        f = format_tile(t, lang)
        title = s["strip_tip"].format(label=f["label"], date=f["date"])
        cells.append(
            f'<div class="strip-tile" title="{html.escape(title)}">'
            f'<div class="t-label">{html.escape(f["label"])}</div>'
            f'<div class="t-value">{html.escape(f["value"])}</div>'
            f'<div class="t-foot"><span class="t-chg {f["dir"]}">{html.escape(f["change"])}</span>'
            f'<span class="t-date">{html.escape(f["date"])}</span></div></div>')
    return f'<div class="strip" role="list">{"".join(cells)}</div>'
