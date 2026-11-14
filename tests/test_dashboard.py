"""Dashboard header pieces: section links and the market strip.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# App tests read the daily table from main's snapshot, not from whatever the
# data branch serves today: an unreachable URL and no local copy.
os.environ.setdefault("REPO_DAILY_URL", "https://example.invalid/daily.csv")
os.environ.setdefault("REPO_DAILY_LOCAL", str(Path(__file__).resolve().parent / "no_daily.csv"))

import dashboard as db  # noqa: E402
from i18n import LANGS, STRINGS  # noqa: E402

DAILY = pd.read_csv(ROOT / "data" / "processed" / "daily.csv", parse_dates=["date"])
SRF_RATES = pd.read_csv(ROOT / "data" / "srf_rates.csv", parse_dates=["effective_date"])
SRF_TAKE = pd.read_csv(ROOT / "data" / "processed" / "srf_takeup.csv", parse_dates=["date"])


class TestStrip(unittest.TestCase):
    def tiles(self):
        return db.strip_tiles(DAILY, None, SRF_RATES, SRF_TAKE)

    def test_eight_tiles_in_order(self):
        self.assertEqual([t["key"] for t in self.tiles()],
                         ["sofr", "effr", "range", "award", "srf_rate", "rrp", "tga", "srf_take"])

    def test_sofr_change_is_in_bp(self):
        last, prev = DAILY.iloc[-1], DAILY.iloc[-2]
        self.assertAlmostEqual(self.tiles()[0]["change"], (last["sofr"] - prev["sofr"]) * 100)

    def test_every_change_carries_its_sign(self):
        for lang in LANGS:
            for t in self.tiles():
                f = db.format_tile(t, lang)
                self.assertRegex(f["change"], r"^(\+|-|±0)")
                expect = {"+": "up", "-": "down", "±": "flat"}[f["change"][0]]
                self.assertEqual(f["dir"], expect)

    def test_change_that_rounds_to_zero_is_flat(self):
        f = db.format_tile({"key": "rrp", "kind": "bn", "value": 0.1, "change": 0.02, "dec": 1,
                            "date": pd.Timestamp("2026-11-06")}, "en")
        self.assertEqual((f["change"], f["dir"]), ("±0\u00a0bn", "flat"))

    def test_direction(self):
        self.assertEqual([db.direction(x) for x in (2.0, -0.5, 0.0)], ["up", "down", "flat"])

    def test_french_formatting(self):
        f = db.format_tile({"key": "sofr", "kind": "rate", "value": 3.93, "change": 1.0,
                            "date": pd.Timestamp("2026-10-26")}, "fr")
        self.assertEqual(f["value"], "3,93 %")
        self.assertEqual(f["change"], "+1 pb")
        self.assertEqual(f["date"], "26 oct.")

    def test_live_balances_preferred(self):
        live = pd.Series({"rrp_bn": 0.1, "d_rrp_bn": 0.02, "rrp_date": "2026-11-06",
                          "tga_bn": 923.0, "d_tga_bn": -13.0, "tga_date": "2026-11-05"})
        t = {x["key"]: x for x in db.strip_tiles(DAILY, live, SRF_RATES, SRF_TAKE)}
        self.assertEqual(t["tga"]["value"], 923.0)
        self.assertEqual(t["tga"]["date"], pd.Timestamp("2026-11-05"))

    def test_html_has_eight_tiles_with_labels(self):
        for lang in LANGS:
            h = db.strip_html(self.tiles(), lang)
            self.assertEqual(h.count('class="strip-tile"'), 8)
            for key in ("sofr", "range", "srf_take"):
                self.assertIn(STRINGS[lang][f"strip_{key}"], h)


class TestNavbar(unittest.TestCase):
    def test_links_to_every_section(self):
        for lang in LANGS:
            h = db.navbar(lang)
            self.assertEqual(re.findall(r'href="#([\w-]+)"', h), db.SECTIONS)

    def test_stylesheet_sets_palette_variables(self):
        sys.path.insert(0, str(ROOT / "src"))
        from design import DESIGNS
        c = DESIGNS["tempered"]["dark"]
        css = db.stylesheet(c)
        self.assertIn(f"--up:{c['blue']}", css)
        self.assertIn(f"--down:{c['red']}", css)
        self.assertTrue(css.startswith("<style>") and css.endswith("</style>"))


class TestAppHeader(unittest.TestCase):
    def test_header_strip_and_anchors_render(self):
        import os
        import tempfile
        from streamlit.testing.v1 import AppTest
        keys = ("REPO_LIVE_URL", "REPO_LIVE_LOCAL", "REPO_TODAY", "REPO_FORWARD_URL",
                "REPO_FORWARD_LOCAL")
        saved = {k: os.environ.get(k) for k in keys}
        with tempfile.TemporaryDirectory() as tmp:
            os.environ.update(REPO_FORWARD_URL="https://example.invalid/forward.csv", REPO_FORWARD_LOCAL=str(ROOT / "tests" / "fixtures" / "forward.csv"), REPO_LIVE_URL="https://example.invalid/x.csv",
                              REPO_LIVE_LOCAL=str(Path(tmp) / "none.csv"), REPO_TODAY="2026-11-09")
            try:
                for lang in LANGS:
                    for theme in ("light", "dark"):
                        with self.subTest(lang=lang, theme=theme):
                            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
                            at.context["theme"] = theme
                            at.run()
                            if lang != "en":
                                at.segmented_control(key="lang").set_value(lang).run()
                            self.assertEqual(len(at.exception), 0, at.exception)
                            htmls = " ".join(h.proto.body for h in at.html)
                            self.assertEqual(htmls.count('class="strip-tile"'), 8)
                            self.assertEqual(re.findall(r'href="#([\w-]+)"', htmls), db.SECTIONS)
                            self.assertEqual([h.proto.anchor for h in at.subheader if h.proto.anchor],
                                             db.SECTIONS[:3] + ["calendar", "corridor", "model",
                                                                "carry", "value", "method"])
                            caps = " ".join(c.value for c in at.caption)
                            self.assertIn(STRINGS[lang]["hdr_delay"], caps)
            finally:
                for k, v in saved.items():
                    os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


if __name__ == "__main__":
    unittest.main()
