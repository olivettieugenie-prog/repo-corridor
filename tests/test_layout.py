"""Dashboard layout: every linked section renders in both languages and both
themes, and every table on the page has a CSV download.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import dashboard as db  # noqa: E402
from i18n import LANGS, STRINGS  # noqa: E402

HEADING = {"forward": "fwd_title", "turns": "turns_title", "term-roll": "calc_title",
           "calendar": "s2", "corridor": "s1", "model": "s3", "carry": "s4",
           "method": "method_title"}
TABLE_KEYS = ["dl_turns", "dl_analogs", "dl_r2", "dl_dm", "dl_cov", "dl_turnscores",
              "dl_rival", "dl_rules"]
KEYS = ("REPO_LIVE_URL", "REPO_LIVE_LOCAL", "REPO_TODAY", "REPO_FORWARD_URL", "REPO_FORWARD_LOCAL")


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in KEYS}
        self.tmp = tempfile.TemporaryDirectory()
        os.environ.update(REPO_LIVE_URL="https://example.invalid/x.csv",
                          REPO_LIVE_LOCAL=str(Path(self.tmp.name) / "none.csv"),
                          REPO_TODAY="2026-11-09", REPO_FORWARD_URL="https://example.invalid/forward.csv", REPO_FORWARD_LOCAL=str(ROOT / "tests" / "fixtures" / "forward.csv"))

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.tmp.cleanup()

    def run_app(self, lang="en", theme="light", window=None):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
        at.context["theme"] = theme
        at.run()
        if lang != "en":
            at.segmented_control(key="lang").set_value(lang).run()
        if window:
            at.date_input(key="calc_start").set_value(window[0])
            at.date_input(key="calc_end").set_value(window[1]).run()
        self.assertEqual(len(at.exception), 0, at.exception)
        return at

    def test_every_section_renders(self):
        for lang in LANGS:
            for theme in ("light", "dark"):
                with self.subTest(lang=lang, theme=theme):
                    at = self.run_app(lang, theme)
                    anchors = {h.proto.anchor: h.value for h in at.subheader if h.proto.anchor}
                    self.assertEqual(set(anchors), set(db.SECTIONS))
                    for sec, key in HEADING.items():
                        self.assertEqual(anchors[sec], STRINGS[lang][key])
                    nav = " ".join(h.proto.body for h in at.html)
                    self.assertEqual(re.findall(r'href="#([\w-]+)"', nav), db.SECTIONS)
                    self.assertGreaterEqual(len(at.expander), 8)     # analogs + 7 in Method

    def test_every_table_has_a_csv_download(self):
        for lang in LANGS:
            for theme in ("light", "dark"):
                with self.subTest(lang=lang, theme=theme):
                    at = self.run_app(lang, theme)
                    keys = [b.key for b in at.download_button]
                    for k in TABLE_KEYS:
                        self.assertIn(f"download_{k}", keys)
                    self.assertEqual(len(at.dataframe), len(TABLE_KEYS))
                    labels = [b.label for b in at.download_button]
                    for k in TABLE_KEYS:
                        self.assertIn(STRINGS[lang][k], labels)

    def test_year_end_table_gets_its_download(self):
        at = self.run_app(window=(date(2026, 12, 28), date(2027, 1, 8)))
        self.assertIn("download_dl_ye", [b.key for b in at.download_button])
        self.assertEqual(len(at.dataframe), len(TABLE_KEYS) + 1)

    def test_forward_panel(self):
        for lang in LANGS:
            for theme in ("light", "dark"):
                with self.subTest(lang=lang, theme=theme):
                    T = STRINGS[lang]
                    at = self.run_app(lang, theme)
                    heads = [h.value for h in at.subheader]
                    self.assertEqual(heads[0], T["fwd_title"])
                    caps = " ".join(c.value for c in at.caption)
                    sc = pd.read_csv(ROOT / "data" / "processed" / "forward_backtest.csv")
                    turn = sc[sc["days"] == "turn days"]["coverage80"]
                    self.assertIn(T["fwd_caption"].split("{")[0], caps)
                    self.assertIn(f"{round(100 * turn.min())}–{round(100 * turn.max())}", caps)
                    self.assertIn("download_dl_forward", [b.key for b in at.download_button])
                    dl = [b for b in at.download_button if b.key == "download_dl_forward"]
                    self.assertEqual(len(dl), 1)

    def test_forward_panel_without_a_path(self):
        os.environ["REPO_FORWARD_LOCAL"] = str(Path(self.tmp.name) / "none.csv")
        at = self.run_app()
        self.assertIn(STRINGS["en"]["fwd_none"], [i.value for i in at.info])

    def test_tables_only_through_show_table(self):
        """show_table registers the CSV; a direct st.dataframe would skip it."""
        src = (ROOT / "app.py").read_text()
        self.assertEqual(src.count("st.dataframe("), 1)
        self.assertIn("def show_table(", src)

    def test_one_short_caption_per_section(self):
        at = self.run_app()
        long_captions = [c.value for c in at.caption if len(c.value) > 260]
        self.assertEqual(long_captions, [])


if __name__ == "__main__":
    unittest.main()
