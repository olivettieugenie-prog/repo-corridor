"""Next turns and term-versus-roll panels, run through AppTest in EN/FR and
light/dark, plus the calculator's blockers.

The live log is switched off (unreachable URL, no local copy), so the panels
price from the committed daily.csv and the results do not depend on the day
the tests run.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import desk  # noqa: E402
import pandas as pd  # noqa: E402
from i18n import LANGS, STRINGS, date_short  # noqa: E402

TODAY = "2026-11-09"
KEYS = ("REPO_LIVE_URL", "REPO_LIVE_LOCAL", "REPO_TODAY")


class TurnsAppTest(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in KEYS}
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["REPO_LIVE_URL"] = "https://example.invalid/forecast.csv"
        os.environ["REPO_LIVE_LOCAL"] = str(Path(self.tmp.name) / "none.csv")
        os.environ["REPO_TODAY"] = TODAY

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

    @staticmethod
    def texts(at) -> str:
        parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
        parts += [w.value for w in at.warning] + [i.value for i in at.info]
        parts += [m.label for m in at.metric]
        return "\n".join(parts)

    @staticmethod
    def turns_table(at, T):
        for df in at.dataframe:
            v = df.value
            if T["tt_col_p95"] in getattr(v, "columns", []):
                return v
        raise AssertionError("no Next turns table")

    def test_panels_in_every_language_and_theme(self):
        for lang in LANGS:
            for theme in ("light", "dark"):
                with self.subTest(lang=lang, theme=theme):
                    T = STRINGS[lang]
                    at = self.run_app(lang, theme)
                    heads = [h.value for h in at.subheader]
                    self.assertEqual(heads[:3], [T["desk_title"], T["turns_title"], T["calc_title"]])
                    tab = self.turns_table(at, T)
                    self.assertEqual(list(tab[T["tt_col_date"]]),
                                     [date_short(pd.Timestamp("2026-11-30"), lang),
                                      date_short(pd.Timestamp("2026-12-31"), lang)])
                    self.assertEqual(list(tab[T["tt_col_type"]]),
                                     [T["turn_month-end"], T["turn_year-end"]])
                    me, ye = tab.iloc[0], tab.iloc[1]
                    for col in ("tt_col_p50", "tt_col_p90", "tt_col_p95"):
                        self.assertRegex(me[T[col]], r"^[+-]\d")      # signed number
                        self.assertEqual(ye[T[col]], "—")
                    self.assertEqual(ye[T["tt_col_cov"]], T["turns_ye_cell"])
                    self.assertEqual(ye[T["tt_col_n"]], "8")
                    txt = self.texts(at)
                    for key in ("calc_central", "calc_p90", "calc_be", "calc_be_p90"):
                        self.assertIn(T[key], txt)
                    self.assertIn(T["turns_table_note"], txt)
                    self.assertGreaterEqual(len(at.expander), 2)        # about, analogs
                    self.assertGreaterEqual(len(at.dataframe), 3)       # turns, analogs, rules
                    verdicts = [T[f"calc_{v}"].split("{")[0] for v in ("term", "insure", "roll")]
                    self.assertTrue(any(v in txt for v in verdicts))
                    self.assertIn(T["calc_note"].split("{")[0], txt)

    def test_analog_picker_switches_turn(self):
        at = self.run_app()
        at.selectbox(key="analog_turn").set_value(1).run()     # the year-end
        self.assertIn(STRINGS["en"]["turn_ye"].format(n=8), self.texts(at))

    def test_fomc_window_has_warning_and_no_verdict(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                T = STRINGS[lang]
                at = self.run_app(lang, window=(date(2026, 12, 1), date(2026, 12, 15)))
                self.assertIn(T["calc_fomc"].format(dates=desk.long_date("2026-12-09", lang)),
                              [w.value for w in at.warning])
                self.assertNotIn(T["calc_central"], [m.label for m in at.metric])

    def test_year_end_window_shows_history_and_prices_the_rest(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                T = STRINGS[lang]
                at = self.run_app(lang, window=(date(2026, 12, 28), date(2027, 1, 8)))
                txt = self.texts(at)
                self.assertIn(T["calc_ye_title"].format(date=desk.long_date("2026-12-31", lang)), txt)
                self.assertIn(T["calc_ye_stats"].split("{")[0], txt)
                self.assertIn(T["calc_ye_rest"], txt)
                self.assertIn(T["calc_central"], [m.label for m in at.metric])
                verdicts = [T[f"calc_{v}"].split("{")[0] for v in ("term", "insure", "roll")]
                self.assertFalse(any(v in txt for v in verdicts))
                self.assertEqual([w.value for w in at.warning], [])

    def test_start_before_first_forecast_day(self):
        at = self.run_app(window=(date(2026, 10, 1), date(2026, 10, 20)))
        self.assertTrue(any(w.value.startswith(STRINGS["en"]["calc_early"].split("{")[0])
                            for w in at.warning))

    def test_end_before_start(self):
        at = self.run_app(window=(date(2026, 11, 20), date(2026, 11, 19)))
        self.assertIn(STRINGS["en"]["calc_empty"], [w.value for w in at.warning])

    def test_no_turns_left(self):
        os.environ["REPO_TODAY"] = "2027-01-04"
        for lang in LANGS:
            with self.subTest(lang=lang):
                at = self.run_app(lang)
                self.assertIn(STRINGS[lang]["turn_none"], [i.value for i in at.info])


if __name__ == "__main__":
    unittest.main()
