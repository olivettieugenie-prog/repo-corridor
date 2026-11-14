"""The carry table is computed from carry_backtest.csv, and the Next turns
window follows today's date instead of stopping at a fixed day.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("REPO_DAILY_URL", "https://example.invalid/daily.csv")
os.environ.setdefault("REPO_DAILY_LOCAL", str(Path(__file__).resolve().parent / "no_daily.csv"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest_carry import run_rule  # noqa: E402
from i18n import LANGS, STRINGS, num  # noqa: E402

KEYS = ("REPO_LIVE_URL", "REPO_LIVE_LOCAL", "REPO_TODAY", "REPO_FORWARD_URL", "REPO_FORWARD_LOCAL")
CARRY = pd.read_csv(ROOT / "data" / "processed" / "carry_backtest.csv", parse_dates=["date"])


def backtest_nets(cost: float = 0.0) -> list[float]:
    a = CARRY["carry_bp"].to_numpy(float)
    preds = (np.full(len(a), 1e9), CARRY["carry_known_bp"], CARRY["carry_ar1"],
             CARRY["carry_ts"], a)
    return [run_rule(np.asarray(p, float), a, cost)["net_bp"] for p in preds]


class AppRun(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in KEYS}
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["REPO_LIVE_URL"] = "https://example.invalid/forecast.csv"
        os.environ["REPO_LIVE_LOCAL"] = str(Path(self.tmp.name) / "none.csv")
        os.environ["REPO_FORWARD_URL"] = "https://example.invalid/forward.csv"
        os.environ["REPO_FORWARD_LOCAL"] = str(ROOT / "tests" / "fixtures" / "forward.csv")
        os.environ["REPO_TODAY"] = "2026-11-09"

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.tmp.cleanup()

    def run_app(self, lang="en"):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
        at.run()
        if lang != "en":
            at.segmented_control(key="lang").set_value(lang).run()
        self.assertEqual(len(at.exception), 0, at.exception)
        return at

    @staticmethod
    def table_with(at, column: str) -> pd.DataFrame:
        for df in at.dataframe:
            v = df.value
            v = v.data if hasattr(v, "data") else v
            if column in getattr(v, "columns", []):
                return v
        raise AssertionError(f"no table with column {column!r}")


class CarryTableTest(AppRun):
    def test_backtest_figures(self):
        self.assertEqual([round(v, 3) for v in backtest_nets()],
                         [5.034, 5.201, 5.270, 5.273, 5.286])

    def test_app_table_equals_backtest_carry(self):
        nets = backtest_nets()
        for lang in LANGS:
            T = STRINGS[lang]
            with self.subTest(lang=lang):
                at = self.run_app(lang)
                tab = self.table_with(at, T["col_net"])
                self.assertEqual(list(tab[T["col_net"]]), [num(v, 3, lang) for v in nets])
                always = [num(v - nets[0], 3, lang, True) for v in nets[1:]]
                self.assertEqual(list(tab[T["col_vs_always"]])[1:], always)
                text = "\n".join([c.value for c in at.caption] + [m.value for m in at.markdown])
                self.assertIn(num(nets[4] - nets[0], 3, lang), text)       # the prize, 0.252
                self.assertNotIn(num(0.303, 3, lang), text)


class RangeValueTest(AppRun):
    def test_value_tables_follow_the_summary(self):
        tv = pd.read_csv(ROOT / "data" / "processed" / "turn_value_summary.csv")
        for lang in LANGS:
            T = STRINGS[lang]
            with self.subTest(lang=lang):
                at = self.run_app(lang)
                self.assertIn(T["s5"], [h.value for h in at.subheader])
                tables = [d.value for d in at.dataframe if T["tv_cush_usd"] in d.value.columns]
                self.assertEqual(len(tables), 2)
                for tab, kind in zip(tables, ("quarter-end", "month-end")):
                    g = tv[tv["type"] == kind].sort_values(["horizon", "method"])
                    self.assertEqual(list(tab[T["tv_hits"]]),
                                     [f"{h}/{k}" for h, k in zip(g["hits"], g["n"])])
                caps = " ".join(c.value for c in at.caption)
                self.assertIn(T["s5_caveat"], caps)
                self.assertIn(T["calc_benchmark"], caps)


class TurnsWindowTest(AppRun):
    def test_turns_after_the_old_end_date(self):
        os.environ["REPO_TODAY"] = "2027-01-04"
        at = self.run_app()
        tab = self.table_with(at, STRINGS["en"]["tt_col_p95"])
        self.assertGreater(len(tab), 0)
        self.assertTrue(all("2027" in d or "Jan" in d or "Feb" in d or "Mar" in d
                            for d in tab[STRINGS["en"]["tt_col_date"]]))


if __name__ == "__main__":
    unittest.main()
