"""Economic test of the turn range (src/turn_value.py).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import turn_value as tv  # noqa: E402

PROC = ROOT / "data" / "processed"


class TurnValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        daily = pd.read_csv(PROC / "daily.csv", parse_dates=["date"])
        hz = pd.read_csv(PROC / "turn_quantiles_horizon.csv", parse_dates=["date", "made_date"])
        cls.v = tv.value(daily, hz)
        cls.s = tv.summary(cls.v)

    def test_usd_conversion_by_hand(self):
        # 22 bp on $1bn for one night, actual/360: 0.0022 * 1e9 / 360 = 6,111.11
        self.assertAlmostEqual(float(tv.usd_per_bn(22, 1)), 6111.11, places=2)
        self.assertAlmostEqual(float(tv.usd_per_bn(22, 3)), 3 * 6111.11, places=1)

    def test_friday_turn_carries_the_weekend(self):
        self.assertEqual(tv.nights_after("2025-10-31"), 3)      # Friday
        self.assertEqual(tv.nights_after("2025-09-30"), 1)      # Tuesday

    def test_hit_counts_match_turn_naive_ranges(self):
        tv.check_hits(self.s, pd.read_csv(PROC / "turn_naive_ranges.csv"))
        qe = self.s[self.s["type"] == "quarter-end"].set_index(["method", "horizon"])["hits"]
        self.assertEqual([qe[("calibrated model", h)] for h in (1, 5, 10, 15)], [15, 16, 16, 15])
        self.assertEqual([qe[("naive range", h)] for h in (1, 5, 10, 15)], [8, 8, 9, 10])
        n = self.s.groupby("type")["n"].first()
        self.assertEqual((n["quarter-end"], n["month-end"]), (16, 40))

    def test_rows_are_consistent(self):
        v = self.v
        self.assertTrue(((v["overshoot_bp"] == 0) | (v["cushion_bp"] == 0)).all())
        self.assertTrue((v.loc[v["hit"] == 1, "overshoot_bp"] == 0).all())
        self.assertTrue((v["overshoot_usd_per_bn"]
                         == tv.usd_per_bn(v["overshoot_bp"], v["nights"])).all())

    def test_committed_summary_is_current(self):
        saved = pd.read_csv(PROC / "turn_value_summary.csv")
        pd.testing.assert_frame_equal(saved[["type", "horizon", "method", "n", "hits"]],
                                      self.s[["type", "horizon", "method", "n", "hits"]],
                                      check_dtype=False)


if __name__ == "__main__":
    unittest.main()
