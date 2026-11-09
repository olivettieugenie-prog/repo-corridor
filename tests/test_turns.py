"""Turn quantiles and analogs: no look-ahead, and the scoring arithmetic.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import analogs as an  # noqa: E402
import turn_quantiles as tq  # noqa: E402

DAILY = pd.read_csv(ROOT / "data" / "processed" / "daily.csv", parse_dates=["date"])


class TestTurnQuantiles(unittest.TestCase):
    def test_pinball(self):
        y, q = np.array([1.0, 3.0]), np.array([2.0, 2.0])
        self.assertAlmostEqual(tq.pinball(y, q, 0.9), (0.1 * 1 + 0.9 * 1) / 2)

    def test_fit_ignores_later_turns(self):
        t = tq.turn_days(DAILY)
        pool = t[t["type"] != "year-end"].dropna(subset=tq.STATE)
        r = pool.iloc[40]
        earlier = pool[pool["date"] < r["date"]]
        q1 = tq.fit_quantiles(earlier, r.to_frame().T)
        bumped = pool.copy()
        bumped.loc[bumped["date"] >= r["date"], "y"] += 500
        q2 = tq.fit_quantiles(bumped[bumped["date"] < r["date"]], r.to_frame().T)
        np.testing.assert_allclose(q1, q2)
        self.assertTrue(q1[0] <= q1[1] <= q1[2])

    def test_saved_forecasts_use_only_earlier_turns(self):
        fc = pd.read_csv(ROOT / "data" / "processed" / "turn_quantiles.csv", parse_dates=["date"])
        t = tq.turn_days(DAILY)
        n_before = {d: int(((t["date"] < d) & (t["type"] != "year-end")).sum()) for d in t["date"]}
        m = fc[fc["method"] == "model, all turns"]
        self.assertTrue((m["n_train"] <= m["date"].map(n_before)).all())


class TestAnalogs(unittest.TestCase):
    def test_only_earlier_turns_of_same_type(self):
        as_of = pd.Timestamp("2025-09-30")
        state = DAILY.set_index("date").loc[as_of, an.STATE].to_dict()
        for kind in an.TYPES:
            a = an.analogs(DAILY, state, kind, as_of)
            avail = sum(1 for _, r in DAILY[(DAILY["date"] < as_of) & (DAILY["month_end"] == 1)]
                        .dropna(subset=an.STATE).iterrows() if an.turn_type(r) == kind)
            self.assertEqual(len(a), min(an.K, avail))   # 7 year-ends before Sep 2025
            self.assertTrue((a["date"] < as_of).all())
            self.assertTrue((a["type"] == kind).all())
            self.assertTrue(a["distance"].is_monotonic_increasing)
            self.assertEqual(bool(a["history_only"].all()), kind == "year-end")

    def test_standardisation_ignores_the_future(self):
        as_of = pd.Timestamp("2023-06-30")
        state = DAILY.set_index("date").loc[as_of, an.STATE].to_dict()
        a1 = an.analogs(DAILY, state, "month-end", as_of)
        later = DAILY.copy()
        later.loc[later["date"] >= as_of, an.STATE] *= 10
        a2 = an.analogs(later, state, "month-end", as_of)
        pd.testing.assert_frame_equal(a1, a2)


if __name__ == "__main__":
    unittest.main()
