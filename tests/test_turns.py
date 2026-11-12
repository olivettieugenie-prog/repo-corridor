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


class TestHorizonCalibration(unittest.TestCase):
    def test_horizon_state_is_known_on_the_forecast_date(self):
        t = tq.turn_days(DAILY)
        d = DAILY.sort_values("date").reset_index(drop=True)
        for h in tq.HORIZONS:
            pool = tq.horizon_turns(t, DAILY, h)
            r = pool.iloc[30]
            src = d.iloc[int(r["row"]) - h + 1]
            np.testing.assert_allclose(r[tq.STATE].astype(float), src[tq.STATE].astype(float))
            # those features use data up to the forecast date only
            self.assertLess(pd.Timestamp(src["rrp_lag1_date"]), pd.Timestamp(src["date"]))
            self.assertEqual(r["made_date"], d.at[int(r["row"]) - h, "date"])

    def test_calibration_ignores_turns_not_yet_printed(self):
        t = tq.turn_days(DAILY)
        fc = tq.walk_forward_horizon(tq.horizon_turns(t, DAILY, 10), 10)
        a = tq.calibrate(fc)
        cut = fc["date"].iloc[25]
        bumped = fc.copy()
        bumped.loc[bumped["date"] > cut, "y"] += 300
        b = tq.calibrate(bumped)
        keep = a["date"] <= cut
        np.testing.assert_allclose(a.loc[keep, "q90_cal"], b.loc[keep, "q90_cal"], equal_nan=True)
        self.assertTrue(a["q90_cal"].iloc[:tq.MIN_CAL].isna().all())


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


class TestNaiveBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tq.turn_days(DAILY)
        cls.hz = tq.walk_forward_horizon(tq.horizon_turns(cls.t, DAILY, 5), 5)

    def test_naive_is_the_spread_on_the_forecast_date(self):
        nz = tq.add_naive(self.hz, DAILY, self.t)
        spread = DAILY.set_index("date")["spread_hi_bp"]
        np.testing.assert_allclose(nz["naive"], nz["made_date"].map(spread))

    def test_previous_turn_is_known_on_the_forecast_date(self):
        nz = tq.add_naive(self.hz, DAILY, self.t)
        turns = self.t.set_index("date")["y"]
        for _, r in nz.dropna(subset=["prev_turn"]).sample(10, random_state=0).iterrows():
            known = turns[turns.index <= r["made_date"]]
            self.assertEqual(r["prev_turn"], known.iloc[-1])

    def test_naive_range_ignores_turns_not_yet_printed(self):
        nz = tq.add_naive(self.hz, DAILY, self.t)
        a = tq.naive_ranges(nz)
        cut = nz["date"].iloc[30]
        bumped = nz.copy()
        bumped.loc[bumped["date"] > cut, "y"] += 300
        b = tq.naive_ranges(bumped)
        keep = a["date"] <= cut
        np.testing.assert_allclose(a.loc[keep, "nq90"], b.loc[keep, "nq90"], equal_nan=True)


class TestNaiveDM(unittest.TestCase):
    def test_groups_and_previous_turn(self):
        import dm_test as dmt
        oos = pd.read_csv(ROOT / "data" / "processed" / "oos_predictions.csv", parse_dates=["date"])
        nv = dmt.naive_table(oos, DAILY)
        n = nv[(nv["forecaster"] == "model") & (nv["benchmark"] == "naive")].set_index("group")["n"]
        self.assertEqual(n["all"], n["turn days"] + n["other days"])
        self.assertEqual(set(nv[nv["benchmark"] == "previous turn"]["group"]), {"turn days"})
        pt = dmt.previous_turn(oos, DAILY)
        turns = DAILY[DAILY["month_end"] == 1].set_index("date")["dev_w_bp"]
        d = oos["date"].iloc[400]
        self.assertEqual(pt.iloc[400], turns[turns.index < d].iloc[-1])
