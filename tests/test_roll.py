"""Term versus roll: day count, averaging, turn handling and blockers.

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

import roll_cost as rc  # noqa: E402

DAILY = pd.read_csv(ROOT / "data" / "processed" / "daily.csv", parse_dates=["date"])
COEFS = pd.read_csv(ROOT / "data" / "processed" / "turn_quantile_coefs.csv")


def state(**kw) -> rc.State:
    base = dict(anchor_date=pd.Timestamp("2026-11-09"), anchor_dev=0.0, base_bp=-11.0,
                target_hi=4.0, ar_c=0.0, ar_phi=0.5, err_q90=2.0,
                scarcity={"rrp_bn_lag1": 1.0, "tga_bn_lag1": 900.0})
    base.update(kw)
    return rc.State(**base)


class TestDayCount(unittest.TestCase):
    def test_weekend_and_holiday(self):
        # Fri 6 Nov to Mon 16 Nov 2026; Wed 11 Nov is Veterans Day.
        d = rc.roll_days("2026-11-06", "2026-11-16")
        self.assertEqual([x.strftime("%a %d") for x in d["date"]],
                         ["Fri 06", "Mon 09", "Tue 10", "Thu 12", "Fri 13"])
        self.assertEqual(d["days"].tolist(), [3, 1, 2, 1, 3])
        self.assertEqual(d["days"].sum(), 10)

    def test_days_sum_to_window_when_start_is_a_settlement_day(self):
        for s, e in (("2026-11-02", "2026-12-15"), ("2026-12-28", "2027-01-08"),
                     ("2027-03-24", "2027-04-02")):
            self.assertEqual(rc.roll_days(s, e)["days"].sum(),
                             (pd.Timestamp(e) - pd.Timestamp(s)).days)

    def test_last_placement_stops_at_end(self):
        d = rc.roll_days("2026-11-13", "2026-11-15")   # Friday, end on Sunday
        self.assertEqual(d["days"].tolist(), [2])

    def test_empty_window(self):
        self.assertTrue(rc.roll_days("2026-11-10", "2026-11-10").empty)


class TestAveraging(unittest.TestCase):
    def test_weighted_by_days(self):
        self.assertAlmostEqual(rc.weighted_rate([4.0, 5.0], [3, 1]), (12 + 5) / 4)

    def test_flat_rates_average_to_themselves(self):
        res = rc.price("2026-11-16", "2026-11-25", 3.80,
                       state(anchor_dev=0.0, ar_c=0.0, err_q90=0.0), COEFS)
        self.assertAlmostEqual(res["central_pct"], 4.0 - 0.11)
        self.assertAlmostEqual(res["breakeven_bp"], (3.89 - 3.80) * 100)

    def test_ar_decay(self):
        r = rc.day_rates(rc.roll_days("2026-11-09", "2026-11-12"),
                         state(anchor_dev=8.0, ar_phi=0.5, ar_c=0.0, base_bp=0.0), COEFS)
        np.testing.assert_allclose((r["central_pct"] - 4.0) * 100, [8.0, 4.0])

    def test_turn_day_uses_quantiles(self):
        st = state()
        r = rc.day_rates(rc.roll_days("2026-11-27", "2026-12-02"), st, COEFS)
        q = rc.turn_quantiles(COEFS, "month-end", st.scarcity, st.base_bp)
        row = r.set_index("date").loc[pd.Timestamp("2026-11-30")]
        self.assertEqual(row["kind"], "month-end")
        self.assertAlmostEqual(row["central_pct"], 4.0 + q["q50"] / 100)
        self.assertAlmostEqual(row["p90_pct"], 4.0 + q["q90"] / 100)


class TestBlockers(unittest.TestCase):
    def test_fomc_window_not_priced(self):
        res = rc.price("2026-12-01", "2026-12-15", 3.9, state(), COEFS)
        self.assertEqual(res["fomc"], [pd.Timestamp("2026-12-09")])
        self.assertNotIn("central_pct", res)
        self.assertIsNone(rc.verdict(res))

    def test_year_end_not_priced(self):
        res = rc.price("2026-12-28", "2027-01-08", 3.9, state(anchor_date=pd.Timestamp("2026-12-28")), COEFS)
        self.assertEqual(res["year_end"], [pd.Timestamp("2026-12-31")])
        self.assertIsNone(rc.verdict(res))

    def test_turn_kinds(self):
        self.assertEqual(rc.turn_kind("2026-11-30"), "month-end")
        self.assertEqual(rc.turn_kind("2027-03-31"), "quarter-end")
        self.assertEqual(rc.turn_kind("2026-12-31"), "year-end")
        self.assertIsNone(rc.turn_kind("2026-11-27"))


class TestAr1(unittest.TestCase):
    def test_matches_model_first_stage(self):
        import model_lasso as m
        from sklearn.linear_model import LinearRegression
        X, y, _ = m.build(DAILY)
        b1 = LinearRegression().fit(X[["dev_w"]].iloc[-500:], y.iloc[-500:])
        c, phi = rc.ar1_coefs(DAILY)
        self.assertAlmostEqual(phi, float(b1.coef_[0]), places=2)
        self.assertAlmostEqual(c, float(b1.intercept_), places=1)


if __name__ == "__main__":
    unittest.main()
