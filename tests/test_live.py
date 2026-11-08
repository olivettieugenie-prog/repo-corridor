"""Checks that the live forecast reproduces the research pipeline exactly.

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

import forecast_next as fn  # noqa: E402
import market_calendar as mc  # noqa: E402
import model_lasso as m  # noqa: E402

DAILY = pd.read_csv(ROOT / "data" / "processed" / "daily.csv", parse_dates=["date"])


class TestCalendar(unittest.TestCase):
    def test_matches_every_sofr_date(self):
        s = pd.DatetimeIndex(DAILY["date"])
        cal = mc.business_days(s.min(), s.max())
        self.assertEqual(list(cal.difference(s)), [], "calendar open, no SOFR")
        self.assertEqual(list(s.difference(cal)), [], "SOFR published, calendar closed")

    def test_known_days(self):
        self.assertFalse(mc.is_business_day("2026-04-03"))   # Good Friday
        self.assertTrue(mc.is_business_day("2021-12-31"))    # Saturday New Year's
        self.assertTrue(mc.is_business_day("2023-11-10"))    # Saturday Veterans Day
        self.assertFalse(mc.is_business_day("2027-06-18"))   # Juneteenth observed
        self.assertEqual(mc.next_business_day("2026-11-10"), pd.Timestamp("2026-11-12"))

    def test_fomc_effective_day(self):
        self.assertTrue(mc.fomc_effective_day("2026-10-29"))
        self.assertFalse(mc.fomc_effective_day("2026-10-28"))


class TestReproducesResearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.when = m.build(DAILY)

    def test_feature_row_matches_build(self):
        d = DAILY.sort_values("date").reset_index(drop=True)
        for k in (700, 1200, 1700, len(self.X) - 1):
            target = self.when.iloc[k]
            asof_i = d.index[d["date"] == target][0] - 1
            hist = d.iloc[:asof_i + 1]
            rc = int(d.at[asof_i + 1, "rate_change"])
            got = fn.feature_row(hist, target, rc).iloc[0]
            pd.testing.assert_series_equal(got.astype(float), self.X.iloc[k].astype(float),
                                           check_names=False, obj=f"row for {target.date()}")

    def test_fit_matches_walk_forward(self):
        res, _, _ = m.walk_forward(self.X, self.y, self.when, window=m.WINDOW)
        start = m.TRAIN_MIN + m.STEP * 10
        tr = slice(start - m.WINDOW, start)
        te = slice(start, start + m.STEP)
        f, a = fn.fit_predict(self.X.iloc[tr], self.y.iloc[tr].to_numpy(float), self.X.iloc[te])
        out = slice(start - m.TRAIN_MIN, start - m.TRAIN_MIN + m.STEP)  # res starts at TRAIN_MIN
        np.testing.assert_allclose(f, res["two_stage"].iloc[out], atol=1e-10)
        np.testing.assert_allclose(a, res["ar1"].iloc[out], atol=1e-10)

    def test_level_mapping_is_exact(self):
        d = DAILY.dropna(subset=["dev_bp"])
        back = d["target_hi"] + (d["base_bp"] + d["dev_bp"]) / 100
        np.testing.assert_allclose(back, d["sofr"], atol=1e-9)
        k = 1500
        sofr, base = fn.implied_sofr(DAILY.iloc[:k], DAILY.at[k, "target_hi"], DAILY.at[k, "dev_bp"])
        self.assertAlmostEqual(base, DAILY.at[k, "base_bp"])
        self.assertAlmostEqual(sofr, DAILY.at[k, "sofr"])


class TestScoring(unittest.TestCase):
    def test_fills_realised_only_when_published(self):
        live = pd.DataFrame([
            {"target_date": "2026-10-23", "lower_bp": 0.0, "upper_bp": 5.0, "realised_bp": np.nan},
            {"target_date": "2099-01-02", "lower_bp": 0.0, "upper_bp": 5.0, "realised_bp": np.nan},
        ])
        out = fn.score(live, DAILY, "now")
        y = float(DAILY.set_index("date").at[pd.Timestamp("2026-10-23"), "dev_w_bp"])
        self.assertAlmostEqual(out.at[0, "realised_bp"], y, places=6)
        self.assertEqual(out.at[0, "hit"], int(0 <= y <= 5))
        self.assertTrue(pd.isna(out.at[1, "realised_bp"]))


if __name__ == "__main__":
    unittest.main()
