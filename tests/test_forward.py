"""Forward-path backtest: h = 1 is the model, intervals see only past errors.

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

import forward_backtest as fb  # noqa: E402

PROC = ROOT / "data" / "processed"
PATHS = pd.read_csv(PROC / "forward_paths.csv", parse_dates=["asof", "target"])
SCORES = pd.read_csv(PROC / "forward_backtest.csv")


class TestForward(unittest.TestCase):
    def test_h1_is_the_walk_forward_forecast(self):
        oos = pd.read_csv(PROC / "oos_predictions.csv", parse_dates=["date"])
        h1 = PATHS[PATHS["h"] == 1].merge(oos, left_on="target", right_on="date")
        self.assertEqual(len(h1), len(oos))
        np.testing.assert_allclose(h1["f"], h1["two_stage"], atol=1e-8)
        np.testing.assert_allclose(h1["y_x"], h1["y_y"])

    def test_targets_are_h_settlement_days_ahead(self):
        dates = pd.read_csv(PROC / "daily.csv", parse_dates=["date"])["date"].sort_values()
        pos = {d: k for k, d in enumerate(dates)}
        gap = PATHS["target"].map(pos) - PATHS["asof"].map(pos)
        self.assertTrue((gap == PATHS["h"]).all())

    def test_intervals_use_only_errors_known_on_the_forecast_date(self):
        g = PATHS[PATHS["h"] == 10].head(700).drop(columns=["lo80", "hi80"])
        a = fb.add_intervals(g)
        cut = a["asof"].iloc[600]
        bumped = g.copy()
        bumped.loc[bumped["target"] > cut, "y"] += 500
        b = fb.add_intervals(bumped)
        keep = a["asof"] <= cut
        np.testing.assert_allclose(a.loc[keep, "lo80"], b.loc[keep, "lo80"], equal_nan=True)

    def test_scores_cover_every_horizon_and_split(self):
        self.assertEqual(sorted(SCORES["h"].unique()), list(fb.HORIZONS))
        self.assertEqual(set(SCORES["days"]), {"all", "turn days", "other days"})
        for h, g in SCORES.groupby("h"):
            n = g.set_index("days")["n"]
            self.assertEqual(n["all"], n["turn days"] + n["other days"])

    def test_tested_horizon_rule(self):
        sc = pd.DataFrame({"h": list(fb.HORIZONS), "days": "other days",
                           "coverage80": [0.8, 0.75, 0.72, 0.65, 0.8, 0.8]})
        self.assertEqual(fb.tested_horizon(sc), 5)    # stops at the first miss
        self.assertEqual(fb.tested_horizon(SCORES), 15)


if __name__ == "__main__":
    unittest.main()


class TestLiveForwardPath(unittest.TestCase):
    """forecast_next.forward_path on the committed data, using the saved
    backtest paths as the error history (horizons 1, 2, 5, 10, 15, 20)."""

    @classmethod
    def setUpClass(cls):
        import forecast_next as fn
        import model_lasso as m
        cls.fn, cls.m = fn, m
        cls.d = pd.read_csv(PROC / "daily.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        cls.X, cls.y, _ = m.build(cls.d)
        cls.f = fn.forward_path(cls.d, cls.X, cls.y, history=PATHS)

    def test_day_one_is_the_model_call(self):
        target = self.f["date"].iloc[0]
        change = self.fn.range_path(self.d, [pd.Timestamp(target)])[0]["rate_change"]
        x = self.fn.feature_row(self.d, pd.Timestamp(target), change)
        call, _ = self.fn.fit_predict(self.X.iloc[-500:], self.y.iloc[-500:].to_numpy(float), x)
        self.assertAlmostEqual(self.f.at[0, "forecast_bp"], float(call[0]), places=8)

    def test_twenty_settlement_days_and_tested_flag(self):
        self.assertEqual(list(self.f["h"]), list(range(1, 21)))
        tested = fb.tested_horizon(SCORES)
        self.assertTrue((self.f["tested"] == (self.f["h"] <= tested).astype(int)).all())
        self.assertTrue(self.f["date"].is_monotonic_increasing)

    def test_ranges_where_the_history_has_the_horizon(self):
        have = self.f[self.f["h"].isin(fb.HORIZONS)]
        self.assertTrue((have["lo80_bp"] < have["forecast_bp"]).all())
        self.assertTrue((have["hi80_bp"] > have["forecast_bp"]).all())


class TestCalculatorUsesForwardPath(unittest.TestCase):
    def test_ordinary_day_inside_the_path(self):
        import roll_cost as rc
        fwd = pd.read_csv(ROOT / "tests" / "fixtures" / "forward.csv", parse_dates=["date"])
        st = rc.State(anchor_date=pd.Timestamp(fwd["date"].iloc[0]), anchor_dev=0.0, base_bp=-11.0,
                      target_hi=4.0, ar_c=0.0, ar_phi=0.5, err_q90=2.0,
                      scarcity={"rrp_bn_lag1": 1.0, "tga_bn_lag1": 900.0})
        coefs = pd.read_csv(PROC / "turn_quantile_coefs_horizon.csv")
        day = fwd[fwd["kind"].isna()].iloc[2]
        r = rc.day_rates(rc.roll_days(day["date"], day["date"] + pd.Timedelta(days=1)), st, coefs, fwd)
        self.assertAlmostEqual(r["central_pct"].iloc[0],
                               day["target_hi"] + (day["base_bp"] + day["forecast_bp"]) / 100)
        self.assertAlmostEqual(r["p90_pct"].iloc[0],
                               day["target_hi"] + (day["base_bp"] + day["p90_bp"]) / 100)
        beyond = fwd["date"].max() + pd.Timedelta(days=7)
        r2 = rc.day_rates(rc.roll_days(beyond, beyond + pd.Timedelta(days=1)), st, coefs, fwd)
        self.assertNotIn(beyond, set(fwd["date"]))
        self.assertTrue(np.isfinite(r2["central_pct"]).all())       # AR fallback beyond the path
