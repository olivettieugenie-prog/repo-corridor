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
