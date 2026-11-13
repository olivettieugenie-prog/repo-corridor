"""The display table: load_daily's fallbacks, and the header date following it.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import desk  # noqa: E402
import live_feed as lf  # noqa: E402
from i18n import LANGS, STRINGS, date_short  # noqa: E402

MAIN = ROOT / "data" / "processed" / "daily.csv"
BAD_URL = "https://example.invalid/daily.csv"


def truncated(tmp: Path, last: str = "2026-09-30") -> Path:
    """A daily table that ends on `last`, standing in for the rebuilt one."""
    d = pd.read_csv(MAIN, parse_dates=["date"])
    path = tmp / "daily_live.csv"
    d[d["date"] <= last].to_csv(path, index=False)
    return path


class TestLoadDaily(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_failed_fetch_without_local_copy_falls_back_to_main(self):
        df, source = lf.load_daily(BAD_URL, self.dir / "missing.csv")
        self.assertEqual(source, "main")
        self.assertEqual(df["date"].max(), pd.read_csv(MAIN, parse_dates=["date"])["date"].max())

    def test_failed_fetch_uses_the_local_copy(self):
        df, source = lf.load_daily(BAD_URL, truncated(self.dir))
        self.assertEqual(source, "local")
        self.assertEqual(df["date"].max(), pd.Timestamp("2026-09-30"))

    def test_successful_fetch_refreshes_the_local_copy(self):
        remote = truncated(self.dir)
        local = self.dir / "copy" / "daily.csv"
        df, source = lf.load_daily(remote.as_uri(), local)
        self.assertEqual(source, "github")
        self.assertTrue(local.exists())
        self.assertEqual(df["date"].max(), pd.Timestamp("2026-09-30"))

    def test_a_broken_file_is_not_accepted(self):
        bad = self.dir / "bad.csv"
        bad.write_text("date,other\n2026-11-01,1\n")       # no sofr column
        df, source = lf.load_daily(bad.as_uri(), self.dir / "missing.csv")
        self.assertEqual(source, "main")


class TestHeaderFollowsTheTable(unittest.TestCase):
    KEYS = ("REPO_DAILY_URL", "REPO_DAILY_LOCAL", "REPO_LIVE_URL", "REPO_LIVE_LOCAL",
            "REPO_FORWARD_URL", "REPO_FORWARD_LOCAL", "REPO_TODAY")

    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in self.KEYS}
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.tmp.cleanup()

    def test_header_date_is_the_max_date_of_the_loaded_table(self):
        from streamlit.testing.v1 import AppTest
        live = truncated(self.dir)
        os.environ.update(REPO_DAILY_URL=live.as_uri(), REPO_DAILY_LOCAL=str(self.dir / "c.csv"),
                          REPO_LIVE_URL=BAD_URL, REPO_LIVE_LOCAL=str(self.dir / "none.csv"),
                          REPO_FORWARD_URL=BAD_URL,
                          REPO_FORWARD_LOCAL=str(ROOT / "tests" / "fixtures" / "forward.csv"),
                          REPO_TODAY="2026-10-05")
        last = pd.read_csv(live, parse_dates=["date"])["date"].max()
        for lang in LANGS:
            with self.subTest(lang=lang):
                at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
                at.run()
                if lang != "en":
                    at.segmented_control(key="lang").set_value(lang).run()
                self.assertEqual(len(at.exception), 0, at.exception)
                T = STRINGS[lang]
                header = T["hdr_asof_nolive"].split("{")[0] + desk.long_date(last, lang)
                self.assertTrue(at.caption[0].value.startswith(header))
                help_ = [m.help for m in at.metric if m.label == T["tile_days"]][0]
                self.assertEqual(help_, T["tile_days_help"].format(last=date_short(last, lang)))

    def test_header_shows_each_balance_date_and_the_target(self):
        """ON RRP and TGA keep their own dates (not the later of the two) and
        the header names the date the forecast is for."""
        from streamlit.testing.v1 import AppTest
        from test_desk import make_log
        log = make_log(3, 0, target_end=pd.Timestamp("2026-11-12"))
        log.loc[log.index[-1], ["rrp_date", "tga_date"]] = ["2026-11-10", "2026-11-06"]
        live = self.dir / "forecast.csv"
        log.to_csv(live, index=False)
        os.environ.update(REPO_LIVE_URL=live.as_uri(), REPO_LIVE_LOCAL=str(self.dir / "copy.csv"),
                          REPO_DAILY_URL=BAD_URL, REPO_DAILY_LOCAL=str(self.dir / "none.csv"),
                          REPO_FORWARD_URL=BAD_URL,
                          REPO_FORWARD_LOCAL=str(ROOT / "tests" / "fixtures" / "forward.csv"),
                          REPO_TODAY="2026-11-12")
        rates = pd.read_csv(MAIN, parse_dates=["date"])["date"].max()
        for lang in LANGS:
            with self.subTest(lang=lang):
                at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
                at.run()
                if lang != "en":
                    at.segmented_control(key="lang").set_value(lang).run()
                self.assertEqual(len(at.exception), 0, at.exception)
                T = STRINGS[lang]
                d = lambda x: desk.long_date(x, lang)  # noqa: E731
                header = at.caption[0].value
                expected = T["hdr_asof"].format(
                    rates=d(rates), rrp=d("2026-11-10"), tga=d("2026-11-06"),
                    target=d("2026-11-12"), run=desk.run_time_et(log["run_time_utc"].iloc[-1], lang))
                self.assertEqual(header, expected)
                # TGA shows its own, earlier date, not the later ON RRP one.
                self.assertIn(("TGA to " if lang == "en" else "TGA au ") + d("2026-11-06"), header)
                self.assertNotIn(("TGA to " if lang == "en" else "TGA au ") + d("2026-11-10"), header)


if __name__ == "__main__":
    unittest.main()
