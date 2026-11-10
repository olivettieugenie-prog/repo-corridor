"""Desk view: AppTest runs in both languages and themes, plus the stale
warning, a failed fetch of the data branch, and the track-record threshold.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import desk  # noqa: E402
from i18n import LANGS, STRINGS, num  # noqa: E402
from market_calendar import business_days  # noqa: E402

TARGET = pd.Timestamp("2026-11-30")   # a Monday and a month-end
TODAY = "2026-11-30"


def make_log(n: int, scored: int, target_end=TARGET, unconfirmed: int = 0) -> pd.DataFrame:
    targets = business_days(target_end - pd.Timedelta(days=3 * n + 10), target_end)[-n:]
    rng = np.random.default_rng(1)
    f = rng.normal(0, 1.5, n)
    df = pd.DataFrame({
        "run_time_utc": [f"{t:%Y-%m-%d}T14:35:00Z" for t in targets],
        "as_of_date": [(t - pd.offsets.BDay()).date() for t in targets],
        "target_date": [t.date() for t in targets],
        "forecast_bp": f, "lower_bp": f - 2.0, "upper_bp": f + 2.5, "ar1_bp": f * 1.1,
        "month_end": [int(t == targets[-1]) for t in targets],
        "quarter_end": 0, "year_end": 0, "mid_settle": 0, "month_start": 0,
        "rate_change": 0, "range_unconfirmed": unconfirmed,
        "target_lo": 3.75, "target_hi": 4.0, "base_bp": -11.0,
        "implied_sofr_pct": 4.0 + (-11.0 + f) / 100,
        "implied_sofr_lo_pct": 4.0 + (-13.0 + f) / 100,
        "implied_sofr_hi_pct": 4.0 + (-8.5 + f) / 100,
        "rrp_date": [t.date() for t in targets], "rrp_bn": 4.2, "d_rrp_bn": -0.8,
        "tga_date": [t.date() for t in targets], "tga_bn": 870.0, "d_tga_bn": 12.0,
    })
    y = f + rng.normal(0, 1.5, n)
    df["realised_bp"] = np.where(np.arange(n) < scored, y, np.nan)
    df["hit"] = np.where(np.arange(n) < scored,
                         ((y >= df["lower_bp"]) & (y <= df["upper_bp"])).astype(float), np.nan)
    df["realised_sofr_pct"] = np.nan
    df["scored_at_utc"] = ""
    return df


class DeskAppTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = {k: os.environ.get(k) for k in ("REPO_LIVE_URL", "REPO_LIVE_LOCAL", "REPO_TODAY",
                                                   "REPO_FORWARD_URL", "REPO_FORWARD_LOCAL")}
        os.environ["REPO_FORWARD_URL"] = "https://example.invalid/forward.csv"
        os.environ["REPO_FORWARD_LOCAL"] = str(ROOT / "tests" / "fixtures" / "forward.csv")

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def serve(self, df: pd.DataFrame | None, fetch_ok: bool = True, local: pd.DataFrame | None = None,
              today: str = TODAY, tag: str = "a") -> None:
        remote = self.dir / f"remote_{tag}.csv"
        if df is not None:
            df.to_csv(remote, index=False)
        local_path = self.dir / f"local_{tag}.csv"
        if local is not None:
            local.to_csv(local_path, index=False)
        os.environ["REPO_LIVE_URL"] = (remote.as_uri() if fetch_ok
                                       else f"https://example.invalid/{tag}.csv")
        os.environ["REPO_LIVE_LOCAL"] = str(local_path)
        os.environ["REPO_TODAY"] = today

    def run_app(self, lang: str = "en", theme: str = "light"):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
        at.context["theme"] = theme
        at.run()
        if lang != "en":
            at.segmented_control(key="lang").set_value(lang).run()
        self.assertEqual(len(at.exception), 0, at.exception)
        return at

    @staticmethod
    def desk_warnings(at) -> list[str]:
        """Warnings raised by the desk panel only; the calculator further down
        has its own (an FOMC meeting in its default window, for example).
        Each desk template is matched on its fixed text before or after the
        {target} placeholder."""
        marks = []
        for lang in LANGS:
            marks.append(STRINGS[lang]["desk_stale"].split("{")[0])
            marks.append(STRINGS[lang]["desk_unconfirmed"].split("}", 1)[1][:30])
        return [w.value for w in at.warning if any(m in w.value for m in marks)]

    @staticmethod
    def texts(at) -> str:
        parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
        parts += [w.value for w in at.warning] + [i.value for i in at.info]
        parts += [m.label for m in at.metric] + [str(m.value) for m in at.metric]
        return "\n".join(parts)

    def test_fresh_forecast_all_languages_and_themes(self):
        for lang in LANGS:
            for theme in ("light", "dark"):
                with self.subTest(lang=lang, theme=theme):
                    T = STRINGS[lang]
                    self.serve(make_log(5, 3), tag=f"fresh_{lang}_{theme}")
                    at = self.run_app(lang, theme)
                    self.assertIn(T["track_title"], [h.value for h in at.subheader])
                    txt = self.texts(at)
                    self.assertEqual(self.desk_warnings(at), [])
                    line1, line2 = desk.morning_note(
                        make_log(5, 3).iloc[-1].pipe(lambda r: r.set_axis(r.index)), lang)
                    self.assertIn(T["flag_month_end"], txt)
                    self.assertIn(T["desk_forecast"], txt)
                    self.assertIn(T["desk_sofr"], txt)
                    self.assertIn(line2.split(".")[0], txt)
                    log = make_log(5, 3)
                    start = desk.long_date(log["target_date"].min(), lang)
                    hit = log.dropna(subset=["realised_bp"])["hit"].mean()
                    self.assertIn(T["track_hit"].format(hit=num(100 * hit, 0, lang), n=3,
                                                        start=start), txt)
                    self.assertNotIn(T["desk_src_local"], txt)

    def test_stale_forecast_warning(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                self.serve(make_log(5, 3), today="2026-12-03", tag=f"stale_{lang}")
                at = self.run_app(lang)
                target = desk.long_date(TARGET, lang)
                self.assertEqual(self.desk_warnings(at),
                                 [STRINGS[lang]["desk_stale"].format(target=target)])

    def test_one_business_day_old_is_not_stale(self):
        self.serve(make_log(5, 3), today="2026-12-01", tag="oneday")
        self.assertEqual(self.desk_warnings(self.run_app()), [])

    def test_failed_fetch_falls_back_to_local_copy(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                self.serve(None, fetch_ok=False, local=make_log(5, 3), tag=f"fallback_{lang}")
                at = self.run_app(lang)
                txt = self.texts(at)
                self.assertIn(STRINGS[lang]["desk_src_local"], txt)
                self.assertIn(STRINGS[lang]["desk_forecast"], txt)

    def test_failed_fetch_without_local_copy(self):
        for lang in LANGS:
            with self.subTest(lang=lang):
                self.serve(None, fetch_ok=False, tag=f"none_{lang}")
                at = self.run_app(lang)
                self.assertEqual([i.value for i in at.info], [STRINGS[lang]["desk_none"]])

    def test_track_record_from_twenty_scored_days(self):
        log = make_log(26, 25)
        tr = desk.track_record(log)
        for lang in LANGS:
            with self.subTest(lang=lang):
                self.serve(log, tag=f"track_{lang}")
                txt = self.texts(self.run_app(lang))
                self.assertIn(f"{tr['n']} ", txt)
                self.assertIn(STRINGS[lang]["track_hit"].split("{")[0], txt)
                head = STRINGS[lang]["desk_track"].split("{")[0]
                self.assertIn(head, txt)

    def test_unconfirmed_range_warning(self):
        self.serve(make_log(5, 3, unconfirmed=1), tag="fomc")
        at = self.run_app()
        w = self.desk_warnings(at)
        self.assertEqual(len(w), 1)
        self.assertIn("FOMC", w[0])


class ClientNote(unittest.TestCase):
    def test_month_end_template(self):
        prints = pd.Series([-12.0, 3.0, 22.0])
        for lang in LANGS:
            T = STRINGS[lang]
            note = desk.client_note("month-end", "2026-11-30", 14, {"q50": -6.7, "q90": 5.8},
                                    prints, lang)
            l1, l2 = note.split("\n")
            self.assertEqual(l1, T["client_line1"].format(
                Kind=T["note_kind_month-end"], days=14, date=desk.long_date("2026-11-30", lang),
                q50=num(-6.7, 1, lang, True), q90=num(5.8, 1, lang, True)))
            self.assertEqual(l2, T["client_line2"].format(kinds=T["turns_month-end"],
                                                          a=num(-12, 0, lang, True),
                                                          b=num(22, 0, lang, True)))

    def test_year_end_is_history_only(self):
        for lang in LANGS:
            note = desk.client_note("year-end", "2026-12-31", 36, None,
                                    pd.Series([-20.0, 50.0]), lang)
            self.assertTrue(note.split("\n")[1].startswith(
                STRINGS[lang]["client_line2_ye"].split("{")[0]))


class DeskLogic(unittest.TestCase):
    def test_track_record_metrics(self):
        log = make_log(30, 22)
        tr = desk.track_record(log)
        s = log.dropna(subset=["realised_bp"])
        self.assertEqual(tr["n"], 22)
        self.assertAlmostEqual(tr["hit"], s["hit"].mean())
        self.assertAlmostEqual(tr["rmse"], float(np.sqrt(((s.realised_bp - s.forecast_bp) ** 2).mean())))
        self.assertAlmostEqual(tr["rmse_ar1"], float(np.sqrt(((s.realised_bp - s.ar1_bp) ** 2).mean())))
        self.assertNotIn("hit", desk.track_record(make_log(30, 19)))

    def test_note_is_template_only(self):
        row = make_log(3, 0).iloc[-1]
        for lang in LANGS:
            l1, l2 = desk.morning_note(row, lang)
            T = STRINGS[lang]
            key = "note_flat" if abs(row.forecast_bp) < desk.FLAT_BP else "note_line1"
            self.assertTrue(l1.startswith(T[key].split("{")[0]))
            self.assertTrue(l2.startswith(T["note_line2_flags"].split("{")[0]))


if __name__ == "__main__":
    unittest.main()
