"""Checks for the language dictionary and a headless run of the app.

Uses only the standard library and Streamlit, so it needs no extra package:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from i18n import LANGS, NNBSP, STRINGS, num  # noqa: E402


class TestDictionary(unittest.TestCase):
    def test_same_keys_in_every_language(self):
        en = STRINGS["en"]
        for lang in LANGS:
            d = STRINGS[lang]
            self.assertEqual(set(d), set(en), f"top-level keys differ in {lang}")
            for k, v in en.items():
                if isinstance(v, dict):
                    self.assertEqual(set(d[k]), set(v), f"{k} keys differ in {lang}")
                if isinstance(v, list):
                    self.assertEqual(len(d[k]), len(v), f"{k} length differs in {lang}")

    def test_no_empty_strings(self):
        for lang in LANGS:
            for k, v in STRINGS[lang].items():
                if isinstance(v, str):
                    self.assertTrue(v.strip(), f"{lang}.{k} is empty")

    def test_placeholders_match(self):
        for k, v in STRINGS["en"].items():
            if isinstance(v, str):
                ph = set(re.findall(r"\{(\w+)\}", v))
                self.assertEqual(ph, set(re.findall(r"\{(\w+)\}", STRINGS["fr"][k])), k)

    def test_acronyms_kept_in_french(self):
        fr = " ".join(v for v in STRINGS["fr"].values() if isinstance(v, str))
        for a in ("SOFR", "EFFR", "ON RRP"):
            self.assertIn(a, fr)

    def test_number_format(self):
        self.assertEqual(num(12345.678, 2, "en"), "12,345.68")
        self.assertEqual(num(12345.678, 2, "fr"), f"12{NNBSP}345,68")
        self.assertEqual(num(0.287, 3, "fr", sign=True), "+0,287")
        self.assertEqual(num(-0.201, 3, "en", sign=True), "-0.201")


class TestAppRuns(unittest.TestCase):
    def run_app(self, lang: str, theme: str):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
        at.context["theme"] = theme
        at.run()
        if lang != "en":
            at.segmented_control(key="lang").set_value(lang).run()
        return at

    def test_all_combinations(self):
        for lang in LANGS:
            for theme in ("light", "dark"):
                with self.subTest(lang=lang, theme=theme):
                    at = self.run_app(lang, theme)
                    self.assertEqual(len(at.exception), 0, at.exception)
                    self.assertEqual(at.session_state["lang"], lang)
                    self.assertEqual(at.title[0].value, STRINGS[lang]["title"])
                    self.assertIn(STRINGS[lang]["s1"], [h.value for h in at.subheader])

    def test_default_is_english(self):
        at = self.run_app("en", "light")
        self.assertEqual(at.session_state["lang"], "en")


if __name__ == "__main__":
    unittest.main()
