"""Four design options for the app: fonts, surfaces and chart colours.

Every chart colour here was generated inside the OKLCH lightness band for its
mode and then run through a palette validator that checks the lightness band,
a chroma floor, colour-blind separation of adjacent pairs, normal-vision
separation and contrast against the surface. All eight palettes pass in both
light and dark. Colours were not chosen by eye.

Set the active design with the REPO_DESIGN environment variable, or change
ACTIVE below. `python src/design.py <name>` writes .streamlit/config.toml for
that design, which is what makes Streamlit's own light/dark switch use it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Google Fonts families. config_toml() turns them into theme.font and
# theme.headingFont values, which is how Streamlit loads them.
DESIGNS: dict[str, dict] = {
    "marine": {
        "label": "A — Marine",
        "note": "Your brief, read literally: navy and electric blue for dark, "
                "lilac-white and deep blue for light. Roboto throughout, which "
                "has tabular numerals so columns of basis points line up.",
        "head_font": "Roboto",
        "body_font": "Roboto",
        "light": {"bg": "#F8F8F9", "surface": "#FFFFFF", "ink": "#111439",
                  "muted": "#5B5E7A", "grid": "#DDDDE6", "primary": "#1A66D4",
                  "blue": "#1A66D4", "orange": "#93690D", "red": "#C03A1B",
                  "band": "#E3E3EC"},
        "dark": {"bg": "#01257D", "surface": "#062F90", "ink": "#F8F8F9",
                 "muted": "#A9B6E0", "grid": "#1A3C95", "primary": "#06999A",
                 "blue": "#06999A", "orange": "#AD7C0E", "red": "#E0603C",
                 "band": "#0B3AA0"},
    },
    "tempered": {
        "label": "B — Marine tempered, Roboto",
        "note": "Same family, but dark mode sits on near-black rather than "
                "saturated navy, which is easier to read for long stretches "
                "and is what most trading dashboards do. Roboto for headings "
                "and body text alike, whose tabular numerals keep columns of "
                "basis points aligned.",
        "head_font": "Roboto",
        "body_font": "Roboto",
        "light": {"bg": "#F8F8F9", "surface": "#FFFFFF", "ink": "#111439",
                  "muted": "#5B5E7A", "grid": "#DEDEE7", "primary": "#0267C7",
                  "blue": "#0267C7", "orange": "#C31405", "red": "#C03A1B",
                  "band": "#E4E4EC"},
        "dark": {"bg": "#0B1020", "surface": "#141A2E", "ink": "#E8EAF2",
                 "muted": "#9AA3C0", "grid": "#242C45", "primary": "#0C84FA",
                 "blue": "#0C84FA", "orange": "#EB3B29", "red": "#E0603C",
                 "band": "#1C2440"},
    },
    "institutional": {
        "label": "C — Institutional",
        "note": "The look of your own course workbook, so the project and your "
                "study materials match. IBM Plex Sans, a blue-teal accent and "
                "rust. The most sober of the four.",
        "head_font": "IBM Plex Sans Condensed",
        "body_font": "IBM Plex Sans",
        "light": {"bg": "#FBFBF9", "surface": "#FFFFFF", "ink": "#16212C",
                  "muted": "#56636F", "grid": "#D9DDE1", "primary": "#01729F",
                  "blue": "#01729F", "orange": "#A94608", "red": "#C03A1B",
                  "band": "#E4E8EB"},
        "dark": {"bg": "#0F151B", "surface": "#161E26", "ink": "#E4EAF0",
                 "muted": "#9AA8B5", "grid": "#2A3641", "primary": "#11957C",
                 "blue": "#11957C", "orange": "#CF5604", "red": "#E0603C",
                 "band": "#1E2832"},
    },
    "editorial": {
        "label": "D — Editorial",
        "note": "The most modern of the four: more whitespace, a violet-blue "
                "accent, deep indigo for dark. Space Grotesk headings over "
                "Inter. Looks least like a bank and most like a product.",
        "head_font": "Space Grotesk",
        "body_font": "Inter",
        "light": {"bg": "#FFFFFF", "surface": "#FAFAFA", "ink": "#0A0A0A",
                  "muted": "#5A5A66", "grid": "#E4E4E9", "primary": "#5156E8",
                  "blue": "#5156E8", "orange": "#7E6F08", "red": "#C03A1B",
                  "band": "#ECECF2"},
        "dark": {"bg": "#0A0E27", "surface": "#121735", "ink": "#F1F5F9",
                 "muted": "#9BA3C4", "grid": "#222848", "primary": "#6773FE",
                 "blue": "#6773FE", "orange": "#988710", "red": "#E0603C",
                 "band": "#1A2040"},
    },
}

ACTIVE = "tempered"


def active_name() -> str:
    import os
    return os.environ.get("REPO_DESIGN", ACTIVE)


def _gf(family: str, weights: str) -> str:
    """Streamlit's supported 'Name:url' font value, pointing at Google Fonts."""
    q = family.replace(" ", "+")
    return (f"{family}:https://fonts.googleapis.com/css2?"
            f"family={q}:wght@{weights}&display=swap")


def config_toml(name: str) -> str:
    d = DESIGNS[name]
    lo, dk = d["light"], d["dark"]
    return f"""# Generated by src/design.py for the "{name}" design. Do not hand-edit.
# Both palettes are declared, so Streamlit's own light/dark switch (the
# Settings menu, or the viewer's system preference) picks between them and the
# charts follow via st.context.theme.
#
# Fonts use Streamlit's supported "Name:url" form rather than injected CSS:
# st.html strips <link> tags, so a stylesheet link never loads, and Streamlit's
# own heading rules outrank an injected h1 selector.
[theme]
font = "{_gf(d['body_font'], '400;500;600')}"
headingFont = "{_gf(d['head_font'], '600;700')}"

[theme.light]
primaryColor = "{lo['primary']}"
backgroundColor = "{lo['bg']}"
secondaryBackgroundColor = "{lo['surface']}"
textColor = "{lo['ink']}"

[theme.dark]
primaryColor = "{dk['primary']}"
backgroundColor = "{dk['bg']}"
secondaryBackgroundColor = "{dk['surface']}"
textColor = "{dk['ink']}"

[browser]
gatherUsageStats = false
"""


def write_config(name: str) -> Path:
    if name not in DESIGNS:
        raise SystemExit(f"unknown design {name!r}; choose from {list(DESIGNS)}")
    path = ROOT / ".streamlit" / "config.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(config_toml(name))
    return path


if __name__ == "__main__":
    n = sys.argv[1] if len(sys.argv) > 1 else ACTIVE
    print("wrote", write_config(n), f"for design {n!r}")
