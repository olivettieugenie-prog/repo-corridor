"""Every user-facing string in the app, in English and French.

STRINGS[lang][key] holds one string per key. app.py reads the active language
from st.session_state["lang"] and looks every label up here, so adding or
rewording text means editing this file only. Both languages must carry the
same keys; tests/test_i18n.py checks that.

French text is written with ordinary spaces and passed through _fr_typography
once at import, which puts the no-break spaces French typesetting needs before
: ; ? ! % and between a number and its unit, so a line never breaks there.

num() formats a number for the active language: comma thousands and a decimal
point in English, a narrow no-break space for thousands and a decimal comma in
French.
"""
from __future__ import annotations

import re

LANGS = ("en", "fr")
LANG_NAMES = {"en": "English", "fr": "Français"}

NBSP = " "         # no-break space
NNBSP = " "        # narrow no-break space


def num(x: float, dec: int = 0, lang: str = "en", sign: bool = False) -> str:
    """Format x with `dec` decimals, thousands grouping and an optional sign."""
    s = f"{x:{'+' if sign else ''},.{dec}f}"
    if lang == "fr":
        s = s.replace(",", NNBSP).replace(".", ",")
    return s


def date_short(ts, lang: str = "en") -> str:
    """17 Sep 2019 / 17 sept. 2019, without depending on the system locale."""
    return f"{ts.day} {STRINGS[lang]['months'][ts.month - 1]} {ts.year}"


EN: dict[str, object] = {
    "page_title": "Repo in the Fed's corridor",
    "lang_label": "Language",
    "title": "Where repo prints inside the Fed's corridor",
    "intro": (
        "Overnight repo trades between two administered rates: the ON RRP "
        "facility below and interest on reserves above. This page asks what "
        "moves SOFR inside that corridor, whether a penalised regression "
        "predicts it better than an autoregression, and whether that "
        "prediction is worth any money. Public data only, from the New York "
        "Fed and the US Treasury."
    ),
    "bp": "bp",
    "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],

    # stat tiles
    "tile_days": "SOFR days covered",
    "tile_days_help": "2 April 2018 to 26 October 2026, from the NY Fed API",
    "tile_skill": "Model skill vs AR(1), 2024+",
    "tile_skill_help": (
        "Out-of-sample R squared of the two-stage LASSO against an "
        "autoregression, rolling 500-day training window. Diebold-Mariano "
        "test: significant since 2024 (p = {p24}), not over the full period "
        "from May 2020 (R² {r2all}, p = {pall})."
    ),
    "tile_carry": "Carry from always lending",
    "tile_carry_help": (
        "Mean annualised pickup of overnight repo over the ON RRP facility, "
        "on the 1,336 days where the award rate is known"
    ),
    "tile_peak": "Carry on 17 Sep 2019",
    "tile_peak_help": (
        "The day the plumbing broke: SOFR printed 5.25% while the Fed's "
        "facility paid 2.00%. Normal range is about -10 to +30 bp."
    ),

    # 1. corridor
    "s1": "1. The corridor",
    "date_range": "Date range",
    "slider_format": "MMM YYYY",
    "series_sofr": "SOFR",
    "series_award": "ON RRP award rate",
    "ax_percent": "percent",
    "ax_position": "SOFR over award rate, bp (clipped at {clip})",
    "tt_date": "date",
    "tt_series": "series",
    "tt_percent": "percent",
    "tt_carry": "carry, bp",
    "ch_band": "Shaded band is the FOMC target range",
    "ch_position": "Where SOFR sits above the facility floor",
    "cap1": (
        "The administered rates come from a table of 15 Federal Reserve "
        "implementation notes, each cited in the repository, because the "
        "offsets are not stable. ON RRP sat at the bottom of the range until "
        "September 2019, 5 bp below it through to March 2020, 5 bp above it "
        "from June 2021, and back at the bottom from 19 December 2024. The "
        "table reproduces all 491 directly observed award rates."
    ),

    # 2. calendar
    "s2": "2. Calendar pressure only bites when reserves are scarce",
    "ax_bp": "basis points",
    "ax_rrp": "USD bn, daily average",
    "tt_year": "year",
    "tt_premium": "premium, bp",
    "tt_rrp": "ON RRP, $bn",
    "ch_premium": "Month-end premium on the spread",
    "ch_rrp": "ON RRP usage",
    "cap2": (
        "Two charts rather than two axes on one. The month-end premium is "
        "about 9 to 13 bp in 2018 and 2019, essentially zero from 2020 to 2022 "
        "when ON RRP usage ran near $2.0tn, and returns as the facility "
        "drains. With the system awash with cash, month-end balance-sheet "
        "constraints had nothing to push against."
    ),

    # 3. skill
    "s3": "3. Does the model add anything to an autoregression?",
    "ax_r2": "out-of-sample R² vs AR(1)",
    "tt_r2": "R² vs AR(1)",
    "ch_r2": "Positive means the LASSO beat the autoregression",
    "cap3": (
        "Zero is a weak benchmark for this target and is not used. The "
        "trailing median removes the level of the spread but not a slow trend "
        "in it, so the deviation stays autocorrelated at 0.81 and an "
        "autoregression alone reaches R² 0.64 against zero. The bar direction "
        "carries the sign, so colour is not the only cue. The model loses in "
        "2020 and 2021 because the rolling window has to pass through the "
        "regime break before it adapts."
    ),
    "cap3_dm": (
        "Is the gain significant? A Diebold-Mariano test on squared errors, "
        "with a HAC variance, finds it significant from 2024 onwards "
        "(p = {p24}) but not over the whole period from May 2020 (p = {pall}): "
        "the losses of 2020 offset the later gains."
    ),

    # 3b. importance
    "s3b": "3b. What is the model actually using?",
    "kind_one": "one at a time",
    "kind_group": "shuffled as a group",
    "ax_rmse": "increase in out-of-sample RMSE, bp",
    "tt_predictor": "predictor",
    "tt_rmse": "RMSE increase, bp",
    "tt_p10": "10th percentile, bp",
    "tt_p90": "90th percentile, bp",
    "ch_importance": "Scramble a predictor and see how much worse the forecast gets",
    # Group names from permutation_importance.csv. Single predictors keep
    # their variable names, which are the same in both languages.
    "groups": {
        "calendar, all five": "calendar, all five",
        "month/quarter/year nest": "month/quarter/year nest",
        "dispersion pair": "dispersion pair",
        "rate-change pair": "rate-change pair",
        "balance sheet, all four": "balance sheet, all four",
    },
    "cap3b": (
        "Shuffled in blocks of 21 days so the serial structure survives, 30 "
        "times each; whiskers span the 10th to 90th percentile of those "
        "repeats. Correlated predictors are also shuffled together, because "
        "the calendar flags are nested: every quarter-end is a month-end and "
        "every year-end is a quarter-end, so one at a time they cover for each "
        "other. The four balance-sheet variables contribute nothing even "
        "jointly, at −0.005 bp. Negative means the shuffle happened to help, "
        "which is noise."
    ),

    # 4. money
    "s4": "4. Is the skill worth any money? No",
    "s4_intro": (
        "Place tomorrow's overnight cash in repo at SOFR or at the facility at "
        "the award rate, decided the day before. The carry splits exactly into "
        "`base + deviation + corridor`, and base and corridor are both known a "
        "day ahead. So the benchmark that matters is a rule given only the "
        "known part, not passive always-lend, which would credit the model for "
        "the corridor."
    ),
    "col_rule": "rule",
    "col_net": "net bp",
    "col_vs_always": "vs always-lend",
    "col_vs_known": "vs known-only",
    "rule_always": "always lend",
    "rule_known": "known only, no model",
    "rule_ar1": "AR(1)",
    "rule_lasso": "two-stage LASSO",
    "rule_perfect": "perfect foresight",
    "conclusion": (
        "The entire opportunity is **0.303 bp** of annualised yield. A rule "
        "with no model captures two thirds of it, and the LASSO's margin over "
        "an autoregression is 0.003 bp, which is nothing. At a 1 bp switching "
        "cost the LASSO falls behind AR(1), +0.155 against +0.186. The Fed's "
        "two facilities bound the corridor so tightly that the overnight "
        "placement decision has almost no money in it. In 2022 always-lend "
        "gave up 1.55 bp against the facility and every smarter rule merely "
        "returns to zero, which is why about $2.0tn sat at ON RRP: not "
        "chasing carry, but because the facility simply paid more."
    ),
    "sources": (
        "Sources: SOFR, EFFR with the FOMC target range, and ON RRP operations "
        "from the Federal Reserve Bank of New York markets API. Treasury "
        "General Account from the US Treasury Fiscal Data API. "
        "Administered-rate offsets read from Federal Reserve implementation "
        "notes. Built with pandas and scikit-learn; no paid data. Limitations "
        "and the extensions left out are in the repository README."
    ),
}

FR: dict[str, object] = {
    "page_title": "Le repo dans le corridor de la Fed",
    "lang_label": "Langue",
    "title": "Où se fixe le repo dans le corridor de la Fed",
    "intro": (
        "Le repo au jour le jour se traite entre deux taux administrés : la "
        "facilité ON RRP en bas et la rémunération des réserves en haut. Cette "
        "page examine ce qui fait varier le SOFR à l'intérieur de ce corridor, "
        "si une régression pénalisée le prévoit mieux qu'un modèle "
        "autorégressif, et si cette prévision a une valeur économique. "
        "Données publiques uniquement, issues de la Fed de New York et du "
        "Trésor américain."
    ),
    "bp": "pb",
    "months": ["janv.", "févr.", "mars", "avr.", "mai", "juin",
               "juil.", "août", "sept.", "oct.", "nov.", "déc."],

    "tile_days": "Jours de SOFR couverts",
    "tile_days_help": "Du 2 avril 2018 au 26 octobre 2026, via l'API de la Fed de New York",
    "tile_skill": "Apport du modèle face à l'AR(1), 2024+",
    "tile_skill_help": (
        "R² hors échantillon du LASSO en deux étapes par rapport à un modèle "
        "autorégressif, fenêtre d'estimation glissante de 500 jours. Test de "
        "Diebold-Mariano : gain significatif depuis 2024 (p = {p24}), mais pas "
        "sur l'ensemble de la période depuis mai 2020 (R² {r2all}, p = {pall})."
    ),
    "tile_carry": "Portage du prêt systématique",
    "tile_carry_help": (
        "Surplus annualisé moyen du repo au jour le jour sur la facilité "
        f"ON RRP, sur les 1{NNBSP}336 jours où le taux servi est connu"
    ),
    "tile_peak": "Portage le 17 sept. 2019",
    "tile_peak_help": (
        "Le jour où la mécanique du marché a cédé : le SOFR s'est fixé à "
        "5,25 % alors que la facilité de la Fed rémunérait à 2,00 %. La plage "
        "habituelle va d'environ -10 à +30 pb."
    ),

    "s1": "1. Le corridor",
    "date_range": "Période",
    "slider_format": "MM/YYYY",
    "series_sofr": "SOFR",
    "series_award": "Taux servi ON RRP",
    "ax_percent": "en %",
    "ax_position": "SOFR moins taux servi, pb (écrêté à {clip})",
    "tt_date": "date",
    "tt_series": "série",
    "tt_percent": "en %",
    "tt_carry": "portage, pb",
    "ch_band": "Zone ombrée : fourchette cible du FOMC",
    "ch_position": "Le SOFR au-dessus du plancher de la facilité",
    "cap1": (
        "Les taux administrés proviennent d'un tableau de 15 notes de mise en "
        "œuvre de la Réserve fédérale, chacune citée dans le dépôt, car les "
        "écarts ne sont pas stables. L'ON RRP se situait au bas de la "
        "fourchette jusqu'en septembre 2019, 5 pb en dessous jusqu'en mars "
        "2020, 5 pb au-dessus à partir de juin 2021, puis de nouveau au bas de "
        "la fourchette depuis le 19 décembre 2024. Le tableau reproduit les "
        "491 taux servis observés directement."
    ),

    "s2": "2. Les effets de calendrier ne jouent que lorsque les réserves sont rares",
    "ax_bp": "points de base",
    "ax_rrp": "Md USD, moyenne quotidienne",
    "tt_year": "année",
    "tt_premium": "prime, pb",
    "tt_rrp": "ON RRP, Md USD",
    "ch_premium": "Prime de fin de mois sur l'écart",
    "ch_rrp": "Recours à l'ON RRP",
    "cap2": (
        "Deux graphiques plutôt que deux axes sur un seul. La prime de fin de "
        "mois avoisine 9 à 13 pb en 2018 et 2019, elle est quasi nulle de 2020 "
        "à 2022, lorsque le recours à l'ON RRP approchait "
        f"2{NNBSP}000 milliards de dollars, et réapparaît à mesure que la "
        "facilité se vide. Dans un système saturé de liquidités, les "
        "contraintes de bilan de fin de mois n'avaient aucune prise."
    ),

    "s3": "3. Le modèle apporte-t-il quelque chose à un modèle autorégressif ?",
    "ax_r2": "R² hors échantillon face à l'AR(1)",
    "tt_r2": "R² face à l'AR(1)",
    "ch_r2": "Valeur positive : le LASSO bat l'AR(1)",
    "cap3": (
        "Zéro est une référence trop faible pour cette variable et n'est pas "
        "retenu. La médiane glissante retire le niveau de l'écart mais pas sa "
        "tendance lente, si bien que la déviation reste autocorrélée à 0,81 et "
        "qu'un modèle autorégressif seul atteint un R² de 0,64 face à zéro. Le "
        "sens des barres porte le signe, la couleur n'est donc pas le seul "
        "repère. Le modèle perd en 2020 et 2021, car la fenêtre glissante doit "
        "traverser la rupture de régime avant de s'adapter."
    ),
    "cap3_dm": (
        "Le gain est-il significatif ? Un test de Diebold-Mariano sur les "
        "erreurs quadratiques, avec variance HAC, le juge significatif depuis "
        "2024 (p = {p24}), mais pas sur l'ensemble de la période depuis mai "
        "2020 (p = {pall}) : les pertes de 2020 compensent les gains ultérieurs."
    ),

    "s3b": "3b. Sur quoi le modèle s'appuie-t-il réellement ?",
    "kind_one": "un par un",
    "kind_group": "permutés en bloc",
    "ax_rmse": "hausse de la RMSE, pb",
    "tt_predictor": "prédicteur",
    "tt_rmse": "hausse de la RMSE, pb",
    "tt_p10": "10e centile, pb",
    "tt_p90": "90e centile, pb",
    "ch_importance": "Dégradation de la prévision après permutation",
    "groups": {
        "calendar, all five": "calendrier, les cinq",
        "month/quarter/year nest": "imbrication mois/trimestre/année",
        "dispersion pair": "paire de dispersion",
        "rate-change pair": "paire de variation de taux",
        "balance sheet, all four": "bilan, les quatre",
    },
    "cap3b": (
        "Permutation par blocs de 21 jours pour préserver la structure "
        "sérielle, répétée 30 fois ; les moustaches couvrent du 10e au 90e "
        "centile de ces répétitions. Les prédicteurs corrélés sont aussi "
        "permutés ensemble, car les indicatrices de calendrier sont "
        "imbriquées : toute fin de trimestre est une fin de mois et toute fin "
        "d'année une fin de trimestre, de sorte qu'un par un ils se "
        "compensent. Les quatre variables de bilan n'apportent rien, même "
        "conjointement, à −0,005 pb. Une valeur négative signifie que la "
        "permutation a amélioré la prévision par hasard, ce qui relève du "
        "bruit."
    ),

    "s4": "4. Cette capacité prédictive a-t-elle une valeur économique ? Non",
    "s4_intro": (
        "La trésorerie au jour le jour du lendemain est placée soit en repo au "
        "SOFR, soit auprès de la facilité au taux servi, sur décision prise la "
        "veille. Le portage se décompose exactement en "
        "`base + déviation + corridor`, et la base comme le corridor sont "
        "connus un jour à l'avance. La référence pertinente est donc une règle "
        "qui ne dispose que de la partie connue, et non le prêt systématique "
        "passif, qui attribuerait au modèle le mérite du corridor."
    ),
    "col_rule": "règle",
    "col_net": "pb nets",
    "col_vs_always": "écart au prêt systématique",
    "col_vs_known": "écart à la partie connue",
    "rule_always": "prêt systématique",
    "rule_known": "partie connue seule, sans modèle",
    "rule_ar1": "AR(1)",
    "rule_lasso": "LASSO en deux étapes",
    "rule_perfect": "prévision parfaite",
    "conclusion": (
        "L'opportunité totale représente **0,303 pb** de rendement annualisé. "
        "Une règle sans modèle en capte les deux tiers, et l'avantage du LASSO "
        "sur un modèle autorégressif est de 0,003 pb, c'est-à-dire rien. Avec "
        "un coût de bascule de 1 pb, le LASSO passe derrière l'AR(1), +0,155 "
        "contre +0,186. Les deux facilités de la Fed encadrent le corridor si "
        "étroitement que la décision de placement au jour le jour ne recèle "
        "presque aucune valeur. En 2022, le prêt systématique cédait 1,55 pb "
        "par rapport à la facilité et toutes les règles plus élaborées ne font "
        f"que revenir à zéro. C'est pourquoi près de 2{NNBSP}000 milliards de "
        "dollars étaient placés à l'ON RRP : non pour rechercher du portage, "
        "mais simplement parce que la facilité rémunérait davantage."
    ),
    "sources": (
        "Sources : SOFR, EFFR avec la fourchette cible du FOMC, et opérations "
        "ON RRP issues de l'API des marchés de la Federal Reserve Bank of New "
        "York. Treasury General Account (TGA) issu de l'API Fiscal Data du "
        "Trésor américain. Écarts des taux administrés tirés des notes de mise "
        "en œuvre de la Réserve fédérale. Réalisé avec pandas et scikit-learn ; "
        "aucune donnée payante. Les limites et les extensions écartées "
        "figurent dans le README du dépôt."
    ),
}


def _fr_typography(s: str) -> str:
    s = re.sub(r" ([:])", NBSP + r"\1", s)
    s = re.sub(r" ([;?!%])", NNBSP + r"\1", s)
    return re.sub(r"(\d) (pb\b)", r"\1" + NBSP + r"\2", s)


def _apply(v, f):
    if isinstance(v, str):
        return f(v)
    if isinstance(v, dict):
        return {k: _apply(x, f) for k, x in v.items()}
    return v


STRINGS: dict[str, dict[str, object]] = {
    "en": EN,
    "fr": {k: (v if k == "months" else _apply(v, _fr_typography))
           for k, v in FR.items()},
}
