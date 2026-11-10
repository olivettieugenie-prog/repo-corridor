# Repo spread: what drives it, and can it be traded?

Individual project for Machine Learning with R (emlyon, 1BLAML).

## Question

Which public factors explain the daily gap between SOFR and the Fed's administered rate (IORB), and does that explanation produce a signal that survives an honest backtest?

## Plan

1. Data: SOFR, EFFR with the target range, ON RRP and Standing Repo Facility operations from the New York Fed; the Treasury General Account from the Treasury's Daily Treasury Statement (Fiscal Data API); administered rates from Federal Reserve implementation notes. Public sources only, no key needed. FRED is optional and skipped when it is unreachable.
2. Explain: LASSO selects the drivers of the spread. Time-ordered splits, no look-ahead.
3. Use: a cash placement rule backtested with costs, then a desk view for a repo trader: the next day's call, the next turns and a term-versus-roll calculator. Results reported as they come, including weak ones.
4. Object: a Streamlit app on Streamlit Community Cloud, code in this repository.

## Run

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-dev.txt   # requirements.txt alone runs only the app
    python src/fetch_data.py       # download SOFR, EFFR, ON RRP, TGA
    python src/build_dataset.py    # merge into data/processed/daily.csv
    python src/model_lasso.py      # walk-forward LASSO vs AR benchmarks
    python src/model_compare.py    # five stage-two models, including trees
    python src/dm_test.py          # Diebold-Mariano test against AR(1)
    python src/intervals.py        # 80% intervals from past errors only
    python src/turn_quantiles.py   # turn-day quantiles, scored against baselines
    python src/fetch_srf.py        # Standing Repo Facility take-up, rate check
    python src/backtest_carry.py   # the cash placement rule
    streamlit run app.py           # the app

The app reads only the files in `data/processed`, so it starts without refitting anything.

## Design and language

The active design is **tempered**: near-black dark mode, deep blue light mode, and Roboto for headings and body text. `src/design.py` holds it alongside three other validated designs (marine, institutional, editorial). To change design, edit `ACTIVE` there (or set `REPO_DESIGN`), then regenerate the Streamlit theme:

    python src/design.py tempered   # writes .streamlit/config.toml

The config declares both light and dark palettes and loads the fonts through `theme.font` and `theme.headingFont`, so the viewer switches night mode through Streamlit's own Settings menu and the charts follow.

The page switches between English and French from the control at the top right; English is the default and the choice is kept in `st.session_state["lang"]`. Every user-facing string lives in `src/i18n.py` as `STRINGS["en"]` and `STRINGS["fr"]`, with the same keys in both. To add or reword text, edit both entries there and refer to the key from `app.py` as `T["key"]`. Numbers go through `num()`, which gives `12,345.68` in English and `12 345,68` in French (narrow no-break space, decimal comma). Altair's axis tick labels stay in Vega's default English format, because `st.altair_chart` has no locale setting; axis titles, tooltips and labels inside the charts are translated.

    python -m unittest discover -s tests -v   # same keys in both languages; app runs in EN/FR, light/dark

## Live forecast

`src/forecast_next.py` downloads every required series again, rebuilds the daily table in a scratch directory (`.live_work/`, so the committed files are untouched), fits the same two-stage model on the latest 500 days and forecasts the deviation for the next settlement day, with the 80% interval from the last 250 walk-forward errors. Each run appends one row to `data/live/forecast.csv`, and fills in the realised value of every earlier row whose target date has since been published. `tests/test_live.py` checks that the live feature row, model fit and SOFR mapping reproduce `model_lasso.py` and `build_dataset.py` exactly.

    REPO_SKIP_FRED=1 python src/forecast_next.py

`.github/workflows/daily.yml` runs it at 14:30 UTC on weekdays, with a backup run at 15:30 UTC because GitHub can start scheduled runs hours late (and by hand through workflow_dispatch) and commits the log to the `data` branch, so main's history carries no daily commits. The app reads that file from GitHub, hourly, and falls back to the local copy.

What exists when the job runs, on settlement day D at 9:30 a.m. ET (winter) or 10:30 a.m. ET (summer):

| series | source | published | latest at run time | used by the model as |
|---|---|---|---|---|
| SOFR, percentiles, volume | NY Fed | about 8:00 a.m. ET, next business day; revisable about 2:30 p.m. | D−1 | day t = D−1 |
| EFFR and target range | NY Fed | about 9:00 a.m. ET, next business day; revisable about 2:30 p.m. | D−1 | day t = D−1 |
| ON RRP take-up and award rate | NY Fed | same day, after the 1:15 p.m. ET close | D−1 | t−1 = D−2 |
| TGA closing balance | Daily Treasury Statement | 4:00 p.m. ET, next business day | D−2 | t−1 = D−2 |
| Calendar flags | rules | known in advance | D | target day D |
| Target range on D | implementation-note table | FOMC decision 2:00 p.m. ET, effective next day | needs a table row | `range_unconfirmed` = 1 if missing after an FOMC meeting |

So the forecast for D is available only on the morning of D: the model needs SOFR for D−1, which the NY Fed publishes at 8:00 a.m. on D. It is a same-day nowcast of the rate that D's trades will set, not a decision made the evening before.

The desk view at the top of the app shows the latest run: target date, the deviation forecast with its 80% range, the implied SOFR level, the calendar flags in play, ON RRP and TGA with their daily change, and a two-line note filled from a fixed template in `src/i18n.py` (`src/desk.py` holds the logic). The implied level is exact, not approximate: `build_dataset.py` defines the deviation as SOFR minus the top of the target range minus the trailing 60-day median of that spread, all known the day before, so SOFR = top of range + (median + deviation) / 100, which `tests/test_live.py` checks on every day of the data. The panel warns when the forecast targets a day more than one business day back, and shows the hit rate and RMSE against AR(1) once 20 target days have been scored; until then it shows when the live record began.

Business days follow the US government securities settlement calendar in `src/market_calendar.py`. It matches every SOFR publication date since April 2018. Against pandas' `USFederalHolidayCalendar` it adds Good Friday, keeps the Friday before a Saturday New Year's Day or Veterans Day open, starts Juneteenth in 2022, and adds the 5 December 2018 closure.

## Turn quantiles

The desk question is what the next month-end, quarter-end or year-end will cost. `src/turn_quantiles.py` forecasts SOFR minus the top of the target range on turn days at the 50th, 90th and 95th percentiles, with linear quantile regression (scikit-learn `QuantileRegressor`, solver `highs`, no penalty) on a quarter-end flag and past-only scarcity state: lagged ON RRP, lagged TGA and the trailing 60-day median spread. Each turn is forecast walk-forward from a fit on earlier turns only. Year-ends, 8 in the data, get no model.

Against the empirical quantiles of earlier turns of the same type, the model wins at every quantile for both turn types (paired pinball-loss tests, p ≤ 0.05; at most 0.001 except the month-end 95th percentile). Against the pooled 80% interval method mapped to the same quantiles it loses on the month-end median (p = 0.015) and is level otherwise. Fitting on the last 500 days only is worse everywhere. Scores in `data/processed/turn_quantile_scores.csv`.

The app forecasts turns weeks ahead, so the model is re-scored with the state taken 1, 5, 10 and 15 settlement days before each turn, then calibrated by split-conformal shifts: each quantile moves by the matching empirical quantile of earlier out-of-sample errors at the same horizon, using only turns already printed. On the 40 month-ends from October 2021:

| horizon | 90th pct coverage, raw | calibrated | 95th pct coverage, raw | calibrated |
|---|---|---|---|---|
| 1 day | 75% | 97.5% | 82.5% | 97.5% |
| 5 days | 80% | 95% | 87.5% | 97.5% |
| 10 days | 77.5% | 95% | 85% | 97.5% |
| 15 days | 72.5% | 90% | 77.5% | 97.5% |

Calibration removes the under-coverage of the upper quantiles, and overshoots a little; pinball loss improves at some horizons and worsens at others, because a wider upper quantile costs loss on calm turns. Quarter-ends (16) were already covered at the upper quantiles. The median stays biased low: even after calibration, month-ends printed above it 52% to 65% of the time, depending on horizon. A finite-sample (n + 1) version of the shift is scored too and is far more conservative. Full table, by type, horizon and version, in `data/processed/turn_quantile_horizons.csv`; coefficients and current shifts per horizon in `turn_quantile_coefs_horizon.csv`. The app uses the calibrated quantiles at the horizon closest to the settlement days left until each turn (15 days for anything further out, since no longer horizon is scored), in the Next turns panel and for turn days in the calculator, and shows that horizon's walk-forward coverage beside each forecast.

## Analog table

`src/analogs.py` finds the 8 past turns of the same type whose scarcity state (lagged ON RRP, lagged TGA, trailing median spread) is closest to today's, by Euclidean distance after standardising each variable on data before the query date only. For each it shows what SOFR minus the top of the range printed and the 99th percentile of that day's SOFR trades. Year-ends are included and labelled history only. In the app each upcoming turn expands to its analog table.

## Term-versus-roll calculator

`src/roll_cost.py` prices rolling overnight across a window against a term repo quote. Each settlement day from `src/market_calendar.py` accrues to the next one, so a Friday carries the weekend, act/360 on both legs. Ordinary days follow the two-stage forecast and its AR(1) decay toward the trailing median; month- and quarter-ends use the turn quantiles. The app shows the expected roll, a stress case and the gap to the quote in bp, with a verdict from fixed templates. The stress case puts every day at its own 90th percentile at once, as if every day were bad together, so it is an upper reference rather than the 90th percentile of the average.

A window touching an FOMC decision or its effective day is not priced, because the cost depends on a policy path; meeting dates come from `data/fomc_meetings.csv`, sourced to the Fed's calendar. A window containing a year-end shows every past year-end print with its min, median and max, labelled history, not a forecast, prices the rest of the window, and reports the year-end print at which the quote would break even. `tests/test_roll.py` covers the day count, the averaging, turn days and these cases.

## Standing Repo Facility data

`data/srf_rates.csv` lists the Standing Repo Facility rate from all 43 FOMC implementation notes since the facility opened on 28 July 2021, each with its URL: a minimum bid rate through October 2025 and a fixed full-allotment rate from December 2025, always equal to the top of the target range. `src/fetch_srf.py` checks that equality on every effective date and writes daily take-up from the New York Fed markets API (`/api/rp/results/search.json`, repo operations from 29 July 2021) to `data/processed/srf_takeup.csv`. The API's range query carries no rate, which is why the notes are the source. The corridor chart draws the rate as a dashed line with a take-up panel below.

## Data

SOFR, EFFR with the Fed target range, and ON RRP come from the NY Fed API. The Treasury General Account (TGA) closing balance comes from the Treasury Fiscal Data API (Daily Treasury Statement), which changed its row labels twice, in October 2021 and April 2022. FRED timed out from the author's network and is optional.

`build_dataset.py` writes `data/processed/daily.csv`, one row per SOFR date from 2018-04-02 to 2026-10-26. Balance-sheet variables are lagged by one observation so that no feature uses information from the day it explains. The script asserts this.

## What the data decided

Three findings shaped the design, each from the data rather than from a prior.

A stress classifier is not viable. SOFR printed above the top of the target range on 62 of 2,140 days, above +10 bp on 16, above +25 bp on 7. There are too few positive cases to fit or validate a classifier.

One day dominates a least-squares fit. 17 September 2019 is 46% of the total squared deviation of the raw spread; the top five days are 53%. The modelling target is therefore the deviation from a trailing 60-day median, winsorised at a fixed ±25 bp (9 rows capped). The cap is stated, not fitted from sample percentiles, so it leaks nothing into the backtest.

The level drifts by regime, and the drift is the economics, not the signal. Yearly means of the raw spread run from -4.5 bp to -21.6 bp as ON RRP fell from about $2.0tn in 2022 to about $2bn in 2026. Subtracting the trailing median collapses yearly means to between -1.1 and +2.2 bp and drops 20-day autocorrelation from 0.39 to 0.10. The slow component is the regime to condition on; the fast component is what the model explains.

## What is being decided

A cash investor chooses each day between lending overnight in repo at SOFR and placing at the Fed's ON RRP facility at the award rate. The carry is the difference.

The carry is now known for all 2,140 days, and its shape across the reserve cycle is the economic spine of the project. Median carry by year: +20 bp in 2018, +15 bp in 2019, +9 bp in 2020, 0 bp in 2021, −1 bp in 2022 with 58.6% of days negative, +1 bp in 2023, +2 bp in 2024, +10 bp in 2025, +14 bp in 2026. High when reserves were scarce, zero or negative when the system was awash, high again as the facility drained.

The extremes are in there too. On 17 September 2019 SOFR printed 5.25% against an ON RRP offering rate of 2.00%, so lending overnight in repo paid 325 bp over the facility. That is what the corridor failing looks like.

That tracks a policy change exactly. From 17 June 2021 the Fed set the ON RRP rate 5 bp above the bottom of the target range, making the facility deliberately attractive, and usage rose to about $2.0tn. Effective 19 December 2024 it was realigned to the bottom of the range, the repo carry turned positive again, and usage fell to about $2bn. The spread against IORB moves the same way, from -11.5 bp in 2022 to -1.4 bp in 2026.

So the model predicts the carry, and the rule allocates between overnight repo and the facility. The benchmark is passive always-lend, which is a hard bar outside 2022.

## Result so far

The target is tomorrow's deviation of the spread from its own trailing 60-day median, winsorised at a fixed ±25 bp. Calendar flags are taken at t+1, because the calendar is known in advance. Everything measured from the market or the Fed's balance sheet is taken at t or earlier.

Zero is a weak benchmark and is not quoted alone. The trailing median removes the level of the spread but not a slow trend in it, so the deviation stays strongly autocorrelated, at 0.81 in the test window. An autoregression alone reaches an out-of-sample R² of 0.64 against zero. The only question worth asking is whether anything adds to it.

Two specification choices decide the answer, and both are reported rather than tuned away.

A single penalty across all coefficients loses to AR(1), at R² −0.07 against it, because the shared penalty shrinks the dominant autoregressive term in order to control the weak ones. Fitting the autoregression first and letting the LASSO explain its residual turns that into +0.04.

An expanding training window also loses, at −0.11 against AR(1), because the calendar effects are regime-dependent. The month-end premium is +9.6 bp in 2018 and +13.3 bp in 2019, essentially zero from 2020 to 2022 (+0.05, +0.07, +0.01), then +2.0, +3.5, +8.7 and +3.6 bp from 2023 to 2026. A model that weights every regime equally fits none of them. With a rolling 500-day window the two-stage model beats AR(1) by +0.04 overall and +0.13 from 2024 onwards, and by year it is +0.07 in 2024, +0.17 in 2025 and +0.08 in 2026.

**Is the gain significant? From 2024, yes. Over the full sample, no.** A Diebold-Mariano test on squared errors (`src/dm_test.py`: Newey-West HAC variance with automatic lag, Harvey-Leybourne-Newbold correction, Student t critical values) gives −3.71 from 2024 onwards, p = 0.0002. 2024 and 2025 are significant on their own (p = 0.008 and 0.005); 2026 to date is not (p = 0.22). Over the whole out-of-sample period from May 2020 the +0.04 is **not** significant (p = 0.26), because the model is significantly worse than AR(1) in 2020 (p = 0.0005). In 2022 the second stage selected nothing, so the forecasts equal AR(1) and the test is undefined. The full table is in `data/processed/dm_test.csv`.

**Prediction intervals.** `src/intervals.py` gives every out-of-sample forecast an 80% interval from the 10th and 90th percentiles of the previous 250 forecast errors, using only errors whose target date came before the forecast; a test that perturbs every later outcome checks there is no look-ahead. From May 2021 the interval covers 78.5% of outcomes overall and 79.5% outside month-ends, but only 58.5% on month-ends (65 days) and 36% on quarter-ends (22 days). Pooled errors understate month-end risk. Table in `data/processed/interval_coverage.csv`; bounds are the `lo80` and `hi80` columns of `oos_predictions.csv`.

The economics behind that is the point of the project. Calendar pressure only bites when reserves are scarce. In 2020 to 2022, with ON RRP usage around $2.0tn, month-end balance-sheet constraints had nothing to push against and the premium disappeared. As reserves drained it returned. The model selects month-end, mid-month settlement, quarter-end and month-start on the residual; the balance-sheet levels are selected rarely and with small coefficients.

## Testing the interaction claim: four rival specifications

The finding above is that calendar pressure only moves the spread when reserves are scarce. That is an interaction, and a linear model on calendar dummies cannot represent it: it learns one average month-end coefficient and applies it in every regime. So the claim is testable. Give a model the interaction explicitly, or give it a model class that finds interactions on its own, and it should do better.

`src/model_compare.py` runs five stage-two specifications through the identical two-stage structure, rolling 500-day window and AR(1) benchmark, so only the model differs. The interaction block is the calendar dummies multiplied by two past-only scarcity measures: the spread's own trailing median, and log ON RRP usage. Tree hyperparameters are fixed and conservative rather than tuned, because 500 rows of a noisy target is where tuning invents performance; the linear models get a cross-validated penalty, which if anything favours them.

| stage-two model | R² vs AR(1), all | 2024+ | RMSE |
|---|---|---|---|
| linear, main effects | **+0.044** | **+0.129** | **1.961** |
| linear + interactions | +0.006 | +0.085 | 1.999 |
| ridge + interactions | −0.319 | −0.020 | 2.303 |
| random forest | −0.093 | −0.070 | 2.096 |
| gradient boosting | −0.415 | −0.134 | 2.385 |

The hypothesis fails. Nothing beats plain main effects, and the more capacity a specification has, the worse it does.

The reading is that the rolling window is already doing the work. It only ever sees one regime at a time and relearns the local month-end coefficient at every refit, so an explicit interaction adds parameters without adding information. On a target this noisy that costs more than it returns. The regime-dependence is real, and it is handled by letting the model forget rather than by modelling it.

One result runs the other way and is worth keeping. In 2020, the regime break itself, the random forest scores −0.087 against the linear model's −1.832, a factor of twenty. A tree does not extrapolate outside the range it was trained on, so it degrades gently when the world moves; a linear model fitted to the old regime extrapolates confidently and is badly wrong. That is an argument for trees in a break and against them the rest of the time.

As a check, this script reproduces `model_lasso.py`'s headline numbers by an independent code path.

## Does the skill make money? No, and that is the finding

The rule places tomorrow's overnight cash either in repo at SOFR or at the ON RRP facility at the award rate, decided the day before, on the 1,336 days where the award rate is known.

The carry decomposes exactly, and the script asserts it: carry = base + deviation + corridor, where base is the spread's trailing median and corridor is the gap between the top of the target range and the award rate. Base and corridor are both known the day before. So most of tomorrow's carry needs no model at all, and a rule given only the known part is the benchmark that matters. Judging against passive always-lend alone would credit the model for the corridor.

Over the 1,605 decision days the model covers, in basis points of annualised yield on overnight cash:

| rule | net bp | vs always-lend | vs known-only |
|---|---|---|---|
| always lend | 5.034 | — | −0.167 |
| known only, no model | 5.201 | +0.167 | — |
| AR(1) | 5.270 | +0.237 | +0.070 |
| two-stage LASSO | 5.273 | +0.239 | +0.072 |
| perfect foresight | 5.286 | +0.252 | +0.085 |

The whole opportunity, with tomorrow's carry known in advance, is 0.252 bp. A rule with no model captures two-thirds of it. The LASSO's margin over AR(1) is 0.003 bp, which is nothing.

This is the result after extending the administered-rate history back to 2018, which added 269 decision days. It did not move the conclusion, which is the point of having run it.

So the model has statistical skill, at R² +0.13 against AR(1) from 2024 (Diebold-Mariano p = 0.0002, though not significant over the full sample from 2020), and that skill is worth essentially nothing here. The reason is structural: the Fed's two facilities bound the corridor so tightly that the overnight placement decision has almost no money in it. The action is confined to 2022 and 2023. In 2022 always-lend gave up 1.55 bp against the facility and every smarter rule merely gets back to zero, which is why about $2.0tn sat at ON RRP: not to chase carry, but because the facility simply paid more. From 2024 the carry is positive every day and all rules collapse onto always-lend. In 2023 the carry is so thin that switching costs take the entire benefit, and even perfect foresight nets less than always-lend.

The honest conclusion is that this is a measurement exercise about where repo prints inside the corridor, not a trading strategy.

## Extensions left out for lack of time

The model forecasts the *size* of the deviation, and that is thrown away by a binary overnight decision. It should matter for the term choice, whether to lend overnight or out to a week or a month, where a few basis points compound over the term. Testing that needs term repo rates, which are not in the free sources used here.

Robust loss, Huber for instance, is the cleaner fix for the 2019 outlier than a fixed winsorising cap, but it sits outside what the course covers.

September 2019 is in the data but not in the backtest. The model needs 500 training days, so its first forecast is 20 May 2020 and the stress episodes of September 2019 and March 2020 fall inside the training window. They are documented as case studies rather than scored as forecasts. Shortening the training window would bring them in, at the cost of fitting the early model on a few hundred days.

## Status

Done: data pipeline with no-look-ahead assertions; walk-forward two-stage LASSO with a Diebold-Mariano test against AR(1); carry backtest; 80% prediction intervals; rival specifications and permutation importance; the daily live forecast committed to the `data` branch by GitHub Actions; and the Streamlit app in English and French with the desk view, next turns, the term-versus-roll calculator and the Standing Repo Facility. The test suite runs the app through AppTest in both languages and themes.

Next: the written report (1 to 3 pages, per the project specification).

## The administered-rate table

`data/fed_administered_rates.csv` holds one row per Federal Reserve implementation note, 15 of them, each with the effective date, the target range, the IORB (or IOER) rate, the ON RRP offering rate, and the URL of the note it was read from. It exists because the offsets are not stable and cannot be assumed.

IORB sits 0 bp below the top of the target range in early 2018, then 5, then 10, then 15, then 20 bp through late 2019 and early 2020, then back to 10 bp from June 2021. A flat 10 bp assumption, which an earlier version of this project used, is wrong on 713 of the 2,140 days, by up to 10 bp. ON RRP sits at the bottom of the range until September 2019, 5 bp below it to March 2020, at the bottom again, 5 bp above it from 17 June 2021, and back at the bottom from 19 December 2024.

Between notes the last note's offsets are carried forward and applied to that day's actual target range, so a rate change with no note of its own is handled. Two checks run on every build and both must pass: on each note's own effective date the range the note states must equal the range the NY Fed series carries, and the table must reproduce all 491 directly observed award rates. It does.

309 days, from 5 May 2022 to 4 November 2024, rely on an offset carried across a rate change no note in the table covers. That stretch is bounded by verified notes on both sides, at 17 March 2022 and 16 June 2022 before it and 27 July 2023 within it, plus the observed award rates from 5 November 2024.

Note that `spread_hi_bp` is SOFR minus the top of the target range, a published number, not a proxy for IORB. The IORB spread is `spread_iorb_bp`.

FRED times out from the author's network and is optional.

## LLM use

Claude (Anthropic) is used for planning, code drafting and review. Prompts and decisions are logged for the report.
