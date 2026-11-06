# Repo spread: what drives it, and can it be traded?

Individual project for Machine Learning with R (emlyon, 1BLAML). Due 31 December 2026.

## Question

Which public factors explain the daily gap between SOFR and the Fed's administered rate (IORB), and does that explanation produce a signal that survives an honest backtest?

## Plan

1. Data: SOFR (NY Fed) and policy, reserve and Treasury balance series (FRED). Public sources only.
2. Explain: LASSO selects the drivers of the spread. Time-ordered splits, no look-ahead.
3. Trade idea: a threshold signal built from the model, backtested with costs, reported as it comes, including a weak result.
4. Object: a Streamlit app on Streamlit Community Cloud, code in this repository.

## Run

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-dev.txt   # requirements.txt alone runs only the app
    python src/fetch_data.py       # download SOFR, EFFR, ON RRP, TGA
    python src/build_dataset.py    # merge into data/processed/daily.csv
    python src/model_lasso.py      # walk-forward LASSO vs AR benchmarks
    python src/model_compare.py    # five stage-two models, including trees
    python src/backtest_carry.py   # the cash placement rule
    streamlit run app.py           # the app

The app reads only the files in `data/processed`, so it starts without refitting anything.

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

So the model has statistical skill, at R² +0.13 against AR(1) from 2024, and that skill is worth essentially nothing here. The reason is structural: the Fed's two facilities bound the corridor so tightly that the overnight placement decision has almost no money in it. The action is confined to 2022 and 2023. In 2022 always-lend gave up 1.55 bp against the facility and every smarter rule merely gets back to zero, which is why about $2.0tn sat at ON RRP: not to chase carry, but because the facility simply paid more. From 2024 the carry is positive every day and all rules collapse onto always-lend. In 2023 the carry is so thin that switching costs take the entire benefit, and even perfect foresight nets less than always-lend.

The honest conclusion is that this is a measurement exercise about where repo prints inside the corridor, not a trading strategy.

## Extensions left out for lack of time

The model forecasts the *size* of the deviation, and that is thrown away by a binary overnight decision. It should matter for the term choice, whether to lend overnight or out to a week or a month, where a few basis points compound over the term. Testing that needs term repo rates, which are not in the free sources used here.

Robust loss, Huber for instance, is the cleaner fix for the 2019 outlier than a fixed winsorising cap, but it sits outside what the course covers.

September 2019 is in the data but not in the backtest. The model needs 500 training days, so its first forecast is 20 May 2020 and the stress episodes of September 2019 and March 2020 fall inside the training window. They are documented as case studies rather than scored as forecasts. Shortening the training window would bring them in, at the cost of fitting the early model on a few hundred days.

## Status

Data downloaded and verified. Dataset builder, walk-forward LASSO and carry backtest all written and run, with assertions for no look-ahead, award-rate consistency and the carry identity. Next: the Streamlit app and the report.

## The administered-rate table

`data/fed_administered_rates.csv` holds one row per Federal Reserve implementation note, 15 of them, each with the effective date, the target range, the IORB (or IOER) rate, the ON RRP offering rate, and the URL of the note it was read from. It exists because the offsets are not stable and cannot be assumed.

IORB sits 0 bp below the top of the target range in early 2018, then 5, then 10, then 15, then 20 bp through late 2019 and early 2020, then back to 10 bp from June 2021. A flat 10 bp assumption, which an earlier version of this project used, is wrong on 713 of the 2,140 days, by up to 10 bp. ON RRP sits at the bottom of the range until September 2019, 5 bp below it to March 2020, at the bottom again, 5 bp above it from 17 June 2021, and back at the bottom from 19 December 2024.

Between notes the last note's offsets are carried forward and applied to that day's actual target range, so a rate change with no note of its own is handled. Two checks run on every build and both must pass: on each note's own effective date the range the note states must equal the range the NY Fed series carries, and the table must reproduce all 491 directly observed award rates. It does.

309 days, from 5 May 2022 to 4 November 2024, rely on an offset carried across a rate change no note in the table covers. That stretch is bounded by verified notes on both sides, at 17 March 2022 and 16 June 2022 before it and 27 July 2023 within it, plus the observed award rates from 5 November 2024.

Note that `spread_hi_bp` is SOFR minus the top of the target range, a published number, not a proxy for IORB. The IORB spread is `spread_iorb_bp`.

FRED times out from the author's network and is optional.

## LLM use

Claude (Anthropic) is used for planning, code drafting and review. Prompts and decisions are logged for the report.
