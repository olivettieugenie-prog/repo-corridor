"""Cash placement rule: overnight repo or the Fed's facility, decided a day ahead.

Run from the project root, after model_lasso.py:
    python src/backtest_carry.py

The decision
    Each day, place tomorrow's overnight cash either in repo, earning SOFR, or
    at the ON RRP facility, earning the award rate. The carry is the difference,
    in basis points of annualised yield.

The decomposition that keeps this honest
    carry = (SOFR - top of range) + (top of range - award rate)
          = spread + corridor
          = base + deviation + corridor
    where base is the spread's trailing 60-day median and corridor is the gap
    between the top of the target range and the award rate. Both base and
    corridor are known the day before: base uses only past spreads, and the
    target range and award rate are administered and announced in advance. So
    most of tomorrow's carry is already known without any model, and only the
    deviation has to be forecast. The script asserts the identity holds.

    That is the trap this file exists to avoid. A rule fed the known part looks
    skilful while contributing nothing, so the model is judged against a rule
    that uses the known part alone, not against passive always-lend.

Benchmarks
    always lend       never use the facility
    known only        decide on base + corridor, no model
    AR(1)             known part plus an autoregressive deviation forecast
    two-stage LASSO   known part plus the model's deviation forecast
    perfect foresight the ceiling, using tomorrow's realised carry
The award rate is rebuilt from the administered-rate table back to 2018, so
it is known on every day (carry_usable is 1 throughout) and the sample is every
out-of-sample date: 1,605 days from 20 May 2020. Costs are charged per switch
and reported at several levels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if "--root" in sys.argv:
    ROOT = Path(sys.argv[sys.argv.index("--root") + 1])
OUT = ROOT / "data" / "processed"

COSTS_BP = (0.0, 0.25, 0.5, 1.0)  # charged on each change of venue


def decide(pred: np.ndarray, cost_bp: float) -> np.ndarray:
    """Lend in repo when the predicted carry clears the switching cost."""
    return (pred > cost_bp).astype(int)


def run_rule(pred: np.ndarray, actual: np.ndarray, cost_bp: float) -> dict:
    lend = decide(pred, cost_bp)
    switches = int(np.sum(np.abs(np.diff(np.concatenate([[1], lend])))))
    gross = np.where(lend == 1, actual, 0.0)
    net = gross.mean() - cost_bp * switches / len(lend)
    return {"mean_bp": gross.mean(), "net_bp": net, "days_lent_pct": 100 * lend.mean(),
            "switches": switches}


def main() -> int:
    daily = pd.read_csv(OUT / "daily.csv", parse_dates=["date"])
    oos = pd.read_csv(OUT / "oos_predictions.csv", parse_dates=["date"])

    cols = ["date", "carry_bp", "base_bp", "target_hi", "award_pct", "carry_usable",
            "spread_hi_bp", "dev_bp"]
    d = oos.drop(columns=[c for c in ("year",) if c in oos.columns]).merge(
        daily[cols], on="date", how="left")
    d = d[d["carry_usable"] == 1].reset_index(drop=True)
    d["year"] = d["date"].dt.year

    # corridor and base are both known the day before the decision bites
    d["corridor_bp"] = (d["target_hi"] - d["award_pct"]) * 100
    d["carry_known_bp"] = d["base_bp"] + d["corridor_bp"]

    # The identity must hold exactly, or the decomposition is wrong somewhere.
    recon = d["carry_known_bp"] + d["dev_bp"]
    err = (recon - d["carry_bp"]).abs().max()
    assert err < 1e-6, f"carry identity broken, max error {err}"
    print(f"carry identity holds (max error {err:.2e} bp)")

    # Forecasts of tomorrow's carry. y and the model columns are deviations.
    d["carry_ar1"] = d["carry_known_bp"] + d["ar1"]
    d["carry_ts"] = d["carry_known_bp"] + d["two_stage"]

    print(f"\n{len(d)} decision days, {d['date'].min().date()} -> {d['date'].max().date()}")
    neg = d["carry_bp"] < 0
    print(f"facility beat repo on {int(neg.sum())} of them ({100 * neg.mean():.1f}%)")
    print(f"always-lend mean carry {d['carry_bp'].mean():.3f} bp, "
          f"perfect foresight {np.maximum(d['carry_bp'], 0).mean():.3f} bp, "
          f"so the whole prize is {np.maximum(d['carry_bp'], 0).mean() - d['carry_bp'].mean():.3f} bp")

    actual = d["carry_bp"].to_numpy(float)
    rules = {
        "always lend": np.full(len(d), 1e9),
        "known only": d["carry_known_bp"].to_numpy(float),
        "AR(1)": d["carry_ar1"].to_numpy(float),
        "two-stage LASSO": d["carry_ts"].to_numpy(float),
        "perfect foresight": actual,
    }

    for cost in COSTS_BP:
        print(f"\ncost {cost:.2f} bp per switch")
        print(f"  {'rule':<20}{'net bp':>9}{'gross bp':>10}{'% lent':>9}{'switches':>10}"
              f"{'vs always':>11}{'vs known':>10}")
        base_net = run_rule(rules["always lend"], actual, cost)["net_bp"]
        known_net = run_rule(rules["known only"], actual, cost)["net_bp"]
        for name, pred in rules.items():
            r = run_rule(pred, actual, cost)
            print(f"  {name:<20}{r['net_bp']:>9.3f}{r['mean_bp']:>10.3f}"
                  f"{r['days_lent_pct']:>9.1f}{r['switches']:>10d}"
                  f"{r['net_bp'] - base_net:>+11.3f}{r['net_bp'] - known_net:>+10.3f}")

    print("\nby year, net bp at 0.25 bp cost:")
    print(f"  {'year':<6}{'n':>5}{'always':>9}{'known':>9}{'AR(1)':>9}{'LASSO':>9}{'perfect':>9}")
    for yr, g in d.groupby("year"):
        a = g["carry_bp"].to_numpy(float)
        row = [run_rule(p, a, 0.25)["net_bp"] for p in
               (np.full(len(g), 1e9), g["carry_known_bp"].to_numpy(float),
                g["carry_ar1"].to_numpy(float), g["carry_ts"].to_numpy(float), a)]
        print(f"  {yr:<6}{len(g):>5}" + "".join(f"{v:>9.2f}" for v in row))

    d.to_csv(OUT / "carry_backtest.csv", index=False)
    print(f"\nwrote {OUT / 'carry_backtest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
