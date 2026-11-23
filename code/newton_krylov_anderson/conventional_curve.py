#!/usr/bin/env python3
"""
RMS(T - T_2000) of conventional Veros from Init_state, for the comparison figure and table:
every 30 days for the first --years years (default 200), from a conventional Veros run made
here, then every 10 years from the 2000-year reference run (analysis.npz, from
code/analyse_2000yr_reference.py). The run here is the same integration as the reference
run (same model, same start), only sampled more often; at years 0, 10 and 100 it agrees with
analysis.npz to about 1e-9 degC, which the script prints as a check.

Writes conventional_curve.npz (days, rms) in RESULTS_DIR (data/intermediate/newton_krylov_anderson/).
Run when nothing else is running if the timings matter; 200 years take about 37 minutes.

    python code/newton_krylov_anderson/conventional_curve.py
    python code/newton_krylov_anderson/conventional_curve.py --years 10   # first decade only, ~2 min
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import DAYS_PER_YEAR, RESULTS_DIR, Run  # noqa: E402  (first: selects the JAX backend)

import numpy as np  # noqa: E402

from nka_paths import REFERENCE_DIR  # noqa: E402

SAMPLE_DAYS = 30


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, default=200, help="years of conventional Veros sampled every 30 days")
    args = ap.parse_args()
    n_days = args.years * DAYS_PER_YEAR
    run = Run("conventional_curve", n_days + 10, log_every=SAMPLE_DAYS)
    days, rms = [0], [run.rms()]
    for _ in range(n_days // SAMPLE_DAYS):
        run.run_main(SAMPLE_DAYS)
        days.append(run.days_main)
        rms.append(run.rms())
    run.close()
    os.remove(os.path.join(RESULTS_DIR, "conventional_curve.csv"))
    days, rms = np.array(days, dtype=float), np.array(rms)

    a = np.load(os.path.join(REFERENCE_DIR, "analysis.npz"))
    check = {int(y): (float(np.interp(y * 360, days, rms)), float(r)) for y, r in zip(a["years"], a["r2000_tot"]) if y in (0, 10, 100) and y * 360 <= days[-1]}
    print("this run vs the 2000-year run (year: here, reference):", check)
    later = a["years"] * 360 > days[-1]
    days = np.concatenate([days, a["years"][later] * 360])
    rms = np.concatenate([rms, a["r2000_tot"][later]])
    np.savez(os.path.join(RESULTS_DIR, "conventional_curve.npz"), days=days, rms=rms)
    print(f"written conventional_curve.npz: {len(days)} points, days {days[0]:.0f}-{days[-1]:.0f}")


if __name__ == "__main__":
    main()
