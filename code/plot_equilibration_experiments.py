#!/usr/bin/env python3
"""
The three figures of Description/equilibration_experiments.tex, from the daily scores written
to data/intermediate/ by

    score_timeseries_from_init_state.py      score_timeseries.csv (direct integration, baseline)
    equilibration_experiment_fixed.py exp1   equilibration_exp1_fixed.csv
    equilibration_experiment_fixed.py exp2   equilibration_exp2_fixed.csv
    jumpstart_variable_scores_dtemp.py       jumpstart_variable_scores_dtemp.csv
    jumpstart_clamped_wrong_temp_fixed.py    jumpstart_clamped_wrong_temp_fixed.csv

Writes Description/equilibration_exp1_exp2_combined.png, jumpstart_variable_scores_dtemp.png
and jumpstart_clamped_wrong_temp_fixed.png. A figure whose CSV files are missing is skipped.
Needs matplotlib.

    python code/plot_equilibration_experiments.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INTERMEDIATE_DIR  # noqa: E402

THRESHOLD = 1e-5
WARMUP_DAYS = 1440  # the 4-year warm-up from which the jump-started runs restart
COLOURS = {"u": "tab:blue", "v": "tab:orange", "tke": "tab:green", "eke": "tab:purple", "temp": "tab:red"}


def load(name):
    path = os.path.join(INTERMEDIATE_DIR, name)
    if not os.path.exists(path):
        print(f"missing {path}")
        return None
    return np.genfromtxt(path, delimiter=",", names=True)


def finish(ax, xlabel, title, filename, legend_loc):
    ax.axhline(THRESHOLD, color="k", ls=":", lw=1, label="$10^{-5}$ threshold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=legend_loc, ncol=2, fontsize=9)
    ax.figure.tight_layout()
    path = os.path.join(DESCRIPTION_DIR, filename)
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
    print(f"written {path}")


def experiments_1_and_2(base):
    exp1, exp2 = load("equilibration_exp1_fixed.csv"), load("equilibration_exp2_fixed.csv")
    if exp1 is None or exp2 is None or base is None:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for v in ("u", "v", "tke", "eke"):
        ax.plot(exp1["day"], exp1[f"score_{v}"], color=COLOURS[v], lw=1.2, label=f"{v} (Exp.1, temperature fixed)")
    ax.plot(exp2["day"], exp2["score_temp"], color=COLOURS["temp"], lw=2, label="temperature (Exp.2, dynamics fixed)")
    ax.plot(base["day"], base["score_unweighted"], color=COLOURS["temp"], lw=2, ls="--", label="temperature (direct integration, baseline)")
    ax.axvline(WARMUP_DAYS, color="dimgrey", lw=1.5, label=f"4-year warm-up point (day {WARMUP_DAYS})")
    finish(ax, "model day (log scale)", "Experiments 1 and 2: temperature vs. dynamics equilibration", "equilibration_exp1_exp2_combined.png", "lower left")


def baseline_after_warmup(base, days):
    """The baseline temperature score on the same model days as a run restarted from day 1440."""
    lookup = dict(zip(base["day"].astype(int), base["score_unweighted"]))
    keep = [d for d in days.astype(int) if WARMUP_DAYS + d in lookup]
    return np.array(keep), np.array([lookup[WARMUP_DAYS + d] for d in keep])


def jump_start(base):
    run = load("jumpstart_variable_scores_dtemp.csv")
    if run is None or base is None:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    days = run["day_after_restart"]
    for v in ("u", "v", "tke", "eke"):
        ax.plot(days, run[f"score_{v}"], color=COLOURS[v], lw=1.2, label=v)
    ax.plot(days, run["score_temp"], color=COLOURS["temp"], lw=2, label="temp")
    d, s = baseline_after_warmup(base, days)
    ax.plot(d, s, color=COLOURS["temp"], lw=1.5, ls="--", label="temperature (direct integration, baseline)")
    finish(ax, "model day after restart (log scale)", "Jump-started run (dtemp patched): all variables free after the 4-year linear-solve restart", "jumpstart_variable_scores_dtemp.png", "upper right")


def clamped_to_wrong_temperature():
    run = load("jumpstart_clamped_wrong_temp_fixed.csv")
    if run is None:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for v in ("u", "v", "tke", "eke"):
        ax.plot(run["day_after_restart"], run[f"score_{v}"], color=COLOURS[v], lw=1.2, label=v)
    finish(ax, "model day after restart (log scale)", "Temperature clamped to the (wrong) linear-solve estimate", "jumpstart_clamped_wrong_temp_fixed.png", "upper right")


if __name__ == "__main__":
    baseline = load("score_timeseries.csv")
    experiments_1_and_2(baseline)
    jump_start(baseline)
    clamped_to_wrong_temperature()
