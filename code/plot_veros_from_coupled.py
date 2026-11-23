#!/usr/bin/env python3
"""
Plot the Veros runs restarted from the coupled temperature-velocity solve
(restart_veros_from_coupled_state.py): the temperature score and the RMS distance of the
temperature from the 200-year state, against total model day (the 4-year warm-up included).

Needs matplotlib:
    python plot_veros_from_coupled.py label[=legend text] [label[=legend text] ...]
Writes veros_from_coupled.png.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401
WARMUP_DAYS = 4 * 360
THRESHOLD = 1e-5


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    base = np.genfromtxt(os.path.join(INTERMEDIATE_DIR, "score_timeseries.csv"), delimiter=",", names=True)
    ax1.semilogy(base["day"] / 360, base["score_unweighted"], color="0.6", lw=3, label="Veros from the initial state (baseline)")
    for arg in sys.argv[1:]:
        label, _, text = arg.partition("=")
        d = np.genfromtxt(os.path.join(INTERMEDIATE_DIR, f"veros_from_coupled_{label}.csv"), delimiter=",", names=True)
        year = (WARMUP_DAYS + d["day"]) / 360
        (line,) = ax1.semilogy(year, d["score_temp"], lw=1, label=text or label)
        ax2.plot(year, d["rms_total"], color=line.get_color(), label=text or label)
    ax1.axhline(THRESHOLD, color="k", ls="--", lw=0.8)
    ax1.axvline(WARMUP_DAYS / 360, color="k", ls=":", lw=0.8)
    ax2.axvline(WARMUP_DAYS / 360, color="k", ls=":", lw=0.8)
    ax1.set_ylabel("temperature score (one model day)")
    ax1.set_ylim(1e-6, 1e-3)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("RMS(T - T at 200 years)  [degC]")
    ax2.set_xlabel("model time [years of 360 days], including the 4-year warm-up")
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DESCRIPTION_DIR, "veros_from_coupled.png"), dpi=150)


if __name__ == "__main__":
    main()
