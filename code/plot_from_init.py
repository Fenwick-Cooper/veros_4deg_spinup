#!/usr/bin/env python3
"""
Figures for the accelerated spin-up started from Init_state.restart.h5 (spinup_from_init.py):

  from_init_scan.png        day on which the temperature score first falls below 1e-5,
                            against the warm-up length, for each number of solver steps
                            and both closure choices (read from from_init/summary.txt)
  from_init_timeseries.png  temperature score and RMS distance from the 200-year state
                            against model day, for the runs named on the command line

Needs matplotlib:
    python plot_from_init.py name[=legend text] [name[=legend text] ...]
where each name is a from_init/<name>.csv written by spinup_from_init.py.
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401
OUT_DIR = os.path.join(INTERMEDIATE_DIR, "from_init")
THRESHOLD = 1e-5
BASELINE_DAY = 3714

LINE = re.compile(r"^warmup\s+(\d+)\s+steps\s+(\d+)\s+dt_first (\S+)\s+growth (\S+)\s+first below 1e-5: total day (\S+)(.*)$")


def read_scan():
    """{closures: {steps: [(warmup, crossing day)]}} for the doubling schedule that starts
    at 0.125 years (tags s2: coefficients held, e: energies held)."""
    scan = {"coefficients": {}, "energies": {}}
    with open(os.path.join(OUT_DIR, "summary.txt")) as f:
        for line in f:
            m = LINE.match(line)
            if not m or m.group(5) == "None" or float(m.group(3)) != 0.125 or float(m.group(4)) != 2:
                continue
            tag = line.rsplit("tag ", 1)[1].strip() if "  tag " in line else ""
            if tag not in ("s2", "e"):
                continue
            closures = "energies" if tag == "e" else "coefficients"
            scan[closures].setdefault(int(m.group(2)), []).append((int(m.group(1)), int(m.group(5))))
    return scan


def plot_scan():
    scan = read_scan()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, closures in zip(axes, ("coefficients", "energies")):
        for steps in sorted(scan[closures]):
            pts = sorted(scan[closures][steps])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"{steps} steps ({0.125 * (2**steps - 1):.0f} pseudo-years)")
        ax.plot([0, 130], [0, 130], color="0.6", ls=":", lw=1)
        ax.set_title(f"solve holds the turbulence {closures} fixed" if closures == "energies" else "solve holds the mixing coefficients fixed", fontsize=10)
        ax.set_xlabel("warm-up before the solve [days]")
        ax.set_xlim(0, 130)
        ax.set_ylim(0, 1300)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("day the temperature score first falls below $10^{-5}$")
    axes[0].legend(fontsize=8, title=f"conventional Veros: day {BASELINE_DAY}", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(DESCRIPTION_DIR, "from_init_scan.png"), dpi=150)


def plot_timeseries(args):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for n, arg in enumerate(args):
        name, _, text = arg.partition("=")
        d = np.genfromtxt(os.path.join(OUT_DIR, f"{name}.csv"), delimiter=",", names=True)
        style = dict(color="0.55", lw=2.5) if n == 0 else dict(lw=1.2)
        (line,) = ax1.semilogy(d["total_day"], d["score_temp"], label=text or name, **style)
        ax2.plot(d["total_day"], d["rms_total"], color=line.get_color(), lw=style["lw"], label=text or name)
    ax1.axhline(THRESHOLD, color="k", ls="--", lw=0.8)
    ax1.set_ylabel("temperature score (one model day)")
    ax1.set_ylim(1e-6, 1e-2)
    ax1.set_xscale("log")
    ax1.set_xlim(1, 7200)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    ax2.set_ylabel("RMS(T - T at 200 years)  [degC]")
    ax2.set_xlabel("model day, counted from Init_state.restart.h5")
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(DESCRIPTION_DIR, "from_init_timeseries.png"), dpi=150)


if __name__ == "__main__":
    plot_scan()
    if len(sys.argv) > 1:
        plot_timeseries(sys.argv[1:])
