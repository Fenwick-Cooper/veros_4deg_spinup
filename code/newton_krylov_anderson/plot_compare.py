#!/usr/bin/env python3
"""
Distance to T_2000 against Veros days integrated (left) and wall-clock time (right), for every
method in summary.jsonl and for conventional Veros (conventional_curve.npz, from
conventional_curve.py), both in RESULTS_DIR (data/intermediate/newton_krylov_anderson/). All wall-clock times are full process
times: each method's harness time plus its start-up (startup_s.json, from startup_times.py), and for
conventional Veros the measured full time of a 3714-day run (conventional_timing.jsonl,
output off as in the method runs) extended at the measured seconds per day. Writes
data/plots/comparison.png, or with --description the figure of Description/method_comparison.tex
into Description/. Needs matplotlib, not Veros.

    python code/newton_krylov_anderson/plot_compare.py
    python code/newton_krylov_anderson/plot_compare.py --exclude C_k0 C_k5f --out comparison_without_C.png --description
"""

import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nka_paths import DESCRIPTION_DIR, PLOTS_DIR  # noqa: E402
from nka_paths import RESULTS_DIR as RESULTS  # noqa: E402


def conventional_wall(days):
    """Full wall-clock time of conventional Veros for `days` days: measured start-up plus days
    times the measured seconds per day (both with Veros's output off, as in the method runs)."""
    speed = json.load(open(os.path.join(RESULTS, "veros_speed.json")))["seconds_per_day"]
    runs = [json.loads(line) for line in open(os.path.join(RESULTS, "conventional_timing.jsonl"))]
    r = [r for r in runs if not r["output"]][-1]
    startup = r["total_s"] - speed * r["days"]
    return startup + speed * np.asarray(days, dtype=float)

# label, colour, line style, line width. L3, D_plain and D_aa3 are the same run up to day 40 and
# D_plain and D_aa3 agree to 0.0005 degC throughout, so they are drawn as stacked widths (widest
# underneath) and D_aa3 dashed, so that all three stay visible.
STYLE = {
    "L3": ("Leap-and-settle, L3", "tab:blue", "-", 2.6),
    "L2": ("Leap-and-settle, L2", "tab:blue", "--", 1.3),
    "D_plain": ("Leap-and-settle, L3 + 3 long rounds", "tab:cyan", "-", 4.5),
    "D_aa3": ("Same, Anderson over rounds", "tab:purple", (0, (3, 2)), 1.3),
    "C_k0": ("Leap-and-settle, Veros residual (defect corr.)", "tab:green", "-", 1.3),
    "C_k5f": ("Leap-and-settle, Veros residual (NK, 5 its)", "tab:green", "--", 1.3),
    "A_m10": ("Anderson, Veros alone", "tab:orange", "-", 1.3),
    "B_k20": ("Newton-Krylov, Veros alone", "tab:red", "-", 1.3),
}


def load(name):
    rows = list(csv.DictReader(open(os.path.join(RESULTS, f"{name}.csv"))))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("wall", "days_total", "rms_T2000")}


def main(args):
    startup = json.load(open(os.path.join(RESULTS, "startup_s.json")))["startup_s"]
    summ = [json.loads(line) for line in open(os.path.join(RESULTS, "summary.jsonl"))]
    summ = [r for r in summ if r["name"] not in args.exclude]
    names = [r["name"] for r in summ]
    crit = {r["name"]: r for r in summ}
    conv = np.load(os.path.join(RESULTS, "conventional_curve.npz"))
    keep = conv["days"] <= 1500 * 360  # beyond this the reference approaches T_2000 itself
    days_ref, rms_ref = np.maximum(conv["days"][keep], 1), conv["rms"][keep]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, key, xlabel in ((axes[0], "days_total", "Veros days integrated"), (axes[1], "wall", "wall-clock time (s)")):
        x_ref = conventional_wall(days_ref) if key == "wall" else days_ref
        ax.plot(x_ref, rms_ref, color="k", lw=2, label="conventional Veros")
        ax.plot(conventional_wall(3714) if key == "wall" else 3714, np.interp(3714, days_ref, rms_ref), "o", color="k", ms=6)
        for name in names:
            label, colour, ls, lw = STYLE.get(name, (name, "grey", ":", 1.3))
            d = load(name)
            x = d[key] + startup[name] if key == "wall" else np.maximum(d[key], 1)
            ax.plot(x, d["rms_T2000"], color=colour, ls=ls, lw=lw, label=label, zorder=10 - lw,
                    alpha=0.45 if lw > 4 else 1.0)  # fmt: skip
            c = crit[name]
            if c["criterion_days_total"] is not None:
                xc = c["criterion_wall_s"] + startup[name] if key == "wall" else c["criterion_days_total"]
                ax.plot(xc, c["rms_at_criterion"], "o", color=colour, ms=8 if lw > 4 else 5, zorder=12 - lw,
                        mfc="none" if ls == "--" else colour, alpha=0.45 if lw > 4 else 1.0)  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.03, 1.6)
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("RMS(T - T$_{2000}$) (°C)")
    axes[0].set_title("against model days integrated (all runs counted)")
    axes[1].set_title("against wall-clock time")
    axes[1].legend(fontsize=8, loc="lower left", title="dots: competition criterion met", title_fontsize=8)
    fig.tight_layout()
    out_dir = DESCRIPTION_DIR if args.description else PLOTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, args.out)
    fig.savefig(out, dpi=150)
    print(f"written {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exclude", nargs="*", default=[], help="run names to leave out, e.g. C_k0 C_k5f")
    parser.add_argument("--out", default="comparison.png", help="file name (in data/plots/, or Description/ with --description)")
    parser.add_argument("--description", action="store_true", help="write into Description/ (a figure used by the documents)")
    main(parser.parse_args())
