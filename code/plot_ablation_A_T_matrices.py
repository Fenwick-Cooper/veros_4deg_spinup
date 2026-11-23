#!/usr/bin/env python3
"""
Plot the results of ablation_A_T_matrices.py: the top-layer temperature of the 4-year
Veros state, the full linear solve, and each of the six leave-one-matrix-out linear
solves, each minus the climatological equilibrium (spinup_200_72000.restart.h5).

Run with a plain Python environment that has matplotlib (not necessarily `veros`):
    python plot_ablation_A_T_matrices.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401
IN_NPZ = os.path.join(INTERMEDIATE_DIR, "ablation_A_T_matrices.npz")
OUT_PNG = os.path.join(PLOTS_DIR, "ablation_A_T_matrices.png")

MATRIX_NAMES = ["M", "E", "D33", "F", "D", "R"]
MATRIX_LABELS = {
    "M": "advection",
    "E": "isoneutral (Redi), explicit",
    "D33": "isoneutral $K_{33}$, implicit",
    "F": "GM skew",
    "D": "vertical mixing",
    "R": "surface relaxation",
}


def main():
    data = np.load(IN_NPZ)
    xt, yt = data["xt"], data["yt"]
    T_clim_top = data["T_clim_top"]

    panels = [("Veros, 4 years\n(no linear solve)", data["T_4yr_top"])]
    panels += [
        (f"leave out {name}\n({MATRIX_LABELS[name]})", data[f"T_leaveout_{name}_top"])
        for name in MATRIX_NAMES
        if name != "R"
    ]
    panels += [("leave out D33 and F\n(isoneutral $K_{33}$ + GM skew)", data["T_leaveout_D33_F_top"])]
    panels += [("full linear solve\n(all 6 matrices)", data["T_full_top"])]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    im = None
    for ax, (title, field) in zip(axes.flat, panels):
        diff = field - T_clim_top
        im = ax.pcolormesh(xt, yt, diff.T, cmap="coolwarm", vmin=-2, vmax=2, shading="nearest")
        ax.set_title(title, fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")

    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$T - T_\mathrm{climatology}$ ($^\circ$C)")
    fig.suptitle("Top-layer temperature minus the climatological equilibrium (spinup_200_72000)")
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
