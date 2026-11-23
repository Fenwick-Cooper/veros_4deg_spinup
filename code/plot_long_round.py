#!/usr/bin/env python3
"""
Distance from the 2000-year state against model time, for the conventional run and for
the accelerated runs whose states were saved with spinup_from_init.py --save-states
(tags t2000, L1, L2). Writes long_round.png. Needs matplotlib.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401
a = np.load(os.path.join(REFERENCE_DIR, "analysis.npz"))
mask, T2000 = a["mask"], a["T2000"]
n_level = np.maximum(mask.sum(axis=(0, 1)), 1)


def dist(T):
    d = np.where(mask, T - T2000, 0.0)
    return np.sqrt((d**2).sum() / mask.sum()), np.sqrt((d**2).sum(axis=(0, 1)) / n_level)[0]


runs = {
    "t2000": "3 rounds, 41/63/60 pseudo-yr (final recipe)",
    "L1": "3 rounds, last one 2044 pseudo-yr",
    "L2": "4 rounds, last one 2016 pseudo-yr",
}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
years = a["years"]
keep = years > 0
for p, j, title in ((ax[0], 0, "whole ocean"), (ax[1], 1, "bottom level")):
    conv = a["r2000_tot"] if j == 0 else a["r2000_lev"][:, 0]
    p.loglog(years[keep], conv[keep], "k", lw=2, label="conventional (decadal snapshots)")
    for tag, label in runs.items():
        s = np.load(os.path.join(INTERMEDIATE_DIR, "from_init", f"cycles_{tag}_states.npz"))
        names = sorted(s.files, key=lambda n: int(n.split("day")[-1]))
        t = np.array([int(n.split("day")[-1]) for n in names]) / 360.0
        d = np.array([dist(s[n].astype(float))[j] for n in names])
        p.loglog(t, d, "o-", ms=3, label=label)
    p.axhline(0.01, color="grey", ls=":", lw=1)
    p.set_xlabel("model years since the initial state")
    p.set_title(f"RMS $T - T_{{2000}}$, {title}")
    p.set_xlim(0.04, 2000)
    p.set_ylim(0.01, 2)
ax[0].set_ylabel("°C")
ax[0].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "long_round.png"), dpi=150)
print("written long_round.png")
