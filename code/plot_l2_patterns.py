#!/usr/bin/env python3
"""
Horizontal patterns of T - T_2000 left by the final configuration (L2), right after its
last solve (day 80) and after 10 more years of Veros, at the surface, ~935 m, ~3010 m
and the bottom level. Land is NaN (white). Writes Description/l2_patterns.png, the figure
of coupled_spinup.tex. Needs matplotlib, data/intermediate/reference_2000yr/analysis.npz
(analyse_2000yr_reference.py) and the L2 states saved by

    python code/spinup_from_init.py --tag L2 --save-states --post-days 3600 --confirm 100000 \
        --cycles 20:30y:0.125,20:60y:2,20:60y:4,20:2000y:32
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INTERMEDIATE_DIR, REFERENCE_DIR  # noqa: E402

a = np.load(os.path.join(REFERENCE_DIR, "analysis.npz"))
mask, ZT, T2000 = a["mask"], a["ZT"], a["T2000"]
s = np.load(os.path.join(INTERMEDIATE_DIR, "from_init", "cycles_L2_states.npz"))
states = [("after the last solve (day 80)", s["patched_day00080"]), ("after 10 more years of Veros", s["end_day03680"])]
levels = [14, 8, 3, 0]
xt = 2.0 + 4.0 * np.arange(mask.shape[0])
yt = -78.0 + 4.0 * np.arange(mask.shape[1])

fig, ax = plt.subplots(2, 4, figsize=(17, 6.5), constrained_layout=True)
for j, k in enumerate(levels):
    diffs = [np.where(mask[:, :, k], T.astype(float)[:, :, k] - T2000[:, :, k], np.nan) for _, T in states]
    lim = np.nanpercentile(np.abs(np.concatenate([d[np.isfinite(d)] for d in diffs])), 98)
    for i, ((label, _), d) in enumerate(zip(states, diffs)):
        p = ax[i, j]
        c = p.pcolormesh(xt, yt, d.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        p.set_title(f"{ZT[k]:.0f} m, {label}\nRMS {np.sqrt(np.nanmean(d**2)):.3f}, level mean {np.nanmean(d):+.3f} °C", fontsize=9)
        p.set_aspect("equal")
        p.set_xlim(0, 360)
        p.set_ylim(-80, 82)
        if i == 1:
            p.set_xlabel("longitude (°E)")
        if j == 0:
            p.set_ylabel("latitude")
    fig.colorbar(c, ax=ax[:, j], orientation="horizontal", shrink=0.8, label="$T - T_{2000}$ (°C)")
fig.suptitle("Final configuration (L2): temperature minus the 2000-year state (unweighted RMS and mean over ocean cells)")
fig.savefig(os.path.join(DESCRIPTION_DIR, "l2_patterns.png"), dpi=130)
print("written l2_patterns.png")
