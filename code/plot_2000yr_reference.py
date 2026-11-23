#!/usr/bin/env python3
"""
Figure of the 2000-year reference run, from data/intermediate/reference_2000yr/analysis.npz
(written by analyse_2000yr_reference.py). Writes Description/reference_2000yr.png, the
figure of coupled_spinup.tex. Needs matplotlib.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, REFERENCE_DIR  # noqa: E402

a = np.load(os.path.join(REFERENCE_DIR, "analysis.npz"))
years, ZT, mask = a["years"], a["ZT"], a["mask"]
r_lev, r_tot = a["r2000_lev"], a["r2000_tot"]
lev_mean = a["lev_mean"]

fig, ax = plt.subplots(2, 2, figsize=(12, 9))

# (a) distance from T_2000 by level
p = ax[0, 0]
cmap = plt.get_cmap("viridis")
for k in range(len(ZT)):
    p.semilogy(years, np.maximum(r_lev[:, k], 1e-4), color=cmap(k / (len(ZT) - 1)), lw=1)
p.semilogy(years, np.maximum(r_tot, 1e-4), "k", lw=2, label="whole ocean")
p.axvline(200, color="r", ls="--", lw=1, label="$T_{200}$")
p.set_xlabel("year")
p.set_ylabel("RMS $T - T_{2000}$ (°C)")
p.set_title("(a) distance from the year-2000 state, by level\n(colour: bottom = purple, surface = yellow)")
p.set_ylim(1e-3, 3)
p.legend()

# (b) level-mean temperature change from year 200
p = ax[0, 1]
i200 = int(np.argmin(abs(years - 200)))
for k in range(len(ZT)):
    p.plot(years, lev_mean[:, k] - lev_mean[i200, k], color=cmap(k / (len(ZT) - 1)), lw=1, label=f"{ZT[k]:.0f} m" if k in (0, 3, 5, 7, 10, 14) else None)
p.axvline(200, color="r", ls="--", lw=1)
p.set_ylim(-0.3, 0.55)
p.set_xlabel("year")
p.set_ylabel("level-mean T minus year 200 (°C)")
p.set_title("(b) level-mean temperature, relative to year 200")
p.legend(fontsize=8, ncol=2)

# (c) zonal-mean T_2000 - T_200
p = ax[1, 0]
d = np.where(mask, a["T2000"] - a["T200"], np.nan)
with np.errstate(invalid="ignore"):
    zm = np.nanmean(d, axis=0)  # (y, z)
yt = -78.0 + 4.0 * np.arange(mask.shape[1])
lim = np.nanmax(abs(zm))
c = p.pcolormesh(yt, ZT, zm.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
p.invert_yaxis()
p.set_xlabel("latitude")
p.set_ylabel("depth (m)")
p.set_title("(c) zonal mean of $T_{2000} - T_{200}$ (°C)")
fig.colorbar(c, ax=p)

# (d) states of the method against T_200 and T_2000
p = ax[1, 1]
rows = [
    ("conventional yr 200 ($T_{200}$)", 0.000, 0.209),
    ("A: solve, 15 steps", 0.194, 0.252),
    ("A: Veros 20 yr after 9 steps", 0.107, 0.243),
    ("A: Veros 20 yr after 15 steps", 0.111, 0.162),
    ("B: one round (60 d warm-up)", 0.226, 0.309),
]
yy = np.arange(len(rows))
p.barh(yy - 0.2, [r[1] for r in rows], 0.4, label="from $T_{200}$", color="tab:grey")
p.barh(yy + 0.2, [r[2] for r in rows], 0.4, label="from $T_{2000}$", color="tab:blue")
p.set_yticks(yy, [r[0] for r in rows])
p.invert_yaxis()
p.set_xlabel("whole-ocean RMS distance (°C)")
p.set_title("(d) saved states: distance from\nthe old and the new reference")
p.legend()

fig.tight_layout()
fig.savefig(os.path.join(DESCRIPTION_DIR, "reference_2000yr.png"), dpi=150)
print("written reference_2000yr.png")
