#!/usr/bin/env python3
"""
Figure: the 2000-year 4-degree reference run in single against double precision, from
data/intermediate/reference_2000yr_float32/comparison.npz (written by
compare_2000yr_float32.py). Writes Description/reference_2000yr_float32.png, the figure
of single_precision_spinup.tex. Needs matplotlib.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, REFERENCE_FLOAT32_DIR  # noqa: E402

a = np.load(os.path.join(REFERENCE_FLOAT32_DIR, "comparison.npz"))
C64, C32 = "#2a78d6", "#eb6834"  # blue: float64, orange: float32
INK, MUTED = "#2b2b2b", "#8a8a85"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6})


def running_median(x, n):
    """Median over blocks of n days (for a readable daily score)."""
    m = len(x) // n
    return np.median(x[: m * n].reshape(m, n), axis=1)


fig, ax = plt.subplots(2, 2, figsize=(12, 9))

# (a) the score, log-log: float32 reaches a floor by year 10
p = ax[0, 0]
yd = a["years_daily"]
p.loglog(yd[:360], a["score64"][:360], color=C64, lw=1.5)
p.loglog(yd[:360], a["score32"][:360], color=C32, lw=1.5)
n = 36  # a tenth of a year
yb = running_median(yd[360:], n)
p.loglog(yb, running_median(a["score64"][360:], n), color=C64, lw=1.5, label="double (float64)")
p.loglog(yb, running_median(a["score32"][360:], n), color=C32, lw=1.5, label="single (float32)")
p.set_xlabel("year")
p.set_ylabel("temperature score (daily change)")
p.set_title("(a) score: daily for the first year, then median of 36 days")
p.legend(frameon=False)

# (b) volume-mean temperature
p = ax[0, 1]
p.plot(yd[::36], a["meanT64"][::36], color=C64, lw=1.5, label="double")
p.plot(yd[::36], a["meanT32"][::36], color=C32, lw=1.5, label="single")
i = -1
p.annotate(f"+{a['meanT32'][i] - a['meanT64'][i]:.3f} °C", (yd[i], (a["meanT32"][i] + a["meanT64"][i]) / 2), xytext=(-60, 0), textcoords="offset points", color=INK, va="center")
p.set_xlabel("year")
p.set_ylabel("volume-mean temperature (°C)")
p.set_title("(b) volume-mean temperature")
p.legend(frameon=False)

# (c) distance from the float64 year-2000 state
p = ax[1, 0]
sy = a["snap_years"]
p.semilogy(sy[:-1], a["d64_tot"][:-1], color=C64, lw=1.5, label="double")
p.semilogy(sy, a["d32_tot"], color=C32, lw=1.5, label="single")
end, reached = a["d32_tot"][-1], int(a["reached"])
p.axhline(end, color=MUTED, lw=1, ls="--")
p.plot([reached], [end], "o", ms=8, color=C64, mec="white", mew=2)
p.annotate(f"double was this close at year {reached}", (reached, end), xytext=(8, 10), textcoords="offset points", color=INK)
p.set_xlabel("year")
p.set_ylabel("RMS $T - T_{2000}^{\\mathrm{double}}$ (°C)")
p.set_title("(c) distance from the double-precision year-2000 state")
p.legend(frameon=False, loc="lower left")

# (d) where the difference is: level-mean single minus double, by depth
p = ax[1, 1]
ZT = a["ZT"]
shades = ["#f8c9b3", "#f29b76", "#eb6834", "#a8431c"]  # one hue, light (early) to dark (late)
for y, c in zip((100, 500, 1000, 2000), shades):
    k = int(np.where(sy == y)[0][0])
    p.plot(a["diff_mean"][k], ZT, "-o", color=c, lw=2, ms=4, label=f"year {y}")
p.axvline(0, color=MUTED, lw=1)
p.invert_yaxis()
p.set_xlabel("level-mean $T_{\\mathrm{single}} - T_{\\mathrm{double}}$ (°C)")
p.set_ylabel("depth (m)")
p.set_title("(d) single minus double precision, by depth")
p.legend(frameon=False, loc="lower left")

fig.suptitle("4° Veros, 2000 years from the same initial state: single against double precision", color=INK)
fig.tight_layout()
out = os.path.join(DESCRIPTION_DIR, "reference_2000yr_float32.png")
fig.savefig(out, dpi=150)
print(f"written {out}")
