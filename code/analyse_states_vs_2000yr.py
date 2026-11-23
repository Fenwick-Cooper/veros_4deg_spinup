#!/usr/bin/env python3
"""
Distance of temperature fields from the 200-year and the 2000-year reference states, and
the conventional year at which the 2000-year run is equally close. The arguments are
restart files (their temperature at the current time level) or .npz files of saved
temperature fields (spinup_from_init.py --save-states), e.g.

    python code/analyse_states_vs_2000yr.py competition.restart.h5 equilibrium.restart.h5
    python code/analyse_states_vs_2000yr.py data/intermediate/from_init/cycles_L2_states.npz

Needs data/intermediate/reference_2000yr/analysis.npz (analyse_2000yr_reference.py) and
numpy and h5py only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import REFERENCE_DIR  # noqa: E402

a = np.load(os.path.join(REFERENCE_DIR, "analysis.npz"))
mask, ZT, years = a["mask"], a["ZT"], a["years"]
T200, T2000 = a["T200"], a["T2000"]
r_tot, r_lev = a["r2000_tot"], a["r2000_lev"]
n_level = np.maximum(mask.sum(axis=(0, 1)), 1)
k1000 = int(np.argmin(abs(ZT - 1000)))


def rms(T, ref):
    d = np.where(mask, T - ref, 0.0)
    return np.sqrt((d**2).sum() / mask.sum()), np.sqrt((d**2).sum(axis=(0, 1)) / n_level)


def equivalent_year(value, series):
    """First decadal snapshot of the conventional run at least as close to T_2000."""
    i = np.where(series <= value)[0]
    return f"{years[i[0]]:.0f}" if len(i) else ">2000"


from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402

fields = {}
for arg in sys.argv[1:]:
    if arg.endswith(".h5"):
        fields[os.path.basename(arg)] = temperature_from_restart(arg).reshape(mask.shape)
    else:
        s = np.load(arg)
        for name in sorted(s.files, key=lambda n: int(n.split("day")[-1])):
            fields[name] = s[name]
print(f"{'state':24s} {'from T_200':>10s} {'from T_2000: total':>19s} {'bottom':>7s} {'~1000 m':>8s} {'surface':>8s}   conventional year as close (total / bottom / ~1000 m)")
for name, T in fields.items():
    T = np.asarray(T, dtype=float)
    d200, _ = rms(T, T200)
    d, lev = rms(T, T2000)
    print(
        f"{name:24s} {d200:10.3f} {d:19.3f} {lev[0]:7.3f} {lev[k1000]:8.3f} {lev[-1]:8.3f}   "
        f"{equivalent_year(d, r_tot)} / {equivalent_year(lev[0], r_lev[:, 0])} / {equivalent_year(lev[k1000], r_lev[:, k1000])}"
    )
print(f"\nfor comparison, conventional run from T_2000: year 0 {r_tot[0]:.3f}, year 10 {r_tot[1]:.3f}, year 20 {r_tot[2]:.3f}, year 200 {r_tot[20]:.3f}")
