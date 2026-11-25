#!/usr/bin/env python3
"""
Single against double precision (Description/single_precision_spinup.tex): the 2000-year
4-degree reference run in float32 (run_2000yr_reference.py --float32) against the float64
run (run_2000yr_reference.py), from the same initial state.

Writes data/intermediate/reference_2000yr_float32/comparison.npz (for
plot_2000yr_float32.py) and prints:
  - the temperature score and volume-mean temperature of both runs over time;
  - the distance of each run's decadal snapshots from the float64 year-2000 state;
  - the year at which the float64 run was as far from its year-2000 state as the float32
    run ends up;
  - the float32 minus float64 difference by level at years 100, 500, 1000 and 2000.

Needs only numpy.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_advection_matrix import COST, DZT, get_mask_t  # noqa: E402
from paths import REFERENCE_DIR as R64  # noqa: E402
from paths import REFERENCE_FLOAT32_DIR as R32  # noqa: E402

mask = get_mask_t() > 0
vol = np.where(mask, COST[None, :, None] * DZT[None, None, :], 0.0)
ZT = (np.cumsum(DZT[::-1]) - 0.5 * DZT[::-1])[::-1]  # cell-centre depth (m), bottom (k = 0) to surface
n_level = np.maximum(mask.sum(axis=(0, 1)), 1)
n_wet = mask.sum()


def daily(run):
    return np.genfromtxt(os.path.join(run, "daily.csv"), delimiter=",", names=True)


def temp(run, day):
    return np.load(os.path.join(run, "snapshots", f"day_{day:06d}.npz"))["temp"].astype(np.float64)


def rms(d):
    d = np.where(mask, d, 0.0)
    return np.sqrt((d**2).sum() / n_wet), np.sqrt((d**2).sum(axis=(0, 1)) / n_level)


def level_mean(d):
    return (np.where(mask, d, 0.0) * vol).sum(axis=(0, 1)) / vol.sum(axis=(0, 1))


a64, a32 = daily(R64), daily(R32)
years_daily = a64["day"] / 360

print("score (median over the preceding half year) and volume-mean T (degC)")
for y in (0.1, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000):
    i = int(round(y * 360)) - 1
    w = slice(max(0, i - 179), i + 1)
    print(
        f"  year {y:6g}: score float64 {np.median(a64['score_temp'][w]):.2e}, float32 {np.median(a32['score_temp'][w]):.2e}; "
        f"mean T float64 {a64['mean_T'][i]:.5f}, float32 {a32['mean_T'][i]:.5f}, difference {a32['mean_T'][i] - a64['mean_T'][i]:+.5f}"
    )

snap_years = np.arange(0, 2001, 10)
T2000 = temp(R64, 2000 * 360)
d64_tot, d64_lev, d32_tot, d32_lev, diff_tot, diff_lev, diff_mean = [], [], [], [], [], [], []
for y in snap_years:
    t64, t32 = temp(R64, y * 360), temp(R32, y * 360)
    for out_tot, out_lev, d in ((d64_tot, d64_lev, t64 - T2000), (d32_tot, d32_lev, t32 - T2000), (diff_tot, diff_lev, t32 - t64)):
        tot, lev = rms(d)
        out_tot.append(tot)
        out_lev.append(lev)
    diff_mean.append(level_mean(t32 - t64))
d64_tot, d32_tot, diff_tot = map(np.array, (d64_tot, d32_tot, diff_tot))
d64_lev, d32_lev, diff_lev, diff_mean = map(np.array, (d64_lev, d32_lev, diff_lev, diff_mean))

end = d32_tot[-1]
reached = snap_years[np.argmax(d64_tot <= end)]
print(f"\nRMS distance from the float64 year-2000 state: float32 at year 2000 {end:.4f} degC;")
print(f"  the float64 run was first that close at year {reached}")
deep = ZT > 3000
end_deep = np.sqrt((d32_lev[-1][deep] ** 2 * n_level[deep]).sum() / n_level[deep].sum())
d64_deep = np.sqrt((d64_lev[:, deep] ** 2 * n_level[deep]).sum(axis=1) / n_level[deep].sum())
print(f"  below 3000 m: float32 at year 2000 {end_deep:.4f} degC; float64 first that close at year {snap_years[np.argmax(d64_deep <= end_deep)]}")

print("\nfloat32 - float64 by level (RMS, and volume mean), degC; depth (m) bottom to surface:")
print("  depth " + " ".join(f"{z:6.0f}" for z in ZT))
for y in (100, 500, 1000, 2000):
    i = int(np.where(snap_years == y)[0][0])
    print(f"  {y:4d} rms  " + " ".join(f"{x:6.3f}" for x in diff_lev[i]))
    print(f"  {y:4d} mean " + " ".join(f"{x:+6.3f}" for x in diff_mean[i]))

np.savez(
    os.path.join(R32, "comparison.npz"),
    years_daily=years_daily, score64=a64["score_temp"], score32=a32["score_temp"], meanT64=a64["mean_T"], meanT32=a32["mean_T"],
    snap_years=snap_years, d64_tot=d64_tot, d32_tot=d32_tot, d64_lev=d64_lev, d32_lev=d32_lev,
    diff_tot=diff_tot, diff_lev=diff_lev, diff_mean=diff_mean, ZT=ZT, reached=reached,
)
print(f"\nwritten {os.path.join(R32, 'comparison.npz')}")
