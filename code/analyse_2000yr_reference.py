#!/usr/bin/env python3
"""
Analysis of the 2000-year reference run (run_2000yr_reference.py).

Writes data/intermediate/reference_2000yr/analysis.npz (for plot_2000yr_reference.py and
analyse_states_vs_2000yr.py) and prints:
  - the drift of the run (decadal and centennial RMS change, by level);
  - the distance of the decadal snapshots from T_200 and from T_2000, by level;
  - an exponential fit of the volume-mean temperature;
  - the distance of the saved states of Experiments A and B (if they have been made) and of
    competition.restart.h5 and equilibrium.restart.h5 (if made) from T_200 and T_2000.

T_200 and T_2000 are the year-200 and year-2000 restart files of the reference run.
Needs numpy, scipy and h5py only.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import INTERMEDIATE_DIR, REFERENCE_DIR, REPO_DIR, T200_FILE, T2000_FILE  # noqa: E402
from build_advection_matrix import COST, DZT, get_mask_t  # noqa: E402
from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402

OUT = REFERENCE_DIR

mask = get_mask_t() > 0
vol = np.where(mask, COST[None, :, None] * DZT[None, None, :], 0.0)
# depth of the cell centres (m, positive down), bottom (k = 0) to surface
ZT = (np.cumsum(DZT[::-1]) - 0.5 * DZT[::-1])[::-1]
n_level = np.maximum(mask.sum(axis=(0, 1)), 1)
n_wet = mask.sum()


def rms(a, b):
    d = np.where(mask, a - b, 0.0)
    return np.sqrt((d**2).sum() / n_wet), np.sqrt((d**2).sum(axis=(0, 1)) / n_level)


def mean_T(T):
    return (np.where(mask, T, 0.0) * vol).sum() / vol.sum()


snaps = sorted(glob.glob(os.path.join(OUT, "snapshots", "day_*.npz")))
days = np.array([int(np.load(f)["day"]) for f in snaps])
T = np.array([np.load(f)["temp"].astype(float) for f in snaps])
years = days / 360.0
T200 = temperature_from_restart(T200_FILE).reshape(mask.shape)
T2000 = temperature_from_restart(T2000_FILE).reshape(mask.shape)

print(f"== the run: {len(T)} decadal snapshots, years {years[0]:.0f} to {years[-1]:.0f} ==")
print(f"  float32 snapshot at year 2000 vs the restart file: max |dT| {abs(np.where(mask, T[-1] - T2000, 0)).max():.2e}")

z_levels = np.arange(mask.shape[2])
dist200 = np.array([rms(t, T200) for t in T], dtype=object)
r200_tot = np.array([x[0] for x in dist200])
r200_lev = np.array([x[1] for x in dist200])
r2000_tot, r2000_lev = zip(*[rms(t, T2000) for t in T])
r2000_tot, r2000_lev = np.array(r2000_tot), np.array(r2000_lev)
meanT = np.array([mean_T(t) for t in T])
# change over 10 years and over 100 years, by level
dec_tot, dec_lev = zip(*[rms(T[i], T[i - 1]) for i in range(1, len(T))])
dec_tot, dec_lev = np.array(dec_tot), np.array(dec_lev)


print("\n== distance from T_2000 (RMS degC) by level, selected years ==")
sel = [int(np.argmin(abs(years - y))) for y in (0, 10, 50, 100, 200, 500, 1000, 1500, 1900)]
print("  level  depth(m) " + " ".join(f"{years[i]:7.0f}" for i in sel))
for k in range(mask.shape[2] - 1, -1, -1):
    print(f"  {k:5d} {ZT[k]:9.0f} " + " ".join(f"{r2000_lev[i][k]:7.3f}" for i in sel))
print("  total          " + " ".join(f"{r2000_tot[i]:7.3f}" for i in sel))

print("\n== drift: RMS change over the last 10 years and per century ==")
for y in (100, 200, 500, 1000, 1500, 2000):
    i = int(np.argmin(abs(years - y)))
    j = int(np.argmin(abs(years - (y - 100))))
    t_c, l_c = rms(T[i], T[j])
    print(f"  year {y:5d}: 10-yr change {dec_tot[i - 1]:.4f} (bottom {dec_lev[i - 1][0]:.4f}); 100-yr change {t_c:.4f} (bottom {l_c[0]:.4f}, surface {l_c[-1]:.4f})")

# exponential fit of the volume-mean temperature over years 300-2000: m(t) = m_inf - A exp(-t/tau)
from scipy.optimize import curve_fit  # noqa: E402

fit_range = years >= 300
f = lambda t, m_inf, A, tau: m_inf - A * np.exp(-t / tau)  # noqa: E731
p, _ = curve_fit(f, years[fit_range], meanT[fit_range], p0=(meanT[-1] + 0.01, 0.2, 500.0))
print(f"\n== volume-mean T: {meanT[0]:.4f} (day 0), {meanT[years == 200][0]:.4f} (yr 200), {meanT[-1]:.4f} (yr 2000) ==")
print(f"  fit over years 300-2000: m_inf {p[0]:.4f}, e-folding {p[2]:.0f} years; remaining at year 2000 {p[0] - meanT[-1]:.4f} degC")
# same fit per level for the level-mean temperature
lev_mean = np.array([(np.where(mask, t, 0) * vol).sum(axis=(0, 1)) / vol.sum(axis=(0, 1)).clip(1e-30) for t in T])
print("  per level: level, depth, mean T at 200 / 2000, fitted limit, e-folding (yr), remaining at 2000")
tau_lev = np.full(mask.shape[2], np.nan)
inf_lev = np.full(mask.shape[2], np.nan)
for k in range(mask.shape[2] - 1, -1, -1):
    try:
        pk, _ = curve_fit(f, years[fit_range], lev_mean[fit_range, k], p0=(lev_mean[-1, k], -0.1, 500.0), maxfev=20000)
        tau_lev[k], inf_lev[k] = pk[2], pk[0]
    except RuntimeError:
        pk = (np.nan, np.nan, np.nan)
    print(f"  {k:3d} {ZT[k]:7.0f}  {lev_mean[years == 200][0][k]:7.3f} {lev_mean[-1, k]:7.3f}  {pk[0]:7.3f} {pk[2]:7.0f} {pk[0] - lev_mean[-1, k]:8.4f}")

print("\n== saved states of the method: RMS distance from T_200 and from T_2000 (total, bottom, mid ~1000 m, surface) ==")
k1000 = int(np.argmin(abs(ZT - 1000)))


def report(label, Tm):
    a, la = rms(Tm, T200)
    b, lb = rms(Tm, T2000)
    print(f"  {label:42s} T_200: {a:.3f} ({la[0]:.3f}, {la[k1000]:.3f}, {la[-1]:.3f})   T_2000: {b:.3f} ({lb[0]:.3f}, {lb[k1000]:.3f}, {lb[-1]:.3f})")
    return a, b, la, lb


methods = {}
report("conventional year 200 (T_200)", T200)
for y in (10, 20, 24, 100, 500, 1000):
    i = int(np.argmin(abs(years - y)))
    methods[f"conventional year {y}"] = report(f"conventional year {years[i]:.0f}", T[i])
fn = os.path.join(INTERMEDIATE_DIR, "coupled_steady_state.npz")
if os.path.exists(fn):
    cs = np.load(fn)
    for s in (4, 8, 12, len(cs["T_steps"]) - 1):
        hist = cs["history"][s]
        methods[f"Exp A solve step {s}"] = report(f"Exp A solve, step {s} ({hist[0]:.0f} pseudo-yr)", cs["T_steps"][s])
for label in ("none", "step05", "step09", "step15"):
    fn = os.path.join(INTERMEDIATE_DIR, f"veros_from_coupled_{label}.npz")
    if os.path.exists(fn):
        methods[f"Exp A Veros {label}"] = report(f"Exp A, Veros 20 yr after restart ({label})", np.load(fn)["T"])
for tag in ("w60_e", "w90_e", "w30_e"):
    fn = os.path.join(INTERMEDIATE_DIR, "from_init", f"solve_{tag}.npz")
    if os.path.exists(fn):
        Ts = np.load(fn)["T_steps"]
        methods[f"Exp B one round {tag}"] = report(f"Exp B one round ({tag}), last solve step", Ts[-1])
for label in ("competition", "equilibrium"):
    fn = os.path.join(REPO_DIR, f"{label}.restart.h5")
    if os.path.exists(fn):
        methods[label] = report(f"{label}.restart.h5", temperature_from_restart(fn).reshape(mask.shape))

np.savez(
    os.path.join(OUT, "analysis.npz"),
    years=years, ZT=ZT, mask=mask,
    r200_tot=r200_tot, r200_lev=np.stack(r200_lev),
    r2000_tot=r2000_tot, r2000_lev=r2000_lev,
    dec_tot=dec_tot, dec_lev=dec_lev, meanT=meanT, lev_mean=lev_mean,
    fit=p, tau_lev=tau_lev, inf_lev=inf_lev,
    T200=T200, T2000=T2000, T0=T[0],
)
print("\nwritten", os.path.join(OUT, "analysis.npz"))
