#!/usr/bin/env python3
"""
Why the competition's initial state is not self-consistent
(Description/equilibration_experiments.tex, Section "The competition's initial state").

Prints, for data/input/Init_state.restart.h5:
  - how far the stored density and N^2 are from those of the stored temperature and
    uniform salinity (at every stored time level);
  - the salinity that, with the stored temperature, reproduces the stored density: its
    range, its volume mean and its correlation with the salinity of the forcing file;
  - the interior faces that are stable in the stored N^2 but unstable for the density of
    the temperature alone;
  - the steady circulation (build_momentum_matrix.solve_velocity, with the friction
    coefficients of the stored state) for the stored density and for the density of the
    temperature alone, compared with the stored u and v (relative L2 difference);
and, if the 2000-year reference run has reached year 200, the same density and N^2 check
for T_200, whose stored fields are consistent.

Run with the Python environment in which Veros is installed (it uses Veros's equation of
state only):
    python code/diagnose_initial_state.py
"""

import os
import sys

import h5netcdf
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import FORCING_FILE, INIT_FILE, T200_FILE  # noqa: E402

import build_momentum_matrix as bm  # noqa: E402
from build_advection_matrix import COST, DZT, SHAPE, get_mask_t, make_grid_z  # noqa: E402
from build_gm_matrix import nsqr_from_TS  # noqa: E402
from veros.core.density import gsw  # noqa: E402

mask = get_mask_t().reshape(SHAPE) > 0
mask_w = np.zeros(SHAPE, dtype=bool)
mask_w[:, :, :-1] = mask[:, :, :-1] & mask[:, :, 1:]  # interior faces between two ocean cells
zt, _ = make_grid_z()
press = np.abs(zt)[np.newaxis, np.newaxis, :] * np.ones(SHAPE)
volume = np.where(mask, COST[np.newaxis, :, np.newaxis] * DZT[np.newaxis, np.newaxis, :], 0.0)


def level(f, name, n):
    return np.array(f[f"core/{name}"][2:-2, 2:-2, :, n])


def consistency(restart_file, label):
    """Largest differences between the stored rho, N^2 and those recomputed from T and S."""
    with h5py.File(restart_file, "r") as f:
        for n in range(3):
            T, S = level(f, "temp", n), level(f, "salt", n)
            d_rho = np.abs(bm.density(T, S) - level(f, "rho", n))[mask].max()
            d_n2 = np.abs(nsqr_from_TS(T, S) - level(f, "Nsqr", n))[mask_w].max()
            print(f"  {label}, time level {n}: max |rho - rho(T, S)| {d_rho:.3g} kg/m^3, max |N^2 - N^2(T, S)| {d_n2:.3g} 1/s^2")


def salinity_for_density(T, rho, S0):
    """S with gsw_rho(S, T, p) = rho in every ocean cell, by Newton iteration."""
    S = np.full(SHAPE, S0)
    for _ in range(30):
        r = np.asarray(gsw.gsw_rho(S, T, press)) - rho
        S = S - r / np.asarray(gsw.gsw_drhodS(S, T, press))
        if np.abs(r[mask]).max() < 1e-12:
            break
    return S


def relative_difference(a, b, m):
    return np.sqrt(((a - b) ** 2)[m].sum() / (b**2)[m].sum())


def main():
    with h5py.File(INIT_FILE, "r") as f:
        tau = int(f["core/tau"][()])
        time_days = float(f["core/time"][()]) / 86400.0
        T, S = level(f, "temp", tau), level(f, "salt", tau)
        rho_s, n2_s = level(f, "rho", tau), level(f, "Nsqr", tau)
        u_s, v_s = level(f, "u", tau), level(f, "v", tau)
        tke, eke = level(f, "tke", tau), level(f, "eke", tau)
    S_uniform = S[mask]
    print(f"Init_state.restart.h5: model time {time_days:.0f} days; salinity {S_uniform.min():.9f} to {S_uniform.max():.9f} g/kg")
    print(f"  largest stored N^2 {n2_s[mask_w].max():.2g} 1/s^2")
    consistency(INIT_FILE, "Init_state")

    S_star = salinity_for_density(T, rho_s, float(S_uniform.mean()))
    err = np.abs(np.asarray(gsw.gsw_rho(S_star, T, press)) - rho_s)[mask].max()
    with h5netcdf.File(FORCING_FILE, "r") as f:
        S_forcing = np.array(f["salinity"]).transpose(2, 1, 0)[:, :, ::-1]  # (z, y, x), surface first, to (x, y, z), bottom first
    corr = np.corrcoef(S_star[mask], S_forcing[mask])[0, 1]
    vol_mean = (S_star * volume).sum() / volume.sum()
    print("salinity that reproduces the stored density with the stored temperature:")
    print(f"  {S_star[mask].min():.2f} to {S_star[mask].max():.2f} g/kg, volume mean {vol_mean:.9f} g/kg (density residual {err:.1e})")
    print(f"  correlation with the salinity of the forcing file: {corr:.3f}")

    n2_T = nsqr_from_TS(T, S)
    n_faces = mask_w.sum()
    flipped = mask_w & (n2_s > 0) & (n2_T < 0)
    print(f"interior faces: {n_faces}; stable in the stored N^2 but unstable for the temperature alone: {flipped.sum()} ({100 * flipped.sum() / n_faces:.1f}%)")
    print(f"  unstable for the temperature alone {100 * (mask_w & (n2_T < 0)).sum() / n_faces:.1f}%, in the stored N^2 {100 * (mask_w & (n2_s < 0)).sum() / n_faces:.1f}%")

    kappaM = bm.kappaM_from_tke(tke, n2_s)
    kappa_gm = bm.kappa_gm_from_eke(n2_s, eke)
    m_u, m_v, _, _ = bm.face_masks()
    m_u, m_v = m_u.reshape(SHAPE) > 0, m_v.reshape(SHAPE) > 0
    for label, rho in (("stored density", rho_s), ("density of the temperature alone", bm.density(T, S))):
        u, v, _, _ = bm.solve_velocity(rho, kappaM, kappa_gm)
        print(f"steady circulation for the {label}: differs from the stored u, v by "
              f"{100 * relative_difference(u, u_s, m_u):.0f}% and {100 * relative_difference(v, v_s, m_v):.0f}%")

    if os.path.exists(T200_FILE):
        consistency(T200_FILE, "T_200")


if __name__ == "__main__":
    main()
