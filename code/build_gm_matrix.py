#!/usr/bin/env python3
"""
Sparse matrix for the Veros GM (Gent--McWilliams) skew-diffusion term of the global
4-degree setup (global_4deg_no_cycle.py), such that

    dT/dt |_GM skew = F @ T

F is frozen given a temperature, salinity, buoyancy frequency and eddy kinetic energy
field (T, S, N^2, E): it reuses the Redi term's isoneutral slopes
(build_isoneutral_matrix.isoneutral_coefficients), since Veros computes them once and
shares them between the two terms, and reuses that module's flux_divergence_matrix for
the actual matrix assembly (see its docstring for exactly what differs between the two
terms). What's new here is K_gm, the thickness diffusivity, which depends on N^2 and on
E (the prognostic eddy kinetic energy): see isoneutral.isoneutral_diffusion_pre (shared
slopes), isoneutral.diffusion._calc_tracer_fluxes/_calc_explicit_part (shared flux and
divergence form) and eke.set_eke_diffusivities_kernel (K_gm).

Why N^2 is an input, not derived from T, S
-------------------------------------------
N^2 is mathematically a function of T and S (via the equation of state): see
nsqr_from_TS below, which reproduces thermodynamics.calc_eq_of_state's formula exactly
and matches Veros's stored N^2 field whenever that field is itself self-consistent with
the restart's T, S (true for a restart saved after the model has actually been stepped,
such as Init_state.restart.h5). It does Veros does *not* recompute N^2 before the
first EKE/GM calculation of a run, though (eke.set_eke_diffusivities runs before
thermodynamics.thermodynamics in veros.Veros.step): it simply uses whatever N^2 is
already stored from the previous step. For a restart saved without ever being stepped
(Init_state.restart.h5), the stored N^2 is consequently not what nsqr_from_TS would
give, and this is what Veros itself would also use. N^2 is therefore taken as an
explicit input here, exactly as kappaH is in build_vertical_mixing_matrix.py, rather
than silently recomputed from T, S; nsqr_from_TS is provided as a correct, independent
way to obtain it when the state is known to be self-consistent (or to inspect how it
depends on T, S).

Unlike the Redi term, the GM skew term is entirely explicit: there is no K_33-like
implicit piece (K_gm's own diagonal entries, K_11 and K_22, are exactly zero; see
diffusion._calc_tracer_fluxes, "K_11 = zeros, K_22 = zeros" when the term is not the
Redi term itself).

Inputs
------
T, S  : as in build_isoneutral_matrix.py: vectors of length 54000, interior fields
        (90, 40, 15) raveled as field[2:-2, 2:-2, :].ravel() (temperature in degC,
        salinity in g/kg, from the restart's "tau" level).
Nsqr  : buoyancy frequency squared (1/s^2), vector of length 54000, same ordering
        (Nsqr[2:-2, 2:-2, :, tau].ravel()); see nsqr_from_restart, nsqr_from_TS.
E     : eddy kinetic energy (m^2/s^2), vector of length 54000, same ordering
        (eke[2:-2, 2:-2, :, tau].ravel()).

Main functions
--------------
    K_gm = eke_diffusivity(Nsqr, E)
    F = gm_skew_matrix(T, S, Nsqr, E)

Command line
------------
    python build_gm_matrix.py [restart_file]

writes gm_skew_matrix.npz (scipy.sparse CSR) to data/intermediate.
"""

import os
import sys

import h5py
import numpy as np
import scipy.sparse as sp

from build_advection_matrix import COST, DX, DY, DZT, N, NY, SHAPE, YT, get_mask_t, make_grid_z
from build_isoneutral_matrix import DZW_FULL, isoneutral_coefficients, flux_divergence_matrix
from build_vertical_mixing_matrix import DZW, RHO_0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from veros.core.density import gsw  # noqa: E402  (pure functions of (sa, ct, p); no state/backend needed)

GRAV = 9.81  # m/s^2, settings.grav
OMEGA = np.pi / 43082.0  # rad/s, settings.omega (Earth's rotation rate)
EKE_K_MAX = 1e4  # m^2/s, settings.eke_k_max
EKE_C_K = 0.4  # settings.eke_c_k
EKE_CROSS = 2.0  # settings.eke_cross
EKE_CRHIN = 1.0  # settings.eke_crhin
EKE_LMIN = 100.0  # m, settings.eke_lmin

from paths import INIT_FILE, INTERMEDIATE_DIR  # noqa: E402

DEFAULT_RESTART = INIT_FILE

# grid: latitude of T points, Coriolis parameter and its meridional gradient (beta);
# geometric, not prognostic, so these are computed once from the grid (numerics.set_coriolis,
# numerics.calc_beta), extended by one row on each side (edge-replicated spacing) for beta's
# centred difference.
_YT_EXT = np.concatenate([[YT[0] - 4.0], YT, [YT[-1] + 4.0]])  # length 42
_CORIOLIS_EXT = 2.0 * OMEGA * np.sin(_YT_EXT * np.pi / 180.0)
CORIOLIS_T = _CORIOLIS_EXT[1:-1]  # length 40
BETA = 0.5 * ((_CORIOLIS_EXT[2:] - _CORIOLIS_EXT[1:-1]) / DY + (_CORIOLIS_EXT[1:-1] - _CORIOLIS_EXT[:-2]) / DY)


def nsqr_from_TS(T, S, mask_t=None):
    """
    Buoyancy frequency squared N^2 (1/s^2), interior (90, 40, 15), on the top face of
    each cell (same place as kappa in the vertical mixing term): following
    thermodynamics.calc_eq_of_state, N^2 is the locally referenced density difference
    across that face, not a linearised (rho_T, rho_S) estimate like the isoneutral
    slopes use. N^2[-1] (the surface) repeats N^2[-2], as Veros does. See the module
    docstring for why this is not used by default in eke_diffusivity/gm_skew_matrix.
    """
    T = np.asarray(T, dtype=float).reshape(SHAPE)
    S = np.asarray(S, dtype=float).reshape(SHAPE)
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    mask_w = np.zeros_like(m)
    mask_w[:, :, :-1] = m[:, :, :-1] * m[:, :, 1:]

    zt, _ = make_grid_z()
    press = np.abs(zt)  # dbar, same approximation as elsewhere in this directory

    rho = gsw.gsw_rho(S, T, press[np.newaxis, np.newaxis, :]) * m
    rho_above = gsw.gsw_rho(S[:, :, 1:], T[:, :, 1:], press[np.newaxis, np.newaxis, :-1])

    nsqr = np.zeros(SHAPE)
    fxa = -GRAV / RHO_0 / DZW[np.newaxis, np.newaxis, :] * mask_w[:, :, :-1]
    nsqr[:, :, :-1] = fxa * (rho_above - rho[:, :, :-1])
    nsqr[:, :, -1] = nsqr[:, :, -2]
    return nsqr


def eke_diffusivity(Nsqr, E, mask_t=None):
    """
    K_gm (m^2/s), interior (90, 40, 15): the GM thickness diffusivity, following
    eke.set_eke_diffusivities_kernel (enable_eke, not enable_eke_isopycnal_diffusion,
    which only affects K_iso -- see build_isoneutral_matrix.py).

    Nsqr : buoyancy frequency squared (1/s^2), vector or (90, 40, 15) array.
    E    : eddy kinetic energy (m^2/s^2), vector or (90, 40, 15) array.
    """
    Nsqr = np.asarray(Nsqr, dtype=float).reshape(SHAPE)
    E = np.asarray(E, dtype=float).reshape(SHAPE)
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)

    # Veros's vs.maskW is the ocean mask itself at the surface (not the "no flux through
    # the surface" face mask used everywhere else in this directory, which is zero there);
    # C_rossby's vertical sum uses that convention, over the full column.
    mask_w_col = m.copy()
    mask_w_col[:, :, :-1] = np.minimum(m[:, :, :-1], m[:, :, 1:])

    c_rossby = np.sum(np.sqrt(np.maximum(0.0, Nsqr)) * DZW_FULL[np.newaxis, np.newaxis, :] * mask_w_col, axis=2) / np.pi

    coriolis = CORIOLIS_T[np.newaxis, :]
    beta = BETA[np.newaxis, :]
    L_rossby = np.minimum(c_rossby / np.maximum(np.abs(coriolis), 1e-16), np.sqrt(c_rossby / np.maximum(2 * beta, 1e-16)))

    sqrteke = np.sqrt(np.maximum(0.0, E))
    L_rhines = np.sqrt(sqrteke / np.maximum(beta[:, :, np.newaxis], 1e-16))
    eke_len = np.maximum(EKE_LMIN, np.minimum(EKE_CROSS * L_rossby[:, :, np.newaxis], EKE_CRHIN * L_rhines))
    return np.minimum(EKE_K_MAX, EKE_C_K * eke_len * sqrteke)


def gm_skew_matrix(T, S, Nsqr, E, mask_t=None):
    """
    Sparse CSR matrix F (54000 x 54000) such that dT/dt |_GM skew = F @ T, for the given
    (T, S, Nsqr, E). Reuses the Redi term's slopes (isoneutral_coefficients, from T, S)
    and its flux divergence assembly (flux_divergence_matrix), with K_11 = K_22 = 0 and
    K_gm (from Nsqr, E) in place of K_iso, with opposite sign on the east/north faces
    (see build_isoneutral_matrix.py and Description/veros_discretisation.tex).
    """
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    _, _, _, Ai_ez, Ai_nz, Ai_bx, Ai_by = isoneutral_coefficients(T, S, mask_t=m)
    K_gm = eke_diffusivity(Nsqr, E, mask_t=m)
    zero = np.zeros(SHAPE)
    return flux_divergence_matrix(zero, zero, -K_gm, K_gm, Ai_ez, Ai_nz, Ai_bx, Ai_by, mask_t=m)


def nsqr_from_restart(restart_file, level="tau"):
    """Buoyancy frequency squared N^2 (1/s^2) vector (length 54000) from a restart file."""
    with h5py.File(restart_file, "r") as f:
        slot = int(f["core/tau"][()]) if level == "tau" else int(level)
        return np.array(f["core/Nsqr"][2:-2, 2:-2, :, slot]).ravel()


def eke_from_restart(restart_file, level="tau"):
    """Eddy kinetic energy vector (length 54000) from a restart file."""
    with h5py.File(restart_file, "r") as f:
        slot = int(f["core/tau"][()]) if level == "tau" else int(level)
        return np.array(f["core/eke"][2:-2, 2:-2, :, slot]).ravel()


def main():
    restart_file = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESTART)
    from build_isoneutral_matrix import salinity_from_restart
    from build_vertical_mixing_matrix import temperature_from_restart

    T = temperature_from_restart(restart_file)
    S = salinity_from_restart(restart_file)
    Nsqr = nsqr_from_restart(restart_file)
    E = eke_from_restart(restart_file)
    F = gm_skew_matrix(T, S, Nsqr, E)

    here = INTERMEDIATE_DIR
    os.makedirs(here, exist_ok=True)
    sp.save_npz(os.path.join(here, "gm_skew_matrix.npz"), F)
    print(f"Built F: {F.shape[0]} x {F.shape[1]} ({F.nnz} non-zeros), from {restart_file}")


if __name__ == "__main__":
    main()
