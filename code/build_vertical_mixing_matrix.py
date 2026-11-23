#!/usr/bin/env python3
"""
Sparse matrix and constant vector for the Veros vertical mixing term (with its
surface heat flux) of the global 4-degree setup
(global_4deg_no_cycle.py), such that

    dT/dt |_vertical mixing = V @ T + b

for a fixed vertical diffusivity kappaH and a fixed ice mask.

The scheme is described in Description/veros_discretisation.tex ("Spatial
discretisation of the vertical mixing term"). The matrix has two parts,

    V = D + R

  D  vertical diffusion: tridiagonal within each column, depends only on kappaH
  R  surface relaxation: diagonal, -Q_nec / (rho_0 c_p0 dz_top) in the
     surface cells that are not ice-masked; zero elsewhere

and b is the part of the surface flux that does not depend on T,
(Q_net + Q_nec T_clim) / (rho_0 c_p0 dz_top) in the surface cells that are not
ice-masked.

Ice mask
--------
The surface flux is switched off where T_top < -1.8 degC and the flux would
cool the ocean. This makes the term non-linear in T. The functions here take a
temperature vector T_ice that defines the ice mask, and V and b are then exact
for any T as long as the ice mask is the one given by T_ice.

Time stepping in Veros
----------------------
Veros treats D implicitly and the surface flux (R and b) explicitly, with the
ice mask and the relaxation evaluated at the current time level tau:

    (I - dt D) T^{n+1} = T* + dt (R T^n + b)

where T* is the temperature after the advection and isoneutral updates. Use
vertical_diffusion_matrix, surface_relaxation_matrix and surface_forcing_vector
separately to reproduce this; vertical_mixing_matrix returns D + R.

Inputs
------
All 3D vectors have length 90 * 40 * 15 = 54000 and use the same ordering as
build_advection_matrix.py:

    n = (i * 40 + j) * 15 + k,   i.e.  field[2:-2, 2:-2, :].ravel()

with k = 0 at the bottom.
  T       temperature (degC), temp[2:-2, 2:-2, :, tau].ravel()
  kappaH  vertical diffusivity (m^2/s) on the TOP face of each cell,
          kappaH[2:-2, 2:-2, :].ravel(). Values at faces that are not between
          two ocean cells (including the surface, k = 14) are ignored.
kappaH is not stored in the restart files; kappaH_from_restart(file)
computes it from the stored TKE state exactly as Veros does at the start of
the next time step.
"""

import os

import h5netcdf
import h5py
import numpy as np
import scipy.sparse as sp

from paths import ECMWF_FILE  # noqa: E402
from build_advection_matrix import COST, DZT, N, NZ, SHAPE, get_mask_t, make_grid_z, read_forcing

RHO_0 = 1024.0  # kg/m^3
CP_0 = 3991.86795711963  # J/(kg K), hard-coded in set_forcing_kernel
T_ICE = -1.8  # degC

# TKE closure settings used by the setup (global_4deg_no_cycle.py and Veros defaults)
C_K = 0.1
MXL_MIN = 1e-8
KAPPAM_MAX = 100.0
KAPPAH_MIN = 2e-5


def make_dzw():
    """Distance between neighbouring T points: dzw[k] = zt[k+1] - zt[k], k = 0..13."""
    zt, _ = make_grid_z()
    return np.diff(zt)


DZW = make_dzw()

_SURFACE = None


def surface_fields():
    """(Q_net, Q_nec, T_clim), each (90, 40), as set up in GlobalFourDegreeSetup.set_initial_conditions.

    Q_net : net surface heat flux into the ocean (W/m^2), global area mean removed, zero on land
    Q_nec : relaxation coefficient (W/m^2/K), the annual mean of the ECMWF Q3 field (0 where missing)
    T_clim: climatological SST (degC)
    """
    global _SURFACE
    if _SURFACE is None:
        with h5netcdf.File(ECMWF_FILE, "r") as f:
            q3 = np.array(f.variables["Q3"])  # float32, (12, 40, 90)
        # Veros takes the 12-month mean in float32 with JAX; this reproduces it bit for bit
        # (sequential float32 sum, multiplied by float32(1/12)).
        q_sum = q3[0].copy()
        for month in q3[1:]:
            q_sum = q_sum + month
        q_nec = (q_sum * np.float32(1.0 / len(q3))).T.astype(float)
        q_nec = np.where(q_nec <= -1e10, 0.0, q_nec)

        q_net = -read_forcing("q_net").astype(float)
        q_net = np.where(q_net <= -1e10, 0.0, q_net)
        area = np.broadcast_to(COST[np.newaxis, :], q_net.shape)  # dx and dy are uniform
        q_net = (q_net - np.sum(q_net * area) / np.sum(area)) * get_mask_t()[:, :, -1]

        t_clim = read_forcing("sst").astype(float)
        _SURFACE = (q_net, q_nec, t_clim)
    return _SURFACE


def surface_heat_flux(T, mask_t=None):
    """Surface temperature flux Q_T (K m/s, positive into the ocean), (90, 40), including the ice mask."""
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    q_net, q_nec, t_clim = surface_fields()
    t_top = np.asarray(T, dtype=float).reshape(SHAPE)[:, :, -1] * m[:, :, -1]
    q_t = (q_net + q_nec * (t_clim - t_top)) * m[:, :, -1] / CP_0 / RHO_0
    return np.where(ice_mask(T, mask_t), 0.0, q_t)


def ice_mask(T, mask_t=None):
    """(90, 40) boolean: True where the surface flux is switched off (T_top < -1.8 and the flux cools)."""
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    q_net, q_nec, t_clim = surface_fields()
    t_top = np.asarray(T, dtype=float).reshape(SHAPE)[:, :, -1] * m[:, :, -1]
    q_t = (q_net + q_nec * (t_clim - t_top)) * m[:, :, -1]
    return (t_top < T_ICE) & (q_t < 0.0)


def _check(vec, name):
    vec = np.asarray(vec, dtype=float)
    if vec.shape != (N,):
        raise ValueError(f"{name} must have shape ({N},), got {vec.shape}")
    return vec


def _surface_active(T_ice, mask_t):
    """(90, 40) array: 1 for surface ocean cells where the surface flux is on, else 0."""
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    return m[:, :, -1] * ~ice_mask(_check(T_ice, "T_ice"), mask_t)


def vertical_diffusion_matrix(kappaH, mask_t=None):
    """
    Sparse CSR matrix D (54000 x 54000): the vertical diffusion part of the vertical mixing term,
    with zero flux through the sea floor and no surface flux (the surface flux is R T + b).

    kappaH : vector of length 54000, the diffusivity on the top face of each cell (see module docstring).
    """
    kappa = _check(kappaH, "kappaH").reshape(SHAPE)
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)

    # face conductances g[k] = kappaH[k] / dzw[k] on the face between k and k+1 (both cells ocean)
    g = np.zeros(SHAPE)
    g[:, :, :-1] = m[:, :, :-1] * m[:, :, 1:] * kappa[:, :, :-1] / DZW
    g_below = np.zeros(SHAPE)
    g_below[:, :, 1:] = g[:, :, :-1]
    dz = DZT[np.newaxis, np.newaxis, :]

    idx = np.arange(N).reshape(SHAPE)
    rows, cols, vals = [], [], []

    def add(coef, col_idx):
        keep = (coef != 0.0) & (m != 0.0)
        rows.append(idx[keep])
        cols.append(col_idx[keep])
        vals.append(coef[keep])

    add(g / dz, np.roll(idx, -1, axis=2))  # cell above T[k+1] (g = 0 at k = 14)
    add(g_below / dz, np.roll(idx, 1, axis=2))  # cell below T[k-1] (g_below = 0 at k = 0)
    add(-(g + g_below) / dz, idx)  # diagonal

    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
    ).tocsr()


def surface_relaxation_matrix(T_ice, mask_t=None):
    """
    Sparse diagonal matrix R (54000 x 54000): -Q_nec / (rho_0 c_p0 dz_top) in the surface ocean
    cells that are not ice-masked, zero elsewhere. The ice mask is computed from T_ice.
    """
    _, q_nec, _ = surface_fields()
    diag = np.zeros(SHAPE)
    diag[:, :, -1] = -_surface_active(T_ice, mask_t) * q_nec / (RHO_0 * CP_0 * DZT[-1])
    return sp.diags(diag.ravel(), format="csr")


def surface_forcing_vector(T_ice, mask_t=None):
    """
    Constant vector b (length 54000): (Q_net + Q_nec T_clim) / (rho_0 c_p0 dz_top) in the surface
    ocean cells that are not ice-masked, zero elsewhere. The ice mask is computed from T_ice.
    """
    q_net, q_nec, t_clim = surface_fields()
    b = np.zeros(SHAPE)
    b[:, :, -1] = _surface_active(T_ice, mask_t) * (q_net + q_nec * t_clim) / (RHO_0 * CP_0 * DZT[-1])
    return b.ravel()


def vertical_mixing_matrix(kappaH, T_ice, mask_t=None):
    """
    Sparse CSR matrix V = D + R (54000 x 54000) such that

        dT/dt |_vertical mixing = V @ T + surface_forcing_vector(T_ice)

    for the given kappaH, with the ice mask fixed at the one computed from T_ice.
    """
    return (vertical_diffusion_matrix(kappaH, mask_t) + surface_relaxation_matrix(T_ice, mask_t)).tocsr()


def kappaH_from_restart(restart_file, level="tau"):
    """
    Vertical diffusivity kappaH (vector of length 54000) that Veros computes from the TKE state
    in restart_file at the start of the next time step (tke.set_tke_diffusivities with this
    setup's settings). Uses tke and Nsqr at the given time level, and K_diss_v.
    """
    with h5py.File(restart_file, "r") as f:
        slot = int(f["core/tau"][()]) if level == "tau" else int(level)
        tke = np.array(f["core/tke"][2:-2, 2:-2, :, slot])
        nsqr = np.array(f["core/Nsqr"][2:-2, 2:-2, :, slot])
        k_diss_v = np.array(f["core/K_diss_v"][2:-2, 2:-2, :])
    return kappaH_from_tke(tke, nsqr, k_diss_v)


def kappaH_from_tke(tke, nsqr, k_diss_v, mask_t=None):
    """
    Vertical diffusivity kappaH (vector of length 54000) as tke.set_tke_diffusivities
    computes it from the turbulent kinetic energy tke, the stratification nsqr and the
    dissipation by vertical friction k_diss_v (each (90, 40, 15) or flat).
    """
    tke = np.asarray(tke, dtype=float).reshape(SHAPE)
    nsqr = np.asarray(nsqr, dtype=float).reshape(SHAPE)
    k_diss_v = np.asarray(k_diss_v, dtype=float).reshape(SHAPE)
    m = get_mask_t() if mask_t is None else mask_t
    mask_w = m.copy()
    mask_w[:, :, :-1] = np.minimum(m[:, :, :-1], m[:, :, 1:])
    _, zw = make_grid_z()

    sqrttke = np.sqrt(np.maximum(0.0, tke))
    mxl = np.sqrt(2) * sqrttke / np.sqrt(np.maximum(1e-12, nsqr)) * mask_w
    # tke_mxl_choice = 2: |d mxl / dz| <= 1, downwards then upwards
    for k in range(NZ - 2, -1, -1):
        mxl[:, :, k] = np.minimum(mxl[:, :, k], mxl[:, :, k + 1] + DZT[k + 1])
    mxl[:, :, -1] = np.minimum(mxl[:, :, -1], MXL_MIN + DZT[-1])
    for k in range(1, NZ):
        mxl[:, :, k] = np.minimum(mxl[:, :, k], mxl[:, :, k - 1] + DZT[k])
    mxl = np.maximum(mxl, MXL_MIN)

    kappa_m = np.minimum(KAPPAM_MAX, C_K * mxl * sqrttke)
    ri = nsqr / np.maximum(k_diss_v / np.maximum(1e-12, kappa_m), 1e-12)
    prandtl = np.maximum(1.0, np.minimum(10.0, 6.6 * ri))
    kappa_h = np.maximum(KAPPAH_MIN, kappa_m / prandtl)
    # Bryan-Lewis background profile (enable_kappaH_profile)
    kappa_bl = (0.8 + 1.05 / np.pi * np.arctan((-zw - 2500.0) / 222.2)) * 1e-4
    kappa_h = np.maximum(kappa_h, kappa_bl[np.newaxis, np.newaxis, :])
    return kappa_h.ravel()


def temperature_from_restart(restart_file, level="tau"):
    """Temperature vector (length 54000) from a restart file."""
    with h5py.File(restart_file, "r") as f:
        slot = int(f["core/tau"][()]) if level == "tau" else int(level)
        return np.array(f["core/temp"][2:-2, 2:-2, :, slot]).ravel()
