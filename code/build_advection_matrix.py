#!/usr/bin/env python3
"""
Sparse matrix M for the Veros tracer advection term of the global 4-degree
setup (global_4deg_no_cycle.py), such that

    dT/dt |_advection = M @ T

for fixed velocities (u, v, w).

The scheme is the second-order centred flux form described in
Description/veros_discretisation.tex ("Spatial discretisation of the
advection term"). The diagonal is zero because velocities computed by the
model satisfy discrete continuity, so the matrix has at most six non-zero
entries per row.

Main function
-------------
    M = advection_matrix(uvw)

uvw is a single vector of length 3 * 90 * 40 * 15 = 162000:

    uvw = np.concatenate([u.ravel(), v.ravel(), w.ravel()])

where u, v and w are the interior velocity fields (Veros index order x, y, z,
z = 0 at the bottom), i.e. u = core/u[2:-2, 2:-2, :, tau] from a restart file:
  u[i,j,k]  eastward velocity on the east face of cell (i,j,k)   (m/s)
  v[i,j,k]  northward velocity on the north face                 (m/s)
  w[i,j,k]  upward velocity on the top face                      (m/s)
uvw_from_restart(file) returns this vector for a restart file.

Vector ordering of T
--------------------
T is temp[2:-2, 2:-2, :, tau] (shape 90 x 40 x 15) flattened in C order:

    n = (i * 40 + j) * 15 + k,      i = 0..89, j = 0..39, k = 0..14

so T_vec = temp[2:-2, 2:-2, :, tau].ravel() and
tendency = (M @ T_vec).reshape(90, 40, 15). Land cells are included in the
vector, but their rows and columns in M are zero. u, v and w use the same
ordering.

Command line
------------
    python build_advection_matrix.py [restart_file]

writes advection_matrix.npz (scipy.sparse CSR, load with
scipy.sparse.load_npz) and advection_matrix_info.npz (maskT, grid, source
file) to data/intermediate. The default restart file is
data/input/Init_state.restart.h5.
"""

import os
import sys
import warnings

import h5netcdf
import h5py
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import FORCING_FILE, INIT_FILE, INTERMEDIATE_DIR  # noqa: E402

DEFAULT_RESTART = INIT_FILE

NX, NY, NZ = 90, 40, 15
SHAPE = (NX, NY, NZ)
N = NX * NY * NZ
# cell thicknesses, bottom (k = 0) to surface (k = 14), as in set_grid
DZT = np.array(
    [50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0]
)[::-1]
RADIUS = 6370e3
DEGTOM = RADIUS / 180.0 * np.pi
DX = 4.0 * DEGTOM
DY = 4.0 * DEGTOM
YT = -78.0 + 4.0 * np.arange(NY)  # latitude of T points
YU = YT + 2.0  # latitude of the north faces (v points)
XT = 2.0 + 4.0 * np.arange(NX)
COST = np.cos(YT * np.pi / 180.0)
COSU = np.cos(YU * np.pi / 180.0)


def read_forcing(var):
    """Same as GlobalFourDegreeSetup._read_forcing for 3D/2D fields."""
    with h5netcdf.File(FORCING_FILE, "r") as infile:
        return np.array(infile.variables[var]).T


def make_grid_z():
    """Depths of T points and top faces (negative, bottom to surface), as Veros computes them."""
    zu = np.concatenate([[0.0], np.cumsum(DZT[1:])])
    zt = np.empty(NZ)
    zt[0] = zu[0] - DZT[0] * 0.5
    for k in range(1, NZ):
        zt[k] = 2 * zu[k - 1] - zt[k - 1]
    # Veros: zt = zt - zw[-1], zw = zw - zw[-1]
    return zt - zu[-1], zu - zu[-1]


def make_mask_t():
    """maskT (90, 40, 15) and kbot (90, 40), following set_topography and calc_topo."""
    zt, _ = make_grid_z()
    bathymetry = read_forcing("bathymetry")
    salt = read_forcing("salinity")[:, :, ::-1]
    land = (zt[np.newaxis, np.newaxis, :] <= bathymetry[..., np.newaxis]) | (salt == 0.0)
    kbot = 1 + land.sum(axis=2)
    kbot = np.where((bathymetry == 0) | (kbot == NZ), 0, kbot)
    ks = np.arange(NZ)
    mask_t = (kbot[..., np.newaxis] > 0) & (kbot[..., np.newaxis] - 1 <= ks)
    return mask_t.astype(float), kbot


_MASK_T = None


def get_mask_t():
    """maskT, computed once from the bathymetry (it does not change in time)."""
    global _MASK_T
    if _MASK_T is None:
        _MASK_T, _ = make_mask_t()
    return _MASK_T


def uvw_from_restart(restart_file, level="tau"):
    """Return the single velocity vector [u, v, w] (length 162000) from a restart file.

    level is "tau" (the current time level, used by Veros for advection), or an integer slot.
    """
    with h5py.File(restart_file, "r") as f:
        slot = int(f["core/tau"][()]) if level == "tau" else int(level)
        u, v, w = (np.array(f[f"core/{name}"][2:-2, 2:-2, :, slot]) for name in ("u", "v", "w"))
    return np.concatenate([u.ravel(), v.ravel(), w.ravel()])


def advection_matrix(uvw, mask_t=None, check_continuity=True):
    """
    Sparse CSR matrix M (54000 x 54000) such that M @ T is the Veros advective tendency.

    uvw : vector of length 162000, np.concatenate([u.ravel(), v.ravel(), w.ravel()])
          with u, v, w of shape (90, 40, 15) (see module docstring).
    mask_t : optional (90, 40, 15) ocean mask; computed from the bathymetry if omitted.
    check_continuity : warn if (u, v, w) do not satisfy discrete continuity, in which
          case the omitted diagonal (-D/2) would not be zero and M would differ from Veros.
    """
    uvw = np.asarray(uvw, dtype=float)
    if uvw.shape != (3 * N,):
        raise ValueError(f"uvw must have shape ({3 * N},), got {uvw.shape}")
    u, v, w = (a.reshape(SHAPE) for a in np.split(uvw, 3))
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)

    # face masks (periodic in x; closed walls in y; no flux through the surface)
    m_u = m * np.roll(m, -1, axis=0)  # east face of cell i: cells i and i+1 (periodic)
    m_v = np.zeros_like(m)
    m_v[:, :-1] = m[:, :-1] * m[:, 1:]  # north face; the face north of j = 39 is a wall
    m_w = np.zeros_like(m)
    m_w[:, :, :-1] = m[:, :, :-1] * m[:, :, 1:]  # top face; the surface face (k = 14) has no flux

    U = m_u * u
    V = m_v * v * COSU[np.newaxis, :, np.newaxis]
    W = m_w * w
    U_west = np.roll(U, 1, axis=0)
    V_south = np.zeros_like(V)
    V_south[:, 1:] = V[:, :-1]
    W_below = np.zeros_like(W)
    W_below[:, :, 1:] = W[:, :, :-1]

    cdx = (COST * DX)[np.newaxis, :, np.newaxis]
    cdy = (COST * DY)[np.newaxis, :, np.newaxis]
    dz = DZT[np.newaxis, np.newaxis, :]

    if check_continuity:
        D = m * ((U - U_west) / cdx + (V - V_south) / cdy + (W - W_below) / dz)
        scale = max(np.abs(U).max() / DX, 1e-300)
        if np.abs(D).max() > 1e-8 * scale:
            warnings.warn(
                f"velocities are not discretely divergence-free (max|D| = {np.abs(D).max():.2e} 1/s, "
                f"max|u|/dx = {scale:.2e} 1/s); the matrix omits the -D/2 diagonal, so it will not match Veros"
            )

    idx = np.arange(N).reshape(SHAPE)
    rows, cols, vals = [], [], []

    def add(coef, col_idx):
        keep = (coef != 0.0) & (m != 0.0)
        rows.append(idx[keep])
        cols.append(col_idx[keep])
        vals.append(coef[keep])

    add(-U / (2 * cdx), np.roll(idx, -1, axis=0))  # east neighbour T[i+1]
    add(U_west / (2 * cdx), np.roll(idx, 1, axis=0))  # west neighbour T[i-1]
    add(-V / (2 * cdy), np.roll(idx, -1, axis=1))  # north neighbour T[j+1] (V = 0 at j = 39)
    add(V_south / (2 * cdy), np.roll(idx, 1, axis=1))  # south neighbour T[j-1] (zero at j = 0)
    add(-W / (2 * dz), np.roll(idx, -1, axis=2))  # cell above T[k+1] (W = 0 at k = 14)
    add(W_below / (2 * dz), np.roll(idx, 1, axis=2))  # cell below T[k-1] (zero at k = 0)

    M = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
    ).tocsr()
    M.sum_duplicates()
    return M


def main():
    restart_file = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESTART)

    zt, zw = make_grid_z()
    mask_t, kbot = make_mask_t()
    with h5py.File(restart_file, "r") as f:
        tau = int(f["core/tau"][()])
    M = advection_matrix(uvw_from_restart(restart_file), mask_t=mask_t)

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    sp.save_npz(os.path.join(INTERMEDIATE_DIR, "advection_matrix.npz"), M)
    np.savez(
        os.path.join(INTERMEDIATE_DIR, "advection_matrix_info.npz"),
        shape=np.array(SHAPE),
        maskT=mask_t.astype(np.int8),
        kbot=kbot,
        xt=XT,
        yt=YT,
        zt=zt,
        zw=zw,
        dzt=DZT,
        restart_file=os.path.basename(restart_file),
        tau=tau,
        ordering="T_vec = temp[2:-2, 2:-2, :, tau].ravel()  (C order over x, y, z; z = 0 is the bottom)",
    )
    print(f"Built {M.shape[0]} x {M.shape[1]} matrix with {M.nnz} non-zeros from {restart_file} (tau = {tau})")


if __name__ == "__main__":
    main()
