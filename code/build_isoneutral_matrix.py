#!/usr/bin/env python3
"""
Sparse matrices for the Veros isoneutral (Redi) diffusion term of the global
4-degree setup (global_4deg_no_cycle.py), such that

    dT/dt |_isoneutral = E @ T                        (explicit part)
    (I - dt D33) T_new = T' + dt * E @ T              (Veros time stepping)

E and D33 are frozen given a temperature and salinity field (T, S): the
isoneutral slopes, the taper and the K_11, K_22, K_33 diffusivities that make
up the Redi tensor all depend non-linearly on T and S (through the equation of
state), but once they are evaluated at a given (T, S), the term is linear (it
has no additive forcing) in the tracer it is applied to. This mirrors
build_vertical_mixing_matrix.py, where kappaH and the ice mask play the role
that the slopes and taper play here.

This is the *neutral* part only (K_iso, no GM skew flux): the term called
"isoneutral diffusion" in thermodynamics.thermodynamics, i.e.
isoneutral.isoneutral_diffusion(state, tr=temp, istemp=True) with iso=True.
The separate GM skew-diffusion term (isoneutral_skew_diffusion) is not
included here.

Physics and discretisation
---------------------------
K_iso is the scalar constant settings.K_iso_0 = 1000 m^2/s in this setup
(settings.enable_eke_isopycnal_diffusion is False, so the EKE-based K_gm is
never substituted for it; see eke.set_eke_diffusivities_kernel). The Redi
tensor is built from this constant K_iso and the tapered isoneutral slopes
S_x, S_y (computed from the locally referenced density gradient, i.e. from T,
S and the TEOS-10 equation of state), following the Griffies triad scheme (4
triads per face) used by isoneutral.isoneutral_diffusion_pre and
isoneutral.diffusion._calc_tracer_fluxes/_calc_explicit_part. The code below
is a direct, un-JAX-ed transcription of those two functions.

K_11 and K_22 (the horizontal diagonal entries of K_Redi, on the east and
north faces respectively) turn out to equal K_iso_steep = K_iso = 1000 exactly
wherever the vertical grid is uniform, and a thickness-weighted average close
to 1000 elsewhere (see Description/veros_discretisation.tex); they are
computed here exactly as Veros does, not approximated as a constant.

K_33 (the vertical diagonal entry) is treated implicitly by Veros, exactly
like kappaH in the vertical mixing term; vertical_diffusion_matrix from
build_vertical_mixing_matrix.py is reused unchanged for it (same tridiagonal
form, same grid), with K_33 as the input diffusivity instead of kappaH.

Inputs
------
T, S : vectors of length 54000, interior fields (90, 40, 15) raveled in the
       same order as everywhere else in this directory:
           n = (i * 40 + j) * 15 + k,   field[2:-2, 2:-2, :].ravel()
       (temperature in degC, salinity in g/kg; from the restart's "tau"
       level, as Veros uses for this term).

Main functions
--------------
    K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by = isoneutral_coefficients(T, S)
    E   = isoneutral_explicit_matrix(T, S)     # or from precomputed coefficients
    D33 = isoneutral_implicit_matrix(T, S)     # = vertical_diffusion_matrix(K_33)
    E, D33 = isoneutral_matrix(T, S)

Command line
------------
    python build_isoneutral_matrix.py [restart_file]

writes isoneutral_explicit_matrix.npz and isoneutral_implicit_matrix.npz
(scipy.sparse CSR) to data/intermediate.
"""

import os
import sys

import h5py
import numpy as np
import scipy.sparse as sp

from build_advection_matrix import COST, COSU, DX, DY, DZT, N, NX, NY, NZ, SHAPE, get_mask_t, make_grid_z
from build_vertical_mixing_matrix import DZW, vertical_diffusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from veros.core.density import gsw  # noqa: E402  (pure functions of (sa, ct, p); no state/backend needed)

K_ISO = 1000.0  # m^2/s; settings.K_iso_0, the only value this setup ever uses for K_iso (see module docstring)
K_ISO_STEEP = 1000.0  # settings.K_iso_steep
ISO_SLOPEC = 1.0 / 1000.0  # settings.iso_slopec
ISO_DSLOPE = 4.0 / 1000.0  # settings.iso_dslope
EPSLN = 1e-20

from paths import INIT_FILE, INTERMEDIATE_DIR  # noqa: E402

DEFAULT_RESTART = INIT_FILE

NXF, NYF = NX + 4, NY + 4  # full grid with Veros's 2-cell halo (periodic in x, land in y)

# cosu "ghost-safe" lookup used by the north-south triads and by K_33's facty weight: for a
# triad using the row-(j + jp - 1) metric factor (jp = 0 or 1), index into this dict. jp = 1
# always stays within the 40 interior rows; jp = 0 needs one row below row 0, which only ever
# multiplies a term that is already exactly zero there (the slope feeding it is masked to zero
# at the closed southern wall), so any finite ghost value is safe -- edge replication is used.
_COSU_JP0 = np.concatenate([[COSU[0]], COSU[:-1]])  # index j = COSU[j - 1], COSU[0] (ghost) at j = 0
COSU_FOR_JP = {0: _COSU_JP0, 1: COSU}


def dm_taper(sx):
    """Tapering function for isopycnal slopes (isoneutral.dm_taper)."""
    return 0.5 * (1.0 + np.tanh((-np.abs(sx) + ISO_SLOPEC) / ISO_DSLOPE))


def to_full(a):
    """
    Pad an interior-shaped (90, 40, ...) array to the full (94, 44, ...) grid Veros
    operates on: periodic halos in x (matching utilities.enforce_boundaries), zero
    (land) halos in y (this setup has closed walls in y, so Veros never fills them
    with anything meaningful; every formula below multiplies them by a zero mask
    before they could matter).
    """
    full = np.zeros((NXF, NYF) + a.shape[2:], dtype=float)
    full[2:-2, 2:-2, ...] = a
    full[-2:, 2:-2, ...] = a[:2, ...]
    full[:2, 2:-2, ...] = a[-2:, ...]
    return full


DZW_FULL = np.concatenate([DZW, [2 * DZT[-1] - DZW[-1]]])  # vs.dzw (length 15): one extra top entry


def isoneutral_coefficients(T, S, mask_t=None):
    """
    K_11, K_22, K_33 (interior, shape (90, 40, 15)) and Ai_ez, Ai_nz, Ai_bx, Ai_by
    (interior, shape (90, 40, 15, 2, 2)), following isoneutral.isoneutral_diffusion_pre.

    T, S : temperature (degC) and salinity (g/kg) vectors of length 54000 (see module
           docstring); only used to compute the locally referenced density gradient
           and hence the isoneutral slopes and taper.
    """
    T = np.asarray(T, dtype=float).reshape(SHAPE)
    S = np.asarray(S, dtype=float).reshape(SHAPE)
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)

    m_full = to_full(m)
    m_u_full = m_full * np.roll(m_full, -1, axis=0)  # periodic in x
    m_v_full = np.zeros_like(m_full)
    m_v_full[:, :-1, :] = m_full[:, :-1, :] * m_full[:, 1:, :]  # no wrap in y (closed walls)
    m_w_full = np.zeros_like(m_full)
    m_w_full[:, :, :-1] = m_full[:, :, :-1] * m_full[:, :, 1:]  # no flux through the surface

    T_full = to_full(T)
    S_full = to_full(S)
    zt, _ = make_grid_z()
    press = np.broadcast_to(np.abs(zt)[np.newaxis, np.newaxis, :], T_full.shape)

    drdT_full = m_full * gsw.gsw_drhodT(S_full, T_full, press)
    drdS_full = m_full * gsw.gsw_drhodS(S_full, T_full, press)

    dTdz_full = np.zeros_like(T_full)
    dSdz_full = np.zeros_like(S_full)
    dTdz_full[:, :, :-1] = m_w_full[:, :, :-1] * (T_full[:, :, 1:] - T_full[:, :, :-1]) / DZW[np.newaxis, np.newaxis, :]
    dSdz_full[:, :, :-1] = m_w_full[:, :, :-1] * (S_full[:, :, 1:] - S_full[:, :, :-1]) / DZW[np.newaxis, np.newaxis, :]

    cost_full = np.empty(NYF)
    cost_full[2:-2] = COST
    cost_full[:2] = COST[0]
    cost_full[-2:] = COST[-1]

    dTdx_full = np.zeros_like(T_full)
    dSdx_full = np.zeros_like(S_full)
    dTdx_full[:-1, :, :] = m_u_full[:-1, :, :] * (T_full[1:, :, :] - T_full[:-1, :, :]) / (DX * cost_full[np.newaxis, :, np.newaxis])
    dSdx_full[:-1, :, :] = m_u_full[:-1, :, :] * (S_full[1:, :, :] - S_full[:-1, :, :]) / (DX * cost_full[np.newaxis, :, np.newaxis])

    dTdy_full = np.zeros_like(T_full)
    dSdy_full = np.zeros_like(S_full)
    dTdy_full[:, :-1, :] = m_v_full[:, :-1, :] * (T_full[:, 1:, :] - T_full[:, :-1, :]) / DY
    dSdy_full[:, :-1, :] = m_v_full[:, :-1, :] * (S_full[:, 1:, :] - S_full[:, :-1, :]) / DY

    # ---- K_11, Ai_ez: east face ----------------------------------------------------
    Ai_ez_full = np.zeros((NXF, NYF, NZ, 2, 2))
    sumz = np.zeros((NXF, NYF, NZ))
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = drdT_full[1 + ip : -2 + ip, 2:-2, ki:] * dTdx_full[1:-2, 2:-2, ki:] + drdS_full[1 + ip : -2 + ip, 2:-2, ki:] * dSdx_full[1:-2, 2:-2, ki:]
            drodze = drdT_full[1 + ip : -2 + ip, 2:-2, ki:] * dTdz_full[1 + ip : -2 + ip, 2:-2, : -1 + kr or None] + drdS_full[1 + ip : -2 + ip, 2:-2, ki:] * dSdz_full[1 + ip : -2 + ip, 2:-2, : -1 + kr or None]
            sxe = -drodxe / (np.minimum(0.0, drodze) - EPSLN)
            taper = dm_taper(sxe)
            sumz[1:-2, 2:-2, ki:] += DZW_FULL[np.newaxis, np.newaxis, : -1 + kr or None] * m_u_full[1:-2, 2:-2, ki:] * np.maximum(K_ISO_STEEP, K_ISO * taper)
            Ai_ez_full[1:-2, 2:-2, ki:, ip, kr] = taper * sxe * m_u_full[1:-2, 2:-2, ki:]
    Ai_ez_full[:, :, -1, :, 1] = 0.0  # no flux through a non-existent face above the surface (pad_z_edges)
    K_11_full = np.zeros((NXF, NYF, NZ))
    K_11_full[1:-2, 2:-2, :] = sumz[1:-2, 2:-2, :] / (4.0 * DZT[np.newaxis, np.newaxis, :])

    # ---- K_22, Ai_nz: north face -----------------------------------------------------
    Ai_nz_full = np.zeros((NXF, NYF, NZ, 2, 2))
    sumz = np.zeros((NXF, NYF, NZ))
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = drdT_full[2:-2, 1 + jp : -2 + jp, ki:] * dTdy_full[2:-2, 1:-2, ki:] + drdS_full[2:-2, 1 + jp : -2 + jp, ki:] * dSdy_full[2:-2, 1:-2, ki:]
            drodzn = drdT_full[2:-2, 1 + jp : -2 + jp, ki:] * dTdz_full[2:-2, 1 + jp : -2 + jp, : -1 + kr or None] + drdS_full[2:-2, 1 + jp : -2 + jp, ki:] * dSdz_full[2:-2, 1 + jp : -2 + jp, : -1 + kr or None]
            syn = -drodyn / (np.minimum(0.0, drodzn) - EPSLN)
            taper = dm_taper(syn)
            sumz[2:-2, 1:-2, ki:] += DZW_FULL[np.newaxis, np.newaxis, : -1 + kr or None] * m_v_full[2:-2, 1:-2, ki:] * np.maximum(K_ISO_STEEP, K_ISO * taper)
            Ai_nz_full[2:-2, 1:-2, ki:, jp, kr] = taper * syn * m_v_full[2:-2, 1:-2, ki:]
    Ai_nz_full[:, :, -1, :, 1] = 0.0  # same edge case as Ai_ez
    K_22_full = np.zeros((NXF, NYF, NZ))
    K_22_full[2:-2, 1:-2, :] = sumz[2:-2, 1:-2, :] / (4.0 * DZT[np.newaxis, np.newaxis, :])

    # ---- K_33, Ai_bx, Ai_by: top face -------------------------------------------------
    Ai_bx_full = np.zeros((NXF, NYF, NZ, 2, 2))
    Ai_by_full = np.zeros((NXF, NYF, NZ, 2, 2))
    sumx = np.zeros((NX, NY, NZ - 1))
    sumy = np.zeros((NX, NY, NZ - 1))
    for kr in range(2):
        drodzb = drdT_full[2:-2, 2:-2, kr : -1 + kr or None] * dTdz_full[2:-2, 2:-2, :-1] + drdS_full[2:-2, 2:-2, kr : -1 + kr or None] * dSdz_full[2:-2, 2:-2, :-1]
        for ip in range(2):
            drodxb = drdT_full[2:-2, 2:-2, kr : -1 + kr or None] * dTdx_full[1 + ip : -3 + ip, 2:-2, kr : -1 + kr or None] + drdS_full[2:-2, 2:-2, kr : -1 + kr or None] * dSdx_full[1 + ip : -3 + ip, 2:-2, kr : -1 + kr or None]
            sxb = -drodxb / (np.minimum(0.0, drodzb) - EPSLN)
            taper = dm_taper(sxb)
            sumx += DX * K_ISO * taper * sxb**2 * m_w_full[2:-2, 2:-2, :-1]
            Ai_bx_full[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * m_w_full[2:-2, 2:-2, :-1]
        for jp in range(2):
            facty = COSU_FOR_JP[jp] * DY
            drodyb = drdT_full[2:-2, 2:-2, kr : -1 + kr or None] * dTdy_full[2:-2, 1 + jp : -3 + jp, kr : -1 + kr or None] + drdS_full[2:-2, 2:-2, kr : -1 + kr or None] * dSdy_full[2:-2, 1 + jp : -3 + jp, kr : -1 + kr or None]
            syb = -drodyb / (np.minimum(0.0, drodzb) - EPSLN)
            taper = dm_taper(syb)
            sumy += facty[np.newaxis, :, np.newaxis] * K_ISO * taper * syb**2 * m_w_full[2:-2, 2:-2, :-1]
            Ai_by_full[2:-2, 2:-2, :-1, jp, kr] = taper * syb * m_w_full[2:-2, 2:-2, :-1]

    K_33_full = np.zeros((NXF, NYF, NZ))
    K_33_full[2:-2, 2:-2, :-1] = sumx / (4 * DX) + sumy / (4 * DY * COST[np.newaxis, :, np.newaxis])

    def interior(a):
        return a[2:-2, 2:-2, ...]

    return (
        interior(K_11_full),
        interior(K_22_full),
        interior(K_33_full),
        interior(Ai_ez_full),
        interior(Ai_nz_full),
        interior(Ai_bx_full),
        interior(Ai_by_full),
    )


def _shift_axis(a, d, axis, periodic):
    """
    Return b with b[c] = a[c - d] along `axis` (c the array index along that axis):
    periodic (x) wraps around; otherwise (y, z) the entries that would come from
    outside the domain are set to zero. Every coefficient this is applied to is zero
    wherever that matters physically (no face exists there), so the zero fill is exact.
    """
    if periodic:
        return np.roll(a, d, axis=axis)
    out = np.zeros_like(a)
    sl_out = [slice(None)] * a.ndim
    sl_in = [slice(None)] * a.ndim
    if d > 0:
        sl_out[axis] = slice(d, None)
        sl_in[axis] = slice(0, -d)
    elif d < 0:
        sl_out[axis] = slice(0, d)
        sl_in[axis] = slice(-d, None)
    else:
        return a.copy()
    out[tuple(sl_out)] = a[tuple(sl_in)]
    return out


def flux_divergence_matrix(K11, K22, K1, K2, Ai_ez, Ai_nz, Ai_bx, Ai_by, mask_t=None):
    """
    Sparse CSR matrix (54000 x 54000) for the explicit flux divergence shared by the
    isoneutral (Redi) term here and the GM skew-diffusion term
    (build_gm_matrix.gm_skew_matrix): both are built from the same triad slopes
    (Ai_ez, Ai_nz, Ai_bx, Ai_by, from isoneutral_coefficients) and the same face and
    divergence structure (diffusion._calc_tracer_fluxes / _calc_explicit_part); only
    the horizontal diagonal entries K11, K22 and the two face-direction diffusivities
    K1 (east/north faces), K2 (top face) differ between the two terms:
      Redi (isoneutral_explicit_matrix): K11, K22 as computed,     K1 = K2 = K_iso
      GM skew (gm_skew_matrix):          K11 = K22 = 0,            K1 = -K_gm, K2 = +K_gm
    K11, K22, K1, K2 are each (90, 40, 15) fields (K1 = K2 = K_iso is a true constant
    here, but is still passed as a field for a uniform interface).
    """
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    idx = np.arange(N).reshape(SHAPE)
    rows, cols, vals = [], [], []

    def add(coef, di=0, dj=0, dk=0):
        keep = (coef != 0.0) & (m != 0.0)
        if not np.any(keep):
            return
        col_idx = np.roll(np.roll(np.roll(idx, -di, axis=0), -dj, axis=1), -dk, axis=2)
        rows.append(idx[keep])
        cols.append(col_idx[keep])
        vals.append(coef[keep])

    def divergence(terms, axis, weight, periodic):
        """
        terms: list of (coefficient_field, (di, dj, dk)), each a contribution
        coef[c] * T[c + shift] to the flux through the face between cell c and
        cell c + unit(axis) (c is the cell's own (i, j, k) index along that
        axis). Adds weight * (flux[c] - flux[c - unit(axis)]) to the matrix.
        """
        unit = [0, 0, 0]
        unit[axis] = 1
        for coef, shift in terms:
            # weight is evaluated at the OUTPUT cell only (it is not re-evaluated at c - unit,
            # even though it varies along `axis` for the y and z divergences) -- only coef is
            # shifted, matching explicit_part's single outer division by (cost*dyt) / dzt.
            add(weight * coef, *shift)
            coef_prev = _shift_axis(coef, 1, axis, periodic)  # coef_prev[c] = coef[c - unit]
            shift_prev = tuple(s - u for s, u in zip(shift, unit))
            add(-weight * coef_prev, *shift_prev)

    # The triad sum in flux_east/flux_north is weighted not by K1 at the output cell
    # itself, but by its average over the 4 points (2 columns x 2 levels) adjacent to
    # the face ("diffloc" in diffusion._calc_tracer_fluxes) -- half-weighted (2 points)
    # at the sea floor, where there is no level below. This is invisible for the Redi
    # term (K1 = K_iso is a true constant, so any average of it is itself), but matters
    # for GM skew diffusion, where K1 = -K_gm varies in space.
    def face_diffloc(K, neighbour):
        """neighbour[i, j, k] = K at the other cell of the face (e.g. K rolled by -1 in x)."""
        pair_sum = K + neighbour
        diffloc = np.empty_like(K)
        diffloc[:, :, 1:] = 0.25 * (pair_sum[:, :, 1:] + pair_sum[:, :, :-1])
        diffloc[:, :, 0] = 0.5 * pair_sum[:, :, 0]
        return diffloc / (4.0 * DZT[np.newaxis, np.newaxis, :])

    cdz_e = face_diffloc(K1, np.roll(K1, -1, axis=0))  # periodic east neighbour
    cdz_n = face_diffloc(K1, _shift_axis(K1, -1, 1, periodic=False))  # north neighbour, zero beyond row 39

    # ---- east/west faces: K11 local term + Ai_ez triads -----------------------------
    # flux_east's own local term is K11*(T[i+1]-T[i])/(cost*dxu); the generic divergence
    # below applies the OUTER 1/(cost*dxt) once more, so this coefficient must carry the
    # inner 1/(cost*dxu) itself (the triad terms carry no such inner factor).
    coef_local_x = K11 / (COST[np.newaxis, :, np.newaxis] * DX)
    terms_x = [(coef_local_x, (1, 0, 0)), (-coef_local_x, (0, 0, 0))]
    for kr in range(2):
        for ip in range(2):
            coef = cdz_e * Ai_ez[:, :, :, ip, kr]
            if kr == 0:
                terms_x += [(coef, (ip, 0, 0)), (-coef, (ip, 0, -1))]
            else:
                terms_x += [(coef, (ip, 0, 1)), (-coef, (ip, 0, 0))]
    weight_x = np.broadcast_to((1.0 / (COST * DX))[np.newaxis, :, np.newaxis], SHAPE)
    divergence(terms_x, axis=0, weight=weight_x, periodic=True)

    # ---- north/south faces: K22 local term + Ai_nz triads ---------------------------
    cosu3 = COSU[np.newaxis, :, np.newaxis]
    # same inner-factor point as coef_local_x: flux_north's own local term is
    # cosu*K22*(T[j+1]-T[j])/dyu, with the outer 1/(cost*dyt) applied by the divergence.
    coef_local_y = cosu3 * K22 / DY
    terms_y = [(coef_local_y, (0, 1, 0)), (-coef_local_y, (0, 0, 0))]
    for kr in range(2):
        for jp in range(2):
            coef = cosu3 * cdz_n * Ai_nz[:, :, :, jp, kr]
            if kr == 0:
                terms_y += [(coef, (0, jp, 0)), (-coef, (0, jp, -1))]
            else:
                terms_y += [(coef, (0, jp, 1)), (-coef, (0, jp, 0))]
    weight_y = np.broadcast_to((1.0 / (COST * DY))[np.newaxis, :, np.newaxis], SHAPE)
    divergence(terms_y, axis=1, weight=weight_y, periodic=False)

    # ---- top/bottom faces: Ai_bx, Ai_by -----------------------------------------------
    # (for the Redi term, K33 -- not part of K2's contribution here -- is treated
    # implicitly, see isoneutral_implicit_matrix; the GM skew term has no such piece)
    cost3 = COST[np.newaxis, :, np.newaxis]
    terms_z = []
    for kr in range(2):
        for ip in range(2):
            coef = (K2 / (4.0 * DX * cost3)) * Ai_bx[:, :, :, ip, kr]
            terms_z += [(coef, (ip, 0, kr)), (-coef, (ip - 1, 0, kr))]
        for jp in range(2):
            coef = (K2 * COSU_FOR_JP[jp][np.newaxis, :, np.newaxis] / (4.0 * DY * cost3)) * Ai_by[:, :, :, jp, kr]
            terms_z += [(coef, (0, jp, kr)), (-coef, (0, jp - 1, kr))]
    weight_z = np.broadcast_to((1.0 / DZT)[np.newaxis, np.newaxis, :], SHAPE)
    divergence(terms_z, axis=2, weight=weight_z, periodic=False)

    M = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
    M.sum_duplicates()
    return M


def isoneutral_explicit_matrix(T, S, mask_t=None, coefficients=None):
    """
    Sparse CSR matrix E (54000 x 54000): the explicit part of the isoneutral diffusion
    term (horizontal diagonal entries K_11, K_22, plus the Ai_ez/Ai_nz/Ai_bx/Ai_by
    triads), such that dT/dt |_isoneutral, explicit = E @ T for the given (T, S).
    A thin wrapper around flux_divergence_matrix with K1 = K2 = K_iso.

    coefficients, if given, is the tuple returned by isoneutral_coefficients(T, S)
    (pass it in to avoid recomputing it, e.g. when also calling
    isoneutral_implicit_matrix for the same T, S).
    """
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    if coefficients is None:
        coefficients = isoneutral_coefficients(T, S, mask_t=m)
    K_11, K_22, _, Ai_ez, Ai_nz, Ai_bx, Ai_by = coefficients
    K1K2 = np.full(SHAPE, K_ISO)
    return flux_divergence_matrix(K_11, K_22, K1K2, K1K2, Ai_ez, Ai_nz, Ai_bx, Ai_by, mask_t=m)


def isoneutral_implicit_matrix(T, S, mask_t=None, coefficients=None):
    """
    Sparse CSR matrix D33 (54000 x 54000): the implicit (K_33) part of the isoneutral
    diffusion term, in the same tridiagonal-in-k form as vertical_diffusion_matrix
    (build_vertical_mixing_matrix.py); Veros solves

        (I - dt * D33) T_new = T' + dt * E @ T

    i.e. D33 should be treated exactly like the D matrix of the vertical mixing term.
    """
    if coefficients is None:
        coefficients = isoneutral_coefficients(T, S, mask_t=mask_t)
    K_33 = coefficients[2]
    return vertical_diffusion_matrix(K_33.ravel(), mask_t=mask_t)


def isoneutral_matrix(T, S, mask_t=None):
    """Convenience: (E, D33) for the given (T, S); see isoneutral_explicit_matrix/_implicit_matrix."""
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    coefficients = isoneutral_coefficients(T, S, mask_t=m)
    return (
        isoneutral_explicit_matrix(T, S, mask_t=m, coefficients=coefficients),
        isoneutral_implicit_matrix(T, S, mask_t=m, coefficients=coefficients),
    )


def salinity_from_restart(restart_file, level="tau"):
    """Salinity vector (length 54000) from a restart file."""
    with h5py.File(restart_file, "r") as f:
        slot = int(f["core/tau"][()]) if level == "tau" else int(level)
        return np.array(f["core/salt"][2:-2, 2:-2, :, slot]).ravel()


def main():
    restart_file = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESTART)
    from build_vertical_mixing_matrix import temperature_from_restart

    T = temperature_from_restart(restart_file)
    S = salinity_from_restart(restart_file)
    E, D33 = isoneutral_matrix(T, S)

    here = INTERMEDIATE_DIR
    os.makedirs(here, exist_ok=True)
    sp.save_npz(os.path.join(here, "isoneutral_explicit_matrix.npz"), E)
    sp.save_npz(os.path.join(here, "isoneutral_implicit_matrix.npz"), D33)
    print(f"Built E: {E.shape[0]} x {E.shape[1]} ({E.nnz} non-zeros), D33: {D33.nnz} non-zeros, from {restart_file}")


if __name__ == "__main__":
    main()
