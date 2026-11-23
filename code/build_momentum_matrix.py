#!/usr/bin/env python3
"""
The steady circulation of the Veros global 4-degree setup as a linear problem: given
the density field (i.e. temperature; salinity is uniform here) and the two vertical
friction coefficients, solve directly for the velocities (u, v, w) that Veros's momentum
equations settle to, instead of time stepping them.

Veros advances u, v by (veros/core/momentum.py, friction.py, external/solve_stream.py)

    du/dt = Coriolis + wind stress + advection + friction - grad(p_hydro) - grad(p_s)

where the surface pressure p_s is not computed explicitly: the depth-mean flow comes from
a streamfunction, which is equivalent to choosing p_s so that the depth-integrated flow
is non-divergent (rigid lid). At a steady state every term balances, which is the linear
system solved here for x = [u, v, p_s]:

    | F_u   C_uv  G_u | | u   |     | f_u |
    | C_vu  F_v   G_v | | v   | = - | f_v |          f = wind + pressure gradient (+ extra)
    | D_u   D_v   0   | | p_s |     | 0   |

    C       Coriolis (the energy-conserving four-point average of tend_coriolisf)
    F       friction: F = Fz + Fh
            Fz  vertical friction (kappaM) followed by the eddy "TEM" friction (kappa_gm).
                Veros applies both implicitly, one after the other, over the momentum
                time step dt, so their tendency is ((I - dt A_gm)^-1 (I - dt A_v)^-1 - I)/dt
                and not simply (A_v + A_gm); with kappaM up to 100 m^2/s the difference
                is large, so Fz is built exactly that way (a dense 15 x 15 block per column)
            Fh  horizontal harmonic friction with cosine scaling
    G       gradient of the surface pressure (p_s is in units of pressure / rho_0)
    D       depth-integrated divergence
    f_wind  surface stress in the top layer
    f_pres  -grad(p_hydro), with p_hydro the hydrostatic integral of the density

Left out: momentum advection and the spherical metric terms, the only non-linear terms
in u, v. They are about 0.1% of the Coriolis term here; pass them as `extra` if wanted.

kappaM (from the TKE closure) and kappa_gm (from the EKE closure) depend on the state in
a strongly non-linear way and are inputs, to be taken from a recent model state.

Vector ordering is that of build_advection_matrix.py: a field on the (90, 40, 15) grid is
flattened as n = (i * 40 + j) * 15 + k (k = 0 at the bottom); u is on east faces, v on
north faces, w on top faces. p_s is on the (90, 40) columns, n2 = i * 40 + j.

Main functions
--------------
    u, v, w, p_s = solve_velocity(rho, kappaM, kappa_gm)
    rho = density(T, S)
    kappaM = kappaM_from_tke(tke, Nsqr, K_diss_v);  kappa_gm = kappa_gm_from_eke(Nsqr, eke)
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_advection_matrix import COST, COSU, DEGTOM, DX, DY, DZT, N, NX, NY, NZ, SHAPE, get_mask_t, make_grid_z, read_forcing  # noqa: E402
from build_gm_matrix import CORIOLIS_T, GRAV, eke_diffusivity  # noqa: E402
from build_isoneutral_matrix import DZW_FULL  # noqa: E402
from build_vertical_mixing_matrix import C_K, DZW, KAPPAM_MAX, MXL_MIN, RHO_0  # noqa: E402
from veros.core.density import gsw  # noqa: E402  (pure functions of (sa, ct, p))

DT_MOM = 1800.0  # s, settings.dt_mom
A_H = (4 * DEGTOM) ** 3 * 2e-11  # m^2/s, settings.A_h
COS_POWER = 1  # settings.hor_friction_cosPower
KAPPAM_MIN = 2e-4  # m^2/s, settings.kappaM_min

N2 = NX * NY
IDX = np.arange(N).reshape(SHAPE)
IDX2 = np.arange(N2).reshape(NX, NY)
_COST3 = COST[np.newaxis, :, np.newaxis]
_COSU3 = COSU[np.newaxis, :, np.newaxis]
_F3 = CORIOLIS_T[np.newaxis, :, np.newaxis]


def _diag(a):
    return sp.diags(np.broadcast_to(a, SHAPE).ravel().astype(float), format="csr")


def _shift(axis, step, periodic):
    """Sparse S with (S a)[i] = a[i + step] along `axis`; zero where i + step is outside
    the grid, unless periodic."""
    cols = np.roll(IDX, -step, axis=axis)
    valid = np.ones(SHAPE, dtype=bool)
    if not periodic:
        sl = [slice(None)] * 3
        sl[axis] = slice(-step, None) if step > 0 else slice(0, -step)
        valid[tuple(sl)] = False
    return sp.csr_matrix((valid.ravel().astype(float), (IDX.ravel(), cols.ravel())), shape=(N, N))


_I = sp.identity(N, format="csr")
_SXP, _SXM = _shift(0, 1, True), _shift(0, -1, True)
_SYP, _SYM = _shift(1, 1, False), _shift(1, -1, False)
# column -> all 15 levels of that column, and its transpose (sum over the column)
_EXPAND = sp.csr_matrix((np.ones(N), (IDX.ravel(), np.repeat(IDX2[:, :, np.newaxis], NZ, axis=2).ravel())), shape=(N, N2))


def _north(a):
    """a[j + 1] in place of a[j] (zero in the last row)."""
    out = np.zeros_like(a)
    out[:, :-1] = a[:, 1:]
    return out


def face_masks(mask_t=None):
    """(maskT, maskU, maskV, maskW) as (90, 40, 15) float arrays. maskW follows Veros:
    both cells wet, except at the surface where it equals maskT."""
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    m_u = m * np.roll(m, -1, axis=0)
    m_v = np.zeros_like(m)
    m_v[:, :-1] = m[:, :-1] * m[:, 1:]
    m_w = m.copy()
    m_w[:, :, :-1] = m[:, :, :-1] * m[:, :, 1:]
    return m, m_u, m_v, m_w


# --------------------------------------------------------------------------------------
# friction coefficients from the closures' state (inputs to the operator)
# --------------------------------------------------------------------------------------
def kappaM_from_tke(tke, Nsqr, K_diss_v=None, mask_t=None):
    """Vertical viscosity kappaM (90, 40, 15), on top faces, as tke.set_tke_diffusivities
    computes it (tke_mxl_choice = 2). K_diss_v is not used by kappaM (only by kappaH)."""
    tke = np.asarray(tke, dtype=float).reshape(SHAPE)
    Nsqr = np.asarray(Nsqr, dtype=float).reshape(SHAPE)
    _, _, _, m_w = face_masks(mask_t)
    sqrttke = np.sqrt(np.maximum(0.0, tke))
    mxl = np.sqrt(2) * sqrttke / np.sqrt(np.maximum(1e-12, Nsqr)) * m_w
    for k in range(NZ - 2, -1, -1):
        mxl[:, :, k] = np.minimum(mxl[:, :, k], mxl[:, :, k + 1] + DZT[k + 1])
    mxl[:, :, -1] = np.minimum(mxl[:, :, -1], MXL_MIN + DZT[-1])
    for k in range(1, NZ):
        mxl[:, :, k] = np.minimum(mxl[:, :, k], mxl[:, :, k - 1] + DZT[k])
    mxl = np.maximum(mxl, MXL_MIN)
    return np.maximum(KAPPAM_MIN, np.minimum(KAPPAM_MAX, C_K * mxl * sqrttke))


def kappa_gm_from_eke(Nsqr, eke, mask_t=None):
    """Eddy (TEM) vertical viscosity kappa_gm = K_gm min(0.01, f^2 / N^2) (90, 40, 15), on
    top faces, as eke.update_kappa_gm computes it."""
    Nsqr = np.asarray(Nsqr, dtype=float).reshape(SHAPE)
    _, _, _, m_w = face_masks(mask_t)
    K_gm = eke_diffusivity(Nsqr, eke, mask_t=mask_t)
    return K_gm * np.minimum(0.01, _F3**2 / np.maximum(1e-9, Nsqr)) * m_w


# --------------------------------------------------------------------------------------
# density and forcing
# --------------------------------------------------------------------------------------
def density(T, S, mask_t=None):
    """In-situ density rho (90, 40, 15) as thermodynamics.calc_eq_of_state computes it
    (eq_of_state_type 5, pressure = |zt| dbar), zero on land."""
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    zt, _ = make_grid_z()
    T = np.asarray(T, dtype=float).reshape(SHAPE)
    S = np.asarray(S, dtype=float).reshape(SHAPE)
    return np.asarray(gsw.gsw_rho(S, T, np.abs(zt)[np.newaxis, np.newaxis, :])) * m


def drho_dT(T, S, mask_t=None):
    """d rho / d T at fixed salinity and pressure (90, 40, 15), zero on land."""
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    zt, _ = make_grid_z()
    T = np.asarray(T, dtype=float).reshape(SHAPE)
    S = np.asarray(S, dtype=float).reshape(SHAPE)
    return np.asarray(gsw.gsw_drhodT(S, T, np.abs(zt)[np.newaxis, np.newaxis, :])) * m


def hydrostatic_pressure_matrix(mask_t=None):
    """Sparse P (54000 x 54000) with p_hydro = P @ rho (pressure / rho_0, m^2/s^2), the
    downward hydrostatic integral of solve_stream.prepare_forcing."""
    m = get_mask_t() if mask_t is None else np.asarray(mask_t, dtype=float)
    column = np.zeros((NZ, NZ))
    column[-1, -1] = 0.5 * DZW_FULL[-1]
    for k in range(NZ - 2, -1, -1):
        column[k] = column[k + 1]
        column[k, k + 1] += 0.5 * DZW[k]
        column[k, k] += 0.5 * DZW[k]
    return (_diag(m) @ sp.kron(sp.identity(N2), GRAV / RHO_0 * column)).tocsr()


def pressure_gradient_matrices(mask_t=None):
    """(PG_u, PG_v), each sparse 54000 x 54000: the tendencies of u and v due to the
    hydrostatic pressure gradient are PG_u @ rho and PG_v @ rho."""
    _, m_u, m_v, _ = face_masks(mask_t)
    P = hydrostatic_pressure_matrix(mask_t)
    PG_u = -_diag(m_u / (_COST3 * DX)) @ (_SXP - _I) @ P
    PG_v = -_diag(m_v / DY) @ (_SYP - _I) @ P
    return PG_u.tocsr(), PG_v.tocsr()


def wind_forcing(mask_t=None):
    """(f_u, f_v), each (90, 40, 15): the wind-stress tendency, in the top layer only."""
    _, m_u, m_v, _ = face_masks(mask_t)
    f_u, f_v = np.zeros(SHAPE), np.zeros(SHAPE)
    f_u[:, :, -1] = m_u[:, :, -1] * read_forcing("tau_x") / DZT[-1] / RHO_0
    f_v[:, :, -1] = m_v[:, :, -1] * read_forcing("tau_y") / DZT[-1] / RHO_0
    return f_u, f_v


# --------------------------------------------------------------------------------------
# the terms that are linear in u, v
# --------------------------------------------------------------------------------------
def coriolis_matrices(mask_t=None):
    """(C_uv, C_vu): du/dt|Coriolis = C_uv @ v, dv/dt|Coriolis = C_vu @ u."""
    _, m_u, m_v, _ = face_masks(mask_t)
    C_uv = _diag(0.25 * m_u * _F3) @ (_I + _SYM + _SXP + _SXP @ _SYM)
    f_n, cost_n = np.roll(_F3, -1, axis=1), np.roll(_COST3, -1, axis=1)
    C_vu = -0.25 * (
        _diag(m_v * _F3 * _COST3 / _COSU3) @ (_SXM + _I)
        + _diag(m_v * f_n * cost_n / _COSU3) @ _SYP @ (_SXM + _I)
    )
    return C_uv.tocsr(), C_vu.tocsr()


def horizontal_friction_matrices(mask_t=None):
    """(Fh_u, Fh_v): harmonic friction with cosine scaling (friction.harmonic_friction)."""
    _, m_u, m_v, _ = face_masks(mask_t)
    cost_n = np.roll(_COST3, -1, axis=1)

    flux_east = _diag(A_H * _COST3**COS_POWER / (_COST3 * DX) * m_u * np.roll(m_u, -1, axis=0)) @ (_SXP - _I)
    flux_north = _diag(A_H * _COSU3**COS_POWER * _COSU3 / DY * m_u * _north(m_u)) @ (_SYP - _I)
    Fh_u = _diag(m_u) @ (
        _diag(1.0 / (_COST3 * DX)) @ (_I - _SXM) @ flux_east + _diag(1.0 / (_COST3 * DY)) @ (_I - _SYM) @ flux_north
    )

    flux_east = _diag(A_H * _COSU3**COS_POWER / (_COSU3 * DX) * m_v * np.roll(m_v, -1, axis=0)) @ (_SXP - _I)
    flux_north = _diag(A_H * cost_n**COS_POWER * cost_n / DY * m_v * _north(m_v)) @ (_SYP - _I)
    Fh_v = _diag(m_v) @ (
        _diag(1.0 / (_COSU3 * DX)) @ (_I - _SXM) @ flux_east + _diag(1.0 / (_COSU3 * DY)) @ (_I - _SYM) @ flux_north
    )
    return Fh_u.tocsr(), Fh_v.tocsr()


def _implicit_step_matrices(kappa_face, m_face):
    """(NX, NY, NZ, NZ) array of the tridiagonal matrices (I - dt A) of one implicit
    vertical friction step, A the flux-form vertical diffusion with coefficient
    kappa_face on the interface above each level and no stress at top or bottom."""
    delta = np.zeros(SHAPE)
    delta[:, :, :-1] = DT_MOM / DZW * kappa_face[:, :, :-1] * m_face[:, :, 1:] * m_face[:, :, :-1]
    delta_below = np.zeros(SHAPE)
    delta_below[:, :, 1:] = delta[:, :, :-1]
    k = np.arange(NZ)
    T = np.zeros(SHAPE + (NZ,))
    T[:, :, k, k] = 1.0 + (delta + delta_below) / DZT
    T[:, :, k[:-1], k[:-1] + 1] = -delta[:, :, :-1] / DZT[:-1]
    T[:, :, k[1:], k[1:] - 1] = -delta[:, :, :-1] / DZT[1:]
    return T


def _vertical_friction_matrix(kappaM_face, kappa_gm_face, m_face):
    T_v = _implicit_step_matrices(kappaM_face, m_face)
    T_gm = _implicit_step_matrices(kappa_gm_face, m_face)
    blocks = (np.linalg.solve(T_gm, np.linalg.inv(T_v)) - np.eye(NZ)) / DT_MOM
    blocks = blocks * m_face[:, :, :, np.newaxis] * m_face[:, :, np.newaxis, :]
    rows = np.broadcast_to(IDX[:, :, :, np.newaxis], blocks.shape)
    cols = np.broadcast_to(IDX[:, :, np.newaxis, :], blocks.shape)
    keep = blocks != 0.0
    return sp.csr_matrix((blocks[keep], (rows[keep], cols[keep])), shape=(N, N))


def vertical_friction_matrices(kappaM, kappa_gm, mask_t=None):
    """(Fz_u, Fz_v): implicit vertical friction followed by implicit TEM friction, as one
    tendency (friction.implicit_vert_friction then isoneutral.isoneutral_friction)."""
    _, m_u, m_v, _ = face_masks(mask_t)
    kappaM = np.asarray(kappaM, dtype=float).reshape(SHAPE)
    kappa_gm = np.asarray(kappa_gm, dtype=float).reshape(SHAPE)

    def at_u(a):
        return 0.5 * (a + np.roll(a, -1, axis=0))

    def at_v(a):
        return 0.5 * (a + _north(a))

    Fz_u = _vertical_friction_matrix(at_u(kappaM), at_u(kappa_gm), m_u)
    Fz_v = _vertical_friction_matrix(at_v(kappaM), at_v(kappa_gm), m_v)
    return Fz_u, Fz_v


def surface_pressure_gradient_matrices(mask_t=None):
    """(G_u, G_v), each 54000 x 3600: the tendencies due to the surface pressure p_s."""
    _, m_u, m_v, _ = face_masks(mask_t)
    G_u = -_diag(m_u / (_COST3 * DX)) @ (_SXP - _I) @ _EXPAND
    G_v = -_diag(m_v / DY) @ (_SYP - _I) @ _EXPAND
    return G_u.tocsr(), G_v.tocsr()


def divergence_matrices(mask_t=None):
    """(Div_u, Div_v), each 54000 x 54000: the horizontal divergence in every T cell is
    Div_u @ u + Div_v @ v (the form used by momentum.vertical_velocity)."""
    _, m_u, m_v, _ = face_masks(mask_t)
    Div_u = _diag(1.0 / (_COST3 * DX)) @ (_I - _SXM) @ _diag(m_u)
    Div_v = _diag(1.0 / (_COST3 * DY)) @ (_I - _SYM) @ _diag(_COSU3 * m_v)
    return Div_u.tocsr(), Div_v.tocsr()


def vertical_velocity_matrices(mask_t=None):
    """(W_u, W_v): w = W_u @ u + W_v @ v, continuity integrated upwards from the bottom."""
    _, _, _, m_w = face_masks(mask_t)
    Div_u, Div_v = divergence_matrices(mask_t)
    cumsum = sp.kron(sp.identity(N2), np.tril(np.ones((NZ, NZ))))
    integrate = cumsum @ _diag(-m_w * DZT[np.newaxis, np.newaxis, :])
    return (integrate @ Div_u).tocsr(), (integrate @ Div_v).tocsr()


# --------------------------------------------------------------------------------------
# assembly and solution
# --------------------------------------------------------------------------------------
def momentum_matrices(kappaM, kappa_gm, mask_t=None):
    """All pieces of the steady momentum system as a dict of sparse matrices."""
    C_uv, C_vu = coriolis_matrices(mask_t)
    Fh_u, Fh_v = horizontal_friction_matrices(mask_t)
    Fz_u, Fz_v = vertical_friction_matrices(kappaM, kappa_gm, mask_t)
    G_u, G_v = surface_pressure_gradient_matrices(mask_t)
    Div_u, Div_v = divergence_matrices(mask_t)
    depth_sum = _EXPAND.T @ _diag(DZT[np.newaxis, np.newaxis, :])
    PG_u, PG_v = pressure_gradient_matrices(mask_t)
    return dict(
        C_uv=C_uv, C_vu=C_vu, Fh_u=Fh_u, Fh_v=Fh_v, Fz_u=Fz_u, Fz_v=Fz_v, G_u=G_u, G_v=G_v,
        D_u=(depth_sum @ Div_u).tocsr(), D_v=(depth_sum @ Div_v).tocsr(), PG_u=PG_u, PG_v=PG_v,
    )  # fmt: skip


def active_unknowns(mask_t=None):
    """Boolean selectors (u, v, p_s) of the unknowns that are solved for: wet faces, and
    the wet columns except one per connected basin, where p_s is pinned to zero (p_s is
    only defined up to a constant in each basin, and one continuity equation per basin
    is redundant)."""
    m, m_u, m_v, _ = face_masks(mask_t)
    wet = m[:, :, -1] > 0
    east = (m_u[:, :, -1] > 0).ravel()
    north = (m_v[:, :, -1] > 0).ravel()
    i2 = IDX2.ravel()
    rows = np.concatenate([i2[east], i2[north]])
    cols = np.concatenate([np.roll(IDX2, -1, axis=0).ravel()[east], np.roll(IDX2, -1, axis=1).ravel()[north]])
    graph = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(N2, N2))
    _, label = csgraph.connected_components(graph, directed=False)
    act_p = wet.ravel().copy()
    for basin in np.unique(label[wet.ravel()]):
        act_p[np.flatnonzero((label == basin) & wet.ravel())[0]] = False
    return m_u.ravel() > 0, m_v.ravel() > 0, act_p


def system_matrix(A, act_u, act_v, act_p):
    """The steady momentum system K (see the module docstring) on the active unknowns."""
    return sp.bmat(
        [
            [(A["Fz_u"] + A["Fh_u"])[act_u][:, act_u], A["C_uv"][act_u][:, act_v], A["G_u"][act_u][:, act_p]],
            [A["C_vu"][act_v][:, act_u], (A["Fz_v"] + A["Fh_v"])[act_v][:, act_v], A["G_v"][act_v][:, act_p]],
            [A["D_u"][act_p][:, act_u], A["D_v"][act_p][:, act_v], None],
        ],
        format="csc",
    )


class MomentumSolver:
    """The steady momentum system for fixed friction coefficients, factorised once, so
    that the velocities for many different density fields are cheap to obtain.

        solver = MomentumSolver(kappaM, kappa_gm)
        u, v, w, p_s = solver.solve(rho)
    """

    def __init__(self, kappaM, kappa_gm, mask_t=None, matrices=None):
        self.mask_t = mask_t
        self.A = momentum_matrices(kappaM, kappa_gm, mask_t) if matrices is None else matrices
        self.act_u, self.act_v, self.act_p = active_unknowns(mask_t)
        self.n_u, self.n_v, self.n_p = (int(a.sum()) for a in (self.act_u, self.act_v, self.act_p))
        self.K = system_matrix(self.A, self.act_u, self.act_v, self.act_p)
        self.lu = spla.splu(self.K)
        self.f_wind = tuple(f.ravel() for f in wind_forcing(mask_t))
        self.W_u, self.W_v = vertical_velocity_matrices(mask_t)

    def forcing(self, rho, extra=None):
        """(f_u, f_v), flat: wind + hydrostatic pressure gradient (+ extra)."""
        rho = np.asarray(rho, dtype=float).ravel()
        f_u = self.f_wind[0] + self.A["PG_u"] @ rho
        f_v = self.f_wind[1] + self.A["PG_v"] @ rho
        if extra is not None:
            f_u = f_u + np.asarray(extra[0], dtype=float).ravel()
            f_v = f_v + np.asarray(extra[1], dtype=float).ravel()
        return f_u, f_v

    def solve_forcing(self, f_u, f_v):
        """Flat (u, v, p_s) with K [u, v, p_s] = -[f_u, f_v, 0]."""
        x = self.lu.solve(-np.concatenate([f_u[self.act_u], f_v[self.act_v], np.zeros(self.n_p)]))
        u, v, p_s = np.zeros(N), np.zeros(N), np.zeros(N2)
        u[self.act_u], v[self.act_v], p_s[self.act_p] = x[: self.n_u], x[self.n_u : self.n_u + self.n_v], x[self.n_u + self.n_v :]
        return u, v, p_s

    def solve(self, rho, extra=None):
        """u, v, w (90, 40, 15) and p_s (90, 40) for the density field rho."""
        u, v, p_s = self.solve_forcing(*self.forcing(rho, extra))
        w = self.W_u @ u + self.W_v @ v
        return u.reshape(SHAPE), v.reshape(SHAPE), w.reshape(SHAPE), p_s.reshape(NX, NY)


def solve_velocity(rho, kappaM, kappa_gm, extra=None, mask_t=None, matrices=None):
    """
    The steady velocities for a given density field.

    rho      : in-situ density (kg/m^3), (90, 40, 15) or flat; see density(T, S).
    kappaM   : vertical viscosity on top faces (m^2/s), see kappaM_from_tke.
    kappa_gm : eddy (TEM) vertical viscosity on top faces (m^2/s), see kappa_gm_from_eke.
    extra    : optional (f_u, f_v) of extra constant tendencies (e.g. advection of momentum).
    matrices : optional result of momentum_matrices(kappaM, kappa_gm), to avoid rebuilding.

    Returns u, v, w (90, 40, 15) and p_s (90, 40).
    """
    return MomentumSolver(kappaM, kappa_gm, mask_t=mask_t, matrices=matrices).solve(rho, extra)


def streamfunction(u, v, mask_t=None):
    """
    The barotropic streamfunction psi (90, 40) of a velocity field, on the corner points
    (xu, yu) as Veros holds it, such that the depth-integrated transports are

        psi[i, j] - psi[i, j-1] = -dyt sum_k(u dzt)              through east face (i, j)
        psi[i, j] - psi[i-1, j] =  cosu[j] dxt sum_k(v dzt)      through north face (i, j)

    (the relations Veros uses to add the barotropic flow in solve_stream.py). These hold
    on every face, land included, so psi is constant along each coastline. Solved in the
    least-squares sense; exact if the depth-integrated flow is non-divergent. psi is
    defined up to a constant and returned with psi[0, 0] = 0.

    Returns (psi, residual): residual is the largest mismatch of the relations above,
    relative to the largest transport.
    """
    m, m_u, m_v, _ = face_masks(mask_t)
    t_u = -DY * ((np.asarray(u).reshape(SHAPE) * m_u) @ DZT)
    t_v = COSU[np.newaxis, :] * DX * ((np.asarray(v).reshape(SHAPE) * m_v) @ DZT)
    if np.abs(t_u[:, 0]).max() > 0:
        raise ValueError("transport through an east face of the southernmost row")

    ones = np.ones(N2)
    here = IDX2.ravel()
    south = np.roll(IDX2, 1, axis=1).ravel()
    west = np.roll(IDX2, 1, axis=0).ravel()
    rows_u = (IDX2[:, 1:]).ravel()  # east faces of rows j >= 1
    G_u = sp.csr_matrix((ones, (here, here)), shape=(N2, N2)) - sp.csr_matrix((ones, (here, south)), shape=(N2, N2))
    G_v = sp.csr_matrix((ones, (here, here)), shape=(N2, N2)) - sp.csr_matrix((ones, (here, west)), shape=(N2, N2))
    G = sp.vstack([G_u[rows_u], G_v]).tocsr()
    t = np.concatenate([t_u.ravel()[rows_u], t_v.ravel()])

    free = np.arange(1, N2)  # psi[0, 0] = 0
    psi = np.zeros(N2)
    psi[free] = spla.spsolve((G.T @ G)[free][:, free].tocsc(), (G.T @ t)[free])
    residual = np.abs(G @ psi - t).max() / max(np.abs(t).max(), 1e-300)
    return psi.reshape(NX, NY), residual
