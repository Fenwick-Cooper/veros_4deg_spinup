#!/usr/bin/env python3
"""
The steady-state temperature problem with the circulation responding to temperature.

solve_steady_state_temperature.py solves A_T T = -b with the velocities frozen. Here the
velocities are instead the steady solution of the momentum equations for the density
belonging to T (build_momentum_matrix.py), so the temperature tendency

    R(T) = [ M(u(T)) + E(T) + D33(T) + F(T) + D + R ] T + b

is a non-linear function of T alone:

    M(u(T))          advection by the velocities that T itself implies
    E, D33, F (T)    isoneutral and GM terms with the isoneutral slopes of T itself
    D                vertical mixing with the frozen kappaH
    R, b             surface relaxation and forcing

What stays frozen ("lagged") is only what comes from the two turbulence closures, which
need a model integration to update. Two choices (CoupledProblem.from_restart, `closures`):

    "coefficients"   kappaH, kappaM, K_gm and kappa_gm themselves are frozen
    "energies"       the turbulence energies (TKE, EKE) are frozen, and the four
                     coefficients are recomputed from them and the stratification of the
                     current T before every step (update_closures), which is what Veros
                     itself does on the first step after T is changed

Main pieces
-----------
    problem = CoupledProblem.from_restart(restart_file)
    u, v, w = problem.velocity(T)
    R = problem.residual(T)                       # the tendency above, degC/s, flat
    N_u, N_v, N_w = advection_sensitivity_matrices(T)   # d(M(u) T) / d(u, v, w)
"""

import os
import sys

import h5py
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_momentum_matrix as bm  # noqa: E402
from build_advection_matrix import COST, COSU, DX, DY, DZT, N, SHAPE, advection_matrix, get_mask_t  # noqa: E402
from build_gm_matrix import eke_from_restart, gm_skew_matrix, nsqr_from_restart, nsqr_from_TS  # noqa: E402
from build_isoneutral_matrix import isoneutral_explicit_matrix, isoneutral_implicit_matrix, salinity_from_restart  # noqa: E402
from build_vertical_mixing_matrix import (  # noqa: E402
    kappaH_from_restart,
    kappaH_from_tke,
    surface_forcing_vector,
    surface_relaxation_matrix,
    temperature_from_restart,
    vertical_diffusion_matrix,
)

_SZP, _SZM = bm._shift(2, 1, False), bm._shift(2, -1, False)


def advection_sensitivity_matrices(T, mask_t=None):
    """
    (N_u, N_v, N_w), each sparse 54000 x 54000: the advective temperature tendency is
    bilinear in velocity and temperature, and for a fixed T

        advection_matrix(uvw) @ T  =  N_u @ u + N_v @ v + N_w @ w.
    """
    m, m_u, m_v, m_w = bm.face_masks(mask_t)
    m_w = m_w.copy()
    m_w[:, :, -1] = 0.0  # no advective flux through the surface
    T = np.asarray(T, dtype=float).ravel()
    cost3, cosu3 = COST[np.newaxis, :, np.newaxis], COSU[np.newaxis, :, np.newaxis]

    def one(shift_plus, shift_minus, scale, face_weight):
        return (
            bm._diag(m * scale) @ (-sp.diags(shift_plus @ T) + sp.diags(shift_minus @ T) @ shift_minus) @ bm._diag(face_weight)
        ).tocsr()

    N_u = one(bm._SXP, bm._SXM, 1.0 / (2 * cost3 * DX), m_u)
    N_v = one(bm._SYP, bm._SYM, 1.0 / (2 * cost3 * DY), m_v * cosu3)
    N_w = one(_SZP, _SZM, 1.0 / (2 * DZT[np.newaxis, np.newaxis, :]), m_w)
    return N_u, N_v, N_w


class CoupledProblem:
    """Temperature tendency as a function of T alone, for fixed closure fields."""

    def __init__(self, S, kappaH, kappaM, kappa_gm, Nsqr, Ekin, mask_t=None, tke=None, K_diss_v=None):
        self._mask_arg = mask_t
        self.tke = None if tke is None else np.asarray(tke, dtype=float).reshape(SHAPE)
        self.K_diss_v = None if K_diss_v is None else np.asarray(K_diss_v, dtype=float).reshape(SHAPE)
        self.mask_t = get_mask_t() if mask_t is None else mask_t
        self.ocean = self.mask_t.ravel() > 0
        self.S = np.asarray(S, dtype=float).ravel()
        self.kappaH_closure = np.asarray(kappaH, dtype=float).ravel()
        self.kappaH = self.kappaH_closure
        self.Nsqr = np.asarray(Nsqr, dtype=float).ravel()
        self.Ekin = np.asarray(Ekin, dtype=float).ravel()
        self.momentum = bm.MomentumSolver(kappaM, kappa_gm, mask_t=mask_t)
        self.D = vertical_diffusion_matrix(self.kappaH, mask_t=self.mask_t)
        self.convecting = np.zeros((SHAPE[0], SHAPE[1], SHAPE[2] - 1), dtype=bool)

    def update_closures(self, T):
        """
        With closures = "energies" (tke given): recompute kappaH, kappaM, kappa_gm and the
        stratification that K_gm uses from the frozen TKE and EKE and the stratification
        of T, and refactorise the momentum operator. Otherwise does nothing.
        Call before update_convection.
        """
        if self.tke is None:
            return
        Nsqr = nsqr_from_TS(T, self.S, mask_t=self.mask_t)
        self.Nsqr = np.asarray(Nsqr, dtype=float).ravel()
        self.kappaH_closure = kappaH_from_tke(self.tke, Nsqr, self.K_diss_v, mask_t=self.mask_t)
        self.kappaH = self.kappaH_closure
        self.momentum = bm.MomentumSolver(
            bm.kappaM_from_tke(self.tke, Nsqr), bm.kappa_gm_from_eke(Nsqr, self.Ekin), mask_t=self._mask_arg
        )
        self.D = vertical_diffusion_matrix(self.kappaH, mask_t=self.mask_t)

    def update_convection(self, T, kappa_conv=10.0, nsqr_off=1e-8):
        """
        Convective switch standing in for the TKE closure's response to an unstable
        column, which the frozen kappaH cannot give: on every interface where T is
        statically unstable (N^2 < 0), kappaH is raised to at least kappa_conv (m^2/s).
        An interface that was switched on stays on until N^2 exceeds nsqr_off, so that
        it does not flicker. Rebuilds self.D; returns the number of interfaces switched on.
        """
        m = self.mask_t
        wet = (m[:, :, :-1] * m[:, :, 1:]) > 0
        nsqr = nsqr_from_TS(T, self.S, mask_t=m)[:, :, :-1]
        self.convecting = wet & ((nsqr < 0) | (self.convecting & (nsqr < nsqr_off)))
        kappaH = self.kappaH_closure.reshape(SHAPE).copy()
        kappaH[:, :, :-1] = np.where(self.convecting, np.maximum(kappaH[:, :, :-1], kappa_conv), kappaH[:, :, :-1])
        self.kappaH = kappaH.ravel()
        self.D = vertical_diffusion_matrix(self.kappaH, mask_t=m)
        return int((self.convecting & (self.kappaH_closure.reshape(SHAPE)[:, :, :-1] < kappa_conv)).sum())

    @classmethod
    def from_restart(cls, restart_file, closures="coefficients"):
        with h5py.File(restart_file, "r") as f:
            tau = int(f["core/tau"][()])
            tke = np.array(f["core/tke"][2:-2, 2:-2, :, tau])
            K_diss_v = np.array(f["core/K_diss_v"][2:-2, 2:-2, :])
        if closures not in ("coefficients", "energies"):
            raise ValueError(f"unknown closures {closures!r}")
        Nsqr = nsqr_from_restart(restart_file)
        Ekin = eke_from_restart(restart_file)
        return cls(
            S=salinity_from_restart(restart_file),
            kappaH=kappaH_from_restart(restart_file),
            kappaM=bm.kappaM_from_tke(tke, Nsqr),
            kappa_gm=bm.kappa_gm_from_eke(Nsqr, Ekin),
            Nsqr=Nsqr,
            Ekin=Ekin,
            tke=tke if closures == "energies" else None,
            K_diss_v=K_diss_v if closures == "energies" else None,
        )

    def velocity(self, T):
        """u, v, w (90, 40, 15): the steady circulation for temperature T."""
        u, v, w, _ = self.momentum.solve(bm.density(T, self.S, self.mask_t))
        return u, v, w

    def mixing_matrices(self, T):
        """Everything in the tendency except advection, with slopes and ice mask from T:
        (L, b) with L = E + D33 + F + D + R."""
        T = np.asarray(T, dtype=float).ravel()
        m = self.mask_t
        L = (
            isoneutral_explicit_matrix(T, self.S, mask_t=m)
            + isoneutral_implicit_matrix(T, self.S, mask_t=m)
            + gm_skew_matrix(T, self.S, self.Nsqr, self.Ekin, mask_t=m)
            + self.D
            + surface_relaxation_matrix(T, mask_t=m)
        )
        return L.tocsr(), surface_forcing_vector(T, mask_t=m)

    def advection(self, uvw):
        u, v, w = uvw
        return advection_matrix(np.concatenate([u.ravel(), v.ravel(), w.ravel()]), mask_t=self.mask_t, check_continuity=False)

    def residual(self, T, uvw=None, mixing=None):
        """
        The temperature tendency R(T) (degC/s, flat, zero on land).

        uvw    : velocities to advect with; default: those implied by T (self.velocity(T)).
        mixing : (L, b) to use; default: those built from T (self.mixing_matrices(T)).
        Passing either one freezes that part of the dependence on T.
        """
        T = np.asarray(T, dtype=float).ravel()
        uvw = self.velocity(T) if uvw is None else uvw
        L, b = self.mixing_matrices(T) if mixing is None else mixing
        return (self.advection(uvw) @ T + L @ T + b) * self.ocean

    def jacobian(self, T, uvw=None, mixing=None, dt=None):
        """
        Sparse Jacobian of the coupled problem at T, on the unknowns
        [T (ocean cells), u, v, p_s (active)]:

            | M(u) + L - I/dt    N(T)  | | dT        |   | -R(T) |
            | PG drho/dT         K     | | du, dv, dp | = |  0    |

        The first row is the temperature tendency linearised in T (at fixed velocity) and
        in the velocities (at fixed T); the second is the momentum balance linearised in
        the density. The isoneutral slopes inside L are held at those of T (their own
        dependence on T is not differentiated). dt, if given, adds a backward-Euler
        pseudo-time step of that length (s), which limits the size of the update.
        """
        T = np.asarray(T, dtype=float).ravel()
        uvw = self.velocity(T) if uvw is None else uvw
        L, _ = self.mixing_matrices(T) if mixing is None else mixing
        mom, oc = self.momentum, self.ocean
        A_T = (self.advection(uvw) + L)[oc][:, oc]
        if dt is not None:
            A_T = A_T - sp.identity(A_T.shape[0]) / dt
        N_u, N_v, N_w = advection_sensitivity_matrices(T, self.mask_t)
        N_cols = sp.hstack(
            [
                (N_u + N_w @ mom.W_u)[oc][:, mom.act_u],
                (N_v + N_w @ mom.W_v)[oc][:, mom.act_v],
                sp.csr_matrix((int(oc.sum()), mom.n_p)),
            ]
        )
        drho = sp.diags(bm.drho_dT(T, self.S, self.mask_t).ravel())
        PG_rows = sp.vstack(
            [
                (mom.A["PG_u"] @ drho)[mom.act_u][:, oc],
                (mom.A["PG_v"] @ drho)[mom.act_v][:, oc],
                sp.csr_matrix((mom.n_p, int(oc.sum()))),
            ]
        )
        return sp.bmat([[A_T, N_cols], [PG_rows, mom.K]], format="csc")


def pseudo_time_solve(
    problem, T, T_norm, dt_first=1.0, dt_max=1e5, growth=2.0, max_change=3.0, max_steps=40, tol=1e-8,
    convection=True, log=print, monitor=None, max_accepted=None, max_change_convecting=None, max_elapsed=None,
):  # fmt: skip
    """
    March the coupled problem towards its steady state with linearly-implicit
    (backward-Euler) pseudo-time steps,

        (I/dt - J) dT = R(T),      T <- T + dT

    with J the Jacobian of CoupledProblem.jacobian, so the velocities respond within the
    step. Short steps follow the physical adjustment; as dt grows this becomes Newton's
    method. dt (years) starts at dt_first and is multiplied by `growth` after every
    accepted step, up to dt_max; a step that changes any cell by more than max_change
    (degC) is rejected and retried with half the dt. If `convection`, the convective
    switch (CoupledProblem.update_convection) is re-evaluated before every step.
    If max_change_convecting is given, cells next to an interface where the convective
    switch is on are allowed to change by up to that much instead (mixing an unstable
    column is a large but legitimate change, however short the step).
    monitor(T), if given, returns a string appended to each log line.
    Stops after max_steps linear solves, after max_accepted accepted steps (if given), once
    the elapsed pseudo-time reaches max_elapsed years (if given), or when the score falls
    below tol.
    Returns (T, history); history is a list of (elapsed pseudo-time in years, dt, score).
    """
    import scipy.sparse.linalg as spla

    oc = problem.ocean
    n_oc = int(oc.sum())
    T = np.asarray(T, dtype=float).ravel().copy()

    def evaluate(T):
        problem.update_closures(T)
        n_conv = problem.update_convection(T) if convection else 0
        uvw = problem.velocity(T)
        mixing = problem.mixing_matrices(T)
        R = problem.residual(T, uvw=uvw, mixing=mixing)
        return uvw, mixing, R, 86400.0 * np.linalg.norm(R[oc]) / T_norm, n_conv

    uvw, mixing, R, score, n_conv = evaluate(T)
    elapsed, dt_year = 0.0, dt_first
    history = [(elapsed, 0.0, score)]
    log(f"   start              : score {score:.3e}  convecting {n_conv:4d}" + (f"  {monitor(T)}" if monitor else ""))
    for _ in range(max_steps):
        J = problem.jacobian(T, uvw=uvw, mixing=mixing, dt=dt_year * 360 * 86400.0)
        dT = np.zeros_like(T)
        dT[oc] = spla.spsolve(J, np.concatenate([-R[oc], np.zeros(J.shape[0] - n_oc)]))[:n_oc]
        change = np.abs(dT).max()
        too_big = not np.isfinite(change) or change > max_change
        if too_big and np.isfinite(change) and max_change_convecting is not None:
            mixing_cell = np.zeros(SHAPE, dtype=bool)
            mixing_cell[:, :, :-1] |= problem.convecting
            mixing_cell[:, :, 1:] |= problem.convecting
            quiet = np.abs(dT.reshape(SHAPE)[~mixing_cell]).max()
            too_big = quiet > max_change or change > max_change_convecting
        if too_big:
            log(f"   dt {dt_year:10.3f} years: rejected, max|dT| {change:.1f} degC")
            dt_year *= 0.5
            continue
        T = T + dT
        elapsed += dt_year
        uvw, mixing, R, score, n_conv = evaluate(T)
        history.append((elapsed, dt_year, score))
        log(
            f"   dt {dt_year:10.3f} years: score {score:.3e}  convecting {n_conv:4d}  max|dT| {change:4.2f} degC"
            + (f"  {monitor(T)}" if monitor else "")
        )
        if score < tol or (max_accepted is not None and len(history) > max_accepted):
            break
        if max_elapsed is not None and elapsed >= max_elapsed:
            break
        dt_year = min(dt_year * growth, dt_max)
    return T, history
