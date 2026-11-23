#!/usr/bin/env python3
"""
Method C: Newton-Krylov on Veros's own residual, preconditioned by the coupled emulator.

The outer structure is that of the competition spin-up (L3): rounds of [Veros for n days,
then a pseudo-time march of the temperature, then patch]. Only the march changes. In
spinup_from_init.py each pseudo-time step solves

    (J_e - I/dt) dT = -R_e(T)                                   (emulator only)

with R_e the emulator's temperature tendency and J_e its Jacobian (temperature and
circulation coupled). Here the tendency is Veros's own,

    G(T) = [ Phi_tau(P(T)) - T ] / tau,

P(T) = the round's model state with T and the emulator's circulation for T patched in,
Phi_tau = tau days of Veros. Each pseudo-time step solves

    (J_V - I/dt) dT = -G(T)

with at most k GMRES iterations, J_V v the exact derivative of G (jax.jvp through Veros, with
the circulation's response to T taken from the emulator), right-preconditioned with the LU
factors of (J_e - I/dt). k = 0 is a defect correction: dT = -(J_e - I/dt)^-1 G(T).
The fixed point is then a steady state of Veros itself (for the round's TKE/EKE), not of the
emulator. Step control (dt doubling, rejection above 3 degC / 20 degC in convecting cells,
convective switch in the preconditioner) is unchanged.

    python code/newton_krylov_anderson/nk_emulator.py --tag C_k10 --k 10
    python code/newton_krylov_anderson/nk_emulator.py --tag C_k0 --k 0
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jax_tangent as jt  # noqa: E402  (first: differentiable Veros settings)
import numpy as np  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

from common import DAYS_PER_YEAR, RESTART_DIR, Run  # noqa: E402
import coupled_temperature_velocity as ctv  # noqa: E402
from build_advection_matrix import SHAPE  # noqa: E402
from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402
from krylov import gmres  # noqa: E402
from paths import INIT_FILE  # noqa: E402

L3 = "10:30y:0.125,10:60y:2,10:60y:4,10:2000y:32"


def march(run, problem, tm, T, years, dt_first, args, say):
    """Pseudo-time steps on Veros's residual until `years` of pseudo-time are covered."""
    oc = problem.ocean
    n_oc = int(oc.sum())
    T = np.asarray(T, dtype=float).ravel().copy()
    T_norm = np.linalg.norm(T[oc]) + np.linalg.norm(temperature_from_restart(INIT_FILE)[oc])
    tau_s = args.tau * 86400.0
    year_s = DAYS_PER_YEAR * 86400.0

    def evaluate(T):
        problem.update_closures(T)
        n_conv = problem.update_convection(T)
        uvw = problem.velocity(T)
        mixing = problem.mixing_matrices(T)
        R_e = problem.residual(T, uvw=uvw, mixing=mixing)
        fields = {"temp": T.reshape(SHAPE), **jt.velocity_fields(problem, tm, T.reshape(SHAPE))}
        out = tm.evaluate(fields)
        G = (out.ravel() - T) / tau_s * oc
        return uvw, mixing, R_e, G, fields, n_conv

    uvw, mixing, R_e, G, fields, n_conv = evaluate(T)
    score = lambda r: 86400.0 * np.linalg.norm(r[oc]) / T_norm  # noqa: E731
    say(f"   start: Veros score {score(G):.3e}, emulator score {score(R_e):.3e}, convecting {n_conv}")
    elapsed, dt_year, n_solves = 0.0, dt_first, 0
    while elapsed < years and n_solves < args.max_solves:
        dt_s = dt_year * year_s
        t0 = time.time()
        # Preconditioner: the emulator's own tau-day map, taken as one backward-Euler step,
        # Phi_e = (I - tau J_e)^-1, in place of Veros's tau-day map. Then
        #     M = (Phi_e - I)/tau - I/dt   and   M^-1 = [(1 + tau/dt) J_e - I/dt]^-1 (I - tau J_e).
        # (J_e - I/dt) alone does not fit: Veros mixes implicitly, so its stiff (convecting)
        # modes saturate at -1/tau, where J_e has eigenvalues of -kappa/dz^2.
        # J_e is the Schur complement of the coupled matrix [[A_T, N], [PG, K]] (velocity
        # eliminated); both inverses are done with the block matrix.
        J0 = problem.jacobian(T, uvw=uvw, mixing=mixing).tocsr()
        c = 1.0 + tau_s / dt_s
        row_scale = np.concatenate([np.full(n_oc, c), np.ones(J0.shape[0] - n_oc)])
        shift = np.concatenate([np.full(n_oc, 1.0 / dt_s), np.zeros(J0.shape[0] - n_oc)])
        import scipy.sparse as sp

        B = sp.diags(row_scale) @ J0 - sp.diags(shift)
        lu = spla.splu(B.tocsc())
        A_T, N_blk, PG, K_lu = J0[:n_oc, :n_oc], J0[:n_oc, n_oc:], J0[n_oc:, :n_oc], problem.momentum.lu  # K = momentum.K
        n_solves += 1

        def J_e(v):
            return A_T @ v - N_blk @ K_lu.solve(PG @ v)

        def Minv(r):
            r = r - tau_s * J_e(r)
            return lu.solve(np.concatenate([r, np.zeros(J0.shape[0] - n_oc)]))[:n_oc]

        def apply_A(v_oc):
            v = np.zeros(T.size)
            v[oc] = v_oc
            tangents = {"temp": v.reshape(SHAPE), **jt.velocity_tangents(problem, tm, T.reshape(SHAPE), v.reshape(SHAPE))}
            _, dout = tm.jvp(fields, tangents)
            JV = (dout.ravel() - v) / tau_s
            return JV[oc] - v_oc / dt_s

        dT_oc, rel, its = gmres(apply_A, -G[oc], Minv, k_max=args.k, rtol=args.rtol)
        dT = np.zeros_like(T)
        dT[oc] = dT_oc
        change = np.abs(dT).max()
        too_big = not np.isfinite(change) or change > args.max_change
        if too_big and np.isfinite(change):
            mixing_cell = np.zeros(SHAPE, dtype=bool)
            mixing_cell[:, :, :-1] |= problem.convecting
            mixing_cell[:, :, 1:] |= problem.convecting
            quiet = np.abs(dT.reshape(SHAPE)[~mixing_cell]).max()
            too_big = quiet > args.max_change or change > args.max_change_convecting
        if too_big:
            say(f"   dt {dt_year:9.3f} y: rejected, max|dT| {change:.1f} degC ({time.time() - t0:.1f}s)")
            dt_year *= 0.5
            continue
        T = T + dT
        elapsed += dt_year
        uvw, mixing, R_e, G, fields, n_conv = evaluate(T)
        say(
            f"   dt {dt_year:9.3f} y: GMRES {its} its (rel {rel:.2e}), max|dT| {change:.2f}; Veros score {score(G):.3e}, "
            f"emulator score {score(R_e):.3e}, convecting {n_conv}; rms(T - T2000) {run.rms(T):.4f} ({time.time() - t0:.1f}s)"
        )
        dt_year *= args.growth
    return T


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cycles", default=L3, help="rounds 'days:years y:dt_first,...' as in spinup_from_init.py")
    ap.add_argument("--k", type=int, default=10, help="GMRES iterations per pseudo-time step (0: defect correction)")
    ap.add_argument("--rtol", type=float, default=0.05)
    ap.add_argument("--tau", type=int, default=1, help="days of Veros in the residual")
    ap.add_argument("--growth", type=float, default=2.0)
    ap.add_argument("--max-change", type=float, default=3.0)
    ap.add_argument("--max-change-convecting", type=float, default=20.0)
    ap.add_argument("--max-solves", type=int, default=30)
    ap.add_argument("--freeze-closures", action="store_true", help="derivative without the closures' response to N^2 (as the emulator)")
    args = ap.parse_args(argv)

    cycles = [(int(n), float(y.rstrip("y")), float(d)) for n, y, d in (c.split(":") for c in args.cycles.split(","))]
    run = Run(args.tag, sum(c[0] for c in cycles) + 4000)
    run.note(f"method C: {args}")
    for n_days, years, dt_first in cycles:
        run.run_main(n_days)
        restart_file = os.path.join(RESTART_DIR, f"{args.tag}_round.restart.h5")
        run.write_restart(restart_file)
        problem = ctv.CoupledProblem.from_restart(restart_file, closures="energies")
        tm = jt.TangentMap(run, args.tau, freeze_closures=args.freeze_closures)
        run.note(f"round: {years:g} pseudo-years from dt {dt_first:g}; rms before {run.rms():.4f}")
        T = march(run, problem, tm, run.temperature(), years, dt_first, args, run.note)
        run.restore(tm.base)
        problem.update_closures(T)
        run.set_temperature(T, problem.velocity(T))
        run.note(f"patched; rms after {run.rms():.4f}")
    return run.finish("C: Newton-Krylov on Veros, emulator preconditioner", vars(args))


if __name__ == "__main__":
    main()
