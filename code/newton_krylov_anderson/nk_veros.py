#!/usr/bin/env python3
"""
Method B: Newton-Krylov on Veros alone (after Bernsen, Dijkstra & Wubs 2008; Merlis &
Khatiwala 2008), with no emulator and no preconditioner.

Residual G(T) = [Phi_tau(T) - T] / tau, Phi_tau = tau days of Veros from the current model
state with T patched in (temperature only, as in method A). Each step solves, with at most
k GMRES iterations,

    (J - I/dt) dT = -G(T),

a pseudo-time (Psi-tc) Newton step whose dt grows by `growth` each step; the update is
scaled down if any cell would change by more than --max-change. J v is the exact derivative
by jax.jvp through Veros (finite differences are unusable here: convection makes them
inconsistent by 1-50 %, see jvp_check.py). The tau days of the residual are the main
trajectory; the Jacobian products are thrown away (but counted in days_total).

    python code/newton_krylov_anderson/nk_veros.py --tag B_tau360 --tau 360 --k 20
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jax_tangent as jt  # noqa: E402  (first: differentiable Veros settings)
import numpy as np  # noqa: E402

from common import DAYS_PER_YEAR, Run  # noqa: E402
from build_advection_matrix import SHAPE  # noqa: E402
from krylov import gmres  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tau", type=int, default=360)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--rtol", type=float, default=0.1)
    ap.add_argument("--dt-first", type=float, default=10.0, help="first pseudo-time step, years")
    ap.add_argument("--growth", type=float, default=2.0)
    ap.add_argument("--max-change", type=float, default=3.0)
    ap.add_argument("--max-years", type=float, default=300.0)
    ap.add_argument("--max-wall", type=float, default=4500.0)
    args = ap.parse_args(argv)

    run = Run(args.tag, int(args.max_years * DAYS_PER_YEAR) + 4000, log_every=10)
    run.note(f"method B: {args}")
    tau_s = args.tau * 86400.0
    dt_year = args.dt_first
    mask = run.mask
    step = 0
    while run.days_total < args.max_years * DAYS_PER_YEAR and run.wall() < args.max_wall:
        T = run.temperature()
        tm = jt.TangentMap(run, args.tau)  # base: the current state
        run.run_main(args.tau)
        fx = run.temperature()
        after = run.snapshot()
        run.restore(tm.base)
        G = ((fx - T) / tau_s)[mask]
        dt_s = dt_year * DAYS_PER_YEAR * 86400.0

        def apply_A(v_oc):
            v = np.zeros(SHAPE)
            v[mask] = v_oc
            _, dout = tm.jvp({"temp": T}, {"temp": v})
            return ((dout - v) / tau_s)[mask] - v_oc / dt_s

        dT_oc, rel, its = gmres(apply_A, -G, None, k_max=args.k, rtol=args.rtol)
        big = np.abs(dT_oc).max()
        if big > args.max_change:
            dT_oc *= args.max_change / big
        T_new = fx.copy()
        T_new[mask] = T[mask] + dT_oc
        run.restore(after)
        run.set_temperature(T_new)
        step += 1
        run.note(
            f"step {step}: dt {dt_year:g} y, |G| {np.linalg.norm(G):.3e} degC/s, GMRES {its} its (rel {rel:.2e}), "
            f"max|dT| {big:.2f}{' (scaled)' if big > args.max_change else ''}; rms Phi(T) {run.rms(fx):.4f}, rms new {run.rms():.4f}"
        )
        dt_year *= args.growth
    return run.finish("B: Newton-Krylov on Veros alone (exact jvp, no preconditioner)", vars(args))


if __name__ == "__main__":
    main()
