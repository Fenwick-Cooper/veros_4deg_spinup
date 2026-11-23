#!/usr/bin/env python3
"""
Test build_momentum_matrix.py against Veros itself, using the settled states saved by
capture_momentum_state.py (momentum_state_<case>.npz).

1. Grid, masks and forcing rebuilt here versus Veros's own arrays.
2. The friction coefficients kappaM and kappa_gm recomputed from tke, eke and Nsqr.
3. Each term of the u, v equations: matrix @ (Veros's u, v) versus the tendency Veros
   computed for that term.
4. The steady solve: velocities from solve_velocity versus Veros's settled velocities.

Run with the Python environment in which Veros is installed:
    python test_momentum_matrix.py [a|b ...]
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401
sys.path.insert(0, HERE)

import build_momentum_matrix as bm  # noqa: E402
from build_advection_matrix import SHAPE  # noqa: E402


def interior(a):
    return np.asarray(a)[2:-2, 2:-2]


def rel(mine, veros):
    """rms(mine - veros) / rms(veros), over the points where either is non-zero."""
    mine, veros = np.asarray(mine).reshape(veros.shape), np.asarray(veros)
    wet = (mine != 0) | (veros != 0)
    if not wet.any():
        return 0.0
    return float(np.sqrt(np.mean((mine - veros)[wet] ** 2)) / max(np.sqrt(np.mean(veros[wet] ** 2)), 1e-300))


def test_case(case):
    d = np.load(os.path.join(INTERMEDIATE_DIR, f"momentum_state_{case}.npz"))
    print(f"\n================ case {case} ================")
    m, m_u, m_v, m_w = bm.face_masks()
    u, v, w = (interior(d[n]) for n in ("u", "v", "w"))
    rho, T, S = (interior(d[n]) for n in ("rho", "temp", "salt"))

    print("1. grid, masks, forcing (max abs difference unless stated)")
    for name, mine in (("maskT", m), ("maskU", m_u), ("maskV", m_v), ("maskW", m_w)):
        print(f"   {name:12s} {np.abs(mine - interior(d[name])).max():.1e}")
    print(f"   {'coriolis_t':12s} {np.abs(bm.CORIOLIS_T[np.newaxis, :] - interior(d['coriolis_t'])).max():.1e}")
    print(f"   {'A_h':12s} relative {abs(bm.A_H - d['A_h']) / d['A_h']:.1e}")
    f_u, f_v = bm.wind_forcing()
    print(f"   {'wind u, v':12s} relative {rel(f_u, interior(d['du_wind'])):.1e}  {rel(f_v, interior(d['dv_wind'])):.1e}")
    print(f"   {'rho(T, S)':12s} relative to rho - 1000: {rel(bm.density(T, S) - 1000 * m, rho - 1000 * m):.1e}")

    print("2. friction coefficients recomputed from the closures' state (relative rms)")
    kappaM = bm.kappaM_from_tke(interior(d["tke"]), interior(d["Nsqr"]))
    kappa_gm = bm.kappa_gm_from_eke(interior(d["Nsqr"]), interior(d["eke"]))
    print(f"   kappaM   {rel(kappaM * m_w, interior(d['kappaM']) * m_w):.1e}")
    print(f"   kappa_gm {rel(kappa_gm, interior(d['kappa_gm'])):.1e}")

    print("3. each term, matrix @ (u, v) versus Veros (relative rms)        u-eq      v-eq")
    A = bm.momentum_matrices(interior(d["kappaM"]), interior(d["kappa_gm"]))
    uf, vf = u.ravel(), v.ravel()
    print(f"   Coriolis (Veros's value includes the metric terms)       {rel(A['C_uv'] @ vf, interior(d['du_cor'])):.1e}   {rel(A['C_vu'] @ uf, interior(d['dv_cor'])):.1e}")
    print(f"   friction (vertical + TEM + horizontal)                   {rel((A['Fz_u'] + A['Fh_u']) @ uf, interior(d['du_mix'])):.1e}   {rel((A['Fz_v'] + A['Fh_v']) @ vf, interior(d['dv_mix'])):.1e}")
    print(f"   hydrostatic pressure gradient, from Veros's rho          {rel(A['PG_u'] @ rho.ravel(), interior(d['du_pg'])):.1e}   {rel(A['PG_v'] @ rho.ravel(), interior(d['dv_pg'])):.1e}")
    rho_T = bm.density(T, S).ravel()
    print(f"   hydrostatic pressure gradient, from rho(T, S)            {rel(A['PG_u'] @ rho_T, interior(d['du_pg'])):.1e}   {rel(A['PG_v'] @ rho_T, interior(d['dv_pg'])):.1e}")
    W_u, W_v = bm.vertical_velocity_matrices()
    print(f"   w from continuity                                        {rel(W_u @ uf + W_v @ vf, w):.1e}")
    print(f"   depth-integrated divergence of Veros's (u, v), max abs / (max|u| H / dx): "
          f"{np.abs(A['D_u'] @ uf + A['D_v'] @ vf).max() / (np.abs(u).max() * 5000 / bm.DX):.1e}")
    adv = np.sqrt(np.mean(interior(d["du_adv"]) ** 2)) / np.sqrt(np.mean(interior(d["du_cor"]) ** 2))
    print(f"   (size of momentum advection relative to Coriolis, u-eq: {adv:.1e})")

    print("4. steady solve versus Veros's settled state (relative rms)          u        v        w")

    def report(label, sol):
        print(f"   {label:58s} {rel(sol[0], u):.1e}  {rel(sol[1], v):.1e}  {rel(sol[2], w):.1e}")

    report("Veros's rho, kappaM, kappa_gm", bm.solve_velocity(rho, interior(d["kappaM"]), interior(d["kappa_gm"]), matrices=A))
    extra = (
        interior(d["du_adv"]) + interior(d["du_cor"]) - (A["C_uv"] @ vf).reshape(SHAPE),
        interior(d["dv_adv"]) + interior(d["dv_cor"]) - (A["C_vu"] @ uf).reshape(SHAPE),
    )
    report("  + Veros's advection and metric terms as extra forcing", bm.solve_velocity(rho, interior(d["kappaM"]), interior(d["kappa_gm"]), extra=extra, matrices=A))
    sol = bm.solve_velocity(rho_T, kappaM, kappa_gm)
    report("everything recomputed: rho(T, S), kappaM(tke), kappa_gm(eke)", sol)
    by_level = [rel(sol[0][:, :, k], u[:, :, k]) for k in range(SHAPE[2] - 1, -1, -1)]
    print("     the last line, u by level, surface ... bottom: " + " ".join(f"{x:.0e}" for x in by_level))
    return dict(u=u, v=v, w=w, sol=sol, T=T, S=S, rho=rho_T.reshape(SHAPE), kappaM=kappaM, kappa_gm=kappa_gm)


def main():
    cases = sys.argv[1:] or ["a", "b"]
    results = {c: test_case(c) for c in cases}
    if "a" in results and "b" in results:
        print("\n================ response to a change of temperature (b minus a) ================")
        ra, rb = results["a"], results["b"]
        for i, name in enumerate("uvw"):
            mine = rb["sol"][i] - ra["sol"][i]
            veros = rb[name] - ra[name]
            print(f"   {name}: operator's change versus Veros's change, relative rms = {rel(mine, veros):.1e}")

        print("\n   predicting the circulation of b from state a (error relative to Veros's change b - a):   u        v        w")

        def report(label, sol):
            errs = [rel(sol[i] - ra[n], rb[n] - ra[n]) for i, n in enumerate("uvw")]
            print(f"   {label:76s} " + "  ".join(f"{e:.1e}" for e in errs))

        report("no change at all (velocities of a)", (ra["u"], ra["v"], ra["w"]))
        report("density of b, friction coefficients kept from a", bm.solve_velocity(rb["rho"], ra["kappaM"], ra["kappa_gm"]))
        rho_lin = ra["rho"] + bm.drho_dT(ra["T"], ra["S"]) * (rb["T"] - ra["T"])
        report("density linearised about a (drho/dT), friction coefficients kept from a", bm.solve_velocity(rho_lin, ra["kappaM"], ra["kappa_gm"]))
        report("density linearised about a (drho/dT), friction coefficients of b", bm.solve_velocity(rho_lin, rb["kappaM"], rb["kappa_gm"]))


if __name__ == "__main__":
    main()
