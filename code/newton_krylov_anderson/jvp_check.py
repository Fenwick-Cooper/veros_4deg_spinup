"""Checks of jax_tangent.py: speed of Veros with the differentiable solvers; the JAX patch
against patch_state; jax.jvp against finite differences; cost of a jvp."""
import os, time
import jax_tangent as jt  # first: selects the differentiable solvers
import numpy as np
from common import Run, RESTART_DIR
import coupled_temperature_velocity as ctv

run = Run("jvp_check", 4000)
t = time.time(); run.run_main(30); print(f"differentiable settings: {(time.time() - t) / 30:.4f} s per Veros day")
T = run.temperature()
mask = run.mask
d = np.where(mask, run.T2000 - T, 0.0); d /= np.abs(d).max()          # smooth, large-scale
rng = np.random.default_rng(0); r = np.where(mask, rng.standard_normal(T.shape), 0.0); r /= np.abs(r).max()

for tau in (1, 10):
    tm = jt.TangentMap(run, tau)
    snap = run.snapshot()
    run.set_temperature(T); ref = run.run_aux(tau); run.restore(snap)
    out = tm.evaluate({"temp": T})
    print(f"tau {tau}: JAX patch vs patch_state, max |diff| after tau days {np.abs(out - ref)[mask].max():.2e}")
    for name, v in (("smooth", d), ("random", r)):
        t = time.time(); o, jv = tm.jvp({"temp": T}, {"temp": v}); t1 = time.time() - t
        t = time.time(); o, jv = tm.jvp({"temp": T}, {"temp": v}); t2 = time.time() - t
        line = f"  {name:6s}: jvp {t1:.1f}s first, {t2:.2f}s again; |Jv| {np.linalg.norm(jv[mask]):.3e}"
        for h in (1e-5, 1e-3):
            fd = (tm.evaluate({"temp": T + h * v}) - o) / h
            line += f"; FD h={h:g} rel err {np.linalg.norm((fd - jv)[mask]) / np.linalg.norm(jv[mask]):.2e}"
        print(line, flush=True)

# with the emulator's circulation in the patch
f = os.path.join(RESTART_DIR, "jvp_check.restart.h5"); run.write_restart(f)
problem = ctv.CoupledProblem.from_restart(f, closures="energies"); problem.update_closures(T)
tm = jt.TangentMap(run, 1)
fields = {"temp": T, **jt.velocity_fields(problem, tm, T)}
t = time.time(); tan = {"temp": d, **jt.velocity_tangents(problem, tm, T, d)}; tv = time.time() - t
o, jv = tm.jvp(fields, tan)
t = time.time(); o, jv = tm.jvp(fields, tan); t2 = time.time() - t
h = 1e-3
fields_h = {"temp": T + h * d, **jt.velocity_fields(problem, tm, T + h * d)}
fd = (tm.evaluate(fields_h) - o) / h
print(f"with circulation, tau 1: velocity tangent {tv:.2f}s, jvp {t2:.2f}s; FD h=1e-3 rel err {np.linalg.norm((fd - jv)[mask]) / np.linalg.norm(jv[mask]):.2e}")
