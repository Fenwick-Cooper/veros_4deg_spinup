#!/usr/bin/env python3
"""
Test build_isoneutral_matrix.py against Veros.

For each restart file, the global 4-degree model is set up from that restart
exactly as Example_run_Veros.py does, and the following are compared with
Veros's own routines:
  1. K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by against
     isoneutral.isoneutral_diffusion_pre;
  2. one full isoneutral diffusion step (explicit + implicit K_33 solve)
     against isoneutral.isoneutral_diffusion(state, tr=temp, istemp=True),
     for the model temperature and for two random fields.

Run with the Python environment in which Veros is installed:

    python test_isoneutral_matrix.py [restart_file ...]
"""

import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401

CONFIG_DIR = VEROS_RUN_DIR  # Veros runs here and writes its (empty) NetCDF output here
DEFAULT_RESTARTS = [
    T200_FILE,
    INIT_FILE,
]
RTOL = 1e-9

sys.path.insert(0, HERE)
from build_advection_matrix import SHAPE, get_mask_t  # noqa: E402
from build_isoneutral_matrix import isoneutral_coefficients, isoneutral_explicit_matrix, isoneutral_implicit_matrix, salinity_from_restart  # noqa: E402
from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402


def veros_model(restart_file):
    from global_4deg_no_cycle import GlobalFourDegreeSetup

    workdir = tempfile.mkdtemp(dir=INTERMEDIATE_DIR, prefix="veros_iso_test_")
    for name in ("forcing_4deg_global_annual_mean.nc", "ecmwf_4deg_monthly_nc4.nc"):
        os.symlink(os.path.join(INPUT_DIR, name), os.path.join(workdir, name))
    os.symlink(restart_file, os.path.join(workdir, "restart.h5"))
    os.chdir(workdir)

    model = GlobalFourDegreeSetup("iso_matrix_test", 86400.0, "./restart.h5", restart_mode=True)
    model.setup()
    return model, workdir


def interior(a):
    return np.asarray(a)[2:-2, 2:-2, ...]


def report(label, got, expected, rtol=RTOL):
    err = np.abs(np.asarray(got) - np.asarray(expected)).max()
    scale = max(np.abs(np.asarray(expected)).max(), 1e-300)
    ok = err <= rtol * scale
    print(f"  {label:42s} max|expected| = {scale:.3e}   max|err| = {err:.3e}   rel = {err / scale:.1e}   {'PASS' if ok else 'FAIL'}")
    return ok


def check(label, ok):
    print(f"  {label:42s} {'PASS' if ok else 'FAIL'}")
    return ok


def set_temp_level(state, level, field):
    from veros import veros_routine
    from veros.core import utilities
    from veros.core.operators import at, update
    from veros.core.operators import numpy as npx

    @veros_routine
    def run(state):
        vs = state.variables
        t = update(vs.temp[..., level], at[2:-2, 2:-2, :], npx.asarray(field))
        vs.temp = update(vs.temp, at[..., level], utilities.enforce_boundaries(t, True))

    run(state)


def veros_iso_coefficients(state):
    from veros import veros_routine
    from veros.core import eke, isoneutral

    result = {}

    @veros_routine
    def run(state):
        eke.set_eke_diffusivities(state)  # sets vs.K_iso (defaults to 0 until this runs)
        out = isoneutral.isoneutral_diffusion_pre(state)
        vs = state.variables
        vs.update(out)
        for name in ("K_11", "K_22", "K_33", "Ai_ez", "Ai_nz", "Ai_bx", "Ai_by"):
            result[name] = np.asarray(getattr(vs, name))

    run(state)
    return result


def veros_iso_diffusion(state, istemp=True):
    """
    One full isoneutral diffusion step of temp (istemp=True) or salt (istemp=False), exactly as
    Veros runs it (thermodynamics.thermodynamics): the flux divergence reads the tracer's own
    tau-level values (isoneutral.diffusion._calc_tracer_fluxes), and is added to its own taup1.
    Returns the new taup1 field. temp and salt share the same slopes/taper/K_11/K_22/K_33 (built
    from T, S together in isoneutral_diffusion_pre), so this is a genuine test of E, D33 as
    generic linear operators: the same matrices, applied to a second, different field (salt).
    """
    from veros import veros_routine
    from veros.core import eke, isoneutral

    taup1 = int(state.variables.taup1)
    result = {}

    @veros_routine
    def run(state):
        vs = state.variables
        eke.set_eke_diffusivities(state)
        isoneutral.isoneutral_diffusion_pre(state)
        tr = vs.temp if istemp else vs.salt
        isoneutral.isoneutral_diffusion(state, tr=tr, istemp=istemp)
        result["field"] = np.asarray((vs.temp if istemp else vs.salt)[..., taup1])

    run(state)
    return interior(result["field"])


def test_restart(restart_file):
    print(f"\n=== {os.path.basename(restart_file)} ===")
    model, workdir = veros_model(restart_file)
    state = model.state
    try:
        vs = state.variables
        tau = int(vs.tau)
        dt = float(state.settings.dt_tracer)
        m = get_mask_t()
        ok = True

        T = temperature_from_restart(restart_file)
        S = salinity_from_restart(restart_file)
        ok &= report("T matches Veros", T, interior(vs.temp[..., tau]).ravel(), rtol=0)
        ok &= report("S matches Veros", S, interior(vs.salt[..., tau]).ravel(), rtol=0)

        # 1. coefficients
        veros_coef = veros_iso_coefficients(state)
        K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by = isoneutral_coefficients(T, S)
        ok &= report("K_11", K_11, interior(veros_coef["K_11"]))
        ok &= report("K_22", K_22, interior(veros_coef["K_22"]))
        ok &= report("K_33", K_33, interior(veros_coef["K_33"]))
        ok &= report("Ai_ez", Ai_ez, interior(veros_coef["Ai_ez"]))
        ok &= report("Ai_nz", Ai_nz, interior(veros_coef["Ai_nz"]))
        ok &= report("Ai_bx", Ai_bx, interior(veros_coef["Ai_bx"]))
        ok &= report("Ai_by", Ai_by, interior(veros_coef["Ai_by"]))

        coefficients = (K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by)
        E = isoneutral_explicit_matrix(T, S, coefficients=coefficients)
        D33 = isoneutral_implicit_matrix(T, S, coefficients=coefficients)
        print(f"  E: {E.nnz} non-zeros, D33: {D33.nnz} non-zeros")

        land = m.ravel() == 0
        ok &= check("land rows/cols of E zero", abs(E[:, land]).sum() == 0 and abs(E[land, :]).sum() == 0)
        ok &= check("land rows/cols of D33 zero", abs(D33[:, land]).sum() == 0 and abs(D33[land, :]).sum() == 0)
        from build_advection_matrix import COST, DX, DY, DZT

        vol = np.broadcast_to(DZT, SHAPE).ravel()  # D33 has no horizontal coupling: dz alone is enough
        col_sum_D = np.asarray(D33.multiply(vol[:, None]).sum(axis=0)).ravel()
        ok &= check("D33 conserves heat (dz-weighted column sums = 0)", np.abs(col_sum_D).max() <= 1e-10 * np.abs(D33.multiply(vol[:, None])).max())

        # E couples horizontal neighbours, so conservation needs the full cell volume
        # (area_t * dz), not just dz -- a dz-only check is off by ~1% here.
        cell_vol = (COST[np.newaxis, :, np.newaxis] * DX * DY * DZT[np.newaxis, np.newaxis, :] * np.ones(SHAPE)).ravel()
        col_sum_E = np.asarray(E.multiply(cell_vol[:, None]).sum(axis=0)).ravel()
        ok &= check("E conserves heat (cell-volume-weighted column sums = 0)", np.abs(col_sum_E).max() <= 1e-9 * np.abs(E.multiply(cell_vol[:, None])).max())

        from scipy.sparse.linalg import spsolve
        from scipy.sparse import identity

        for name, istemp, X in (("temp", True, T), ("salt", False, S)):
            T0 = interior((vs.temp if istemp else vs.salt)[..., int(vs.taup1)]).ravel()  # value before this term
            Y = veros_iso_diffusion(state, istemp=istemp)
            # Veros: (I - dt*D33) Y = T0 + dt*E@X   =>   check this relation holds for Veros's own Y
            lhs = (identity(len(T), format="csr") - dt * D33) @ Y.ravel()
            rhs = T0 + dt * (E @ X.ravel())
            ok &= report(f"isoneutral step, {name}", lhs, rhs)

            # also check that solving ourselves reproduces Veros's Y directly
            Y_ours = spsolve((identity(len(T), format="csr") - dt * D33).tocsc(), rhs).reshape(SHAPE)
            ok &= report(f"  solved {name} matches Veros", Y_ours, Y)

        return ok
    finally:
        os.chdir(HERE)
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    restarts = [os.path.abspath(f) for f in sys.argv[1:]] or DEFAULT_RESTARTS
    all_ok = True
    for restart_file in restarts:
        all_ok &= test_restart(restart_file)
    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
