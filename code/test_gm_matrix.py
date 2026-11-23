#!/usr/bin/env python3
"""
Test build_gm_matrix.py against Veros.

For each restart file, the global 4-degree model is set up from that restart exactly
as Example_run_Veros.py does, and the following are compared with Veros's own routines:
  1. coriolis_t, beta, N^2 and K_gm against eke.set_eke_diffusivities;
  2. one full GM skew-diffusion step against
     isoneutral.isoneutral_skew_diffusion(state, tr=temp, istemp=True), for temperature
     and salinity (the same cross-tracer check used for the Redi term).

Run with the Python environment in which Veros is installed:

    python test_gm_matrix.py [restart_file ...]
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
RTOL = 1e-8

sys.path.insert(0, HERE)
from build_advection_matrix import SHAPE, get_mask_t  # noqa: E402
from build_gm_matrix import BETA, CORIOLIS_T, eke_diffusivity, eke_from_restart, gm_skew_matrix, nsqr_from_restart  # noqa: E402
from build_isoneutral_matrix import salinity_from_restart  # noqa: E402
from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402


def veros_model(restart_file):
    from global_4deg_no_cycle import GlobalFourDegreeSetup

    workdir = tempfile.mkdtemp(dir=INTERMEDIATE_DIR, prefix="veros_gm_test_")
    for name in ("forcing_4deg_global_annual_mean.nc", "ecmwf_4deg_monthly_nc4.nc"):
        os.symlink(os.path.join(INPUT_DIR, name), os.path.join(workdir, name))
    os.symlink(restart_file, os.path.join(workdir, "restart.h5"))
    os.chdir(workdir)

    model = GlobalFourDegreeSetup("gm_matrix_test", 86400.0, "./restart.h5", restart_mode=True)
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


def veros_eke_fields(state):
    from veros import veros_routine
    from veros.core import eke

    result = {}

    @veros_routine
    def run(state):
        eke.set_eke_diffusivities(state)
        vs = state.variables
        for name in ("coriolis_t", "beta", "Nsqr", "K_gm"):
            result[name] = np.asarray(getattr(vs, name))

    run(state)
    return result


def veros_gm_diffusion(state, istemp=True):
    """One full GM skew-diffusion step of temp (istemp=True) or salt (istemp=False)."""
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
        isoneutral.isoneutral_skew_diffusion(state, tr=tr, istemp=istemp)
        result["field"] = np.asarray((vs.temp if istemp else vs.salt)[..., taup1])

    run(state)
    return interior(result["field"])


def test_restart(restart_file):
    print(f"\n=== {os.path.basename(restart_file)} ===")
    model, workdir = veros_model(restart_file)
    state = model.state
    try:
        vs = state.variables
        dt = float(state.settings.dt_tracer)
        m = get_mask_t()
        ok = True

        T = temperature_from_restart(restart_file)
        S = salinity_from_restart(restart_file)
        Nsqr = nsqr_from_restart(restart_file)
        E = eke_from_restart(restart_file)

        veros_eke = veros_eke_fields(state)
        ok &= report("coriolis_t", CORIOLIS_T, interior(veros_eke["coriolis_t"])[0, :])
        ok &= report("beta", BETA, interior(veros_eke["beta"])[0, :])
        ok &= report("N^2 matches restart", Nsqr, interior(veros_eke["Nsqr"])[..., int(vs.tau)].ravel(), rtol=0)

        K_gm = eke_diffusivity(Nsqr, E)
        ok &= report("K_gm", K_gm, interior(veros_eke["K_gm"]))

        F = gm_skew_matrix(T, S, Nsqr, E)
        print(f"  F: {F.nnz} non-zeros")
        land = m.ravel() == 0
        ok &= check("land rows/cols of F zero", abs(F[:, land]).sum() == 0 and abs(F[land, :]).sum() == 0)
        from build_advection_matrix import COST, DX, DY, DZT

        cell_vol = (COST[np.newaxis, :, np.newaxis] * DX * DY * DZT[np.newaxis, np.newaxis, :] * np.ones(SHAPE)).ravel()
        col_sum = np.asarray(F.multiply(cell_vol[:, None]).sum(axis=0)).ravel()
        ok &= check("F conserves heat (cell-volume-weighted column sums = 0)", np.abs(col_sum).max() <= 1e-9 * np.abs(F.multiply(cell_vol[:, None])).max())

        for name, istemp, X in (("temp", True, T), ("salt", False, S)):
            T0 = interior((vs.temp if istemp else vs.salt)[..., int(vs.taup1)]).ravel()
            Y = veros_gm_diffusion(state, istemp=istemp)
            expected = T0 + dt * (F @ X.ravel())  # GM skew is purely explicit: no implicit solve
            ok &= report(f"GM skew step, {name}", Y.ravel(), expected)

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
