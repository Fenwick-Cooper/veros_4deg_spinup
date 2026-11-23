#!/usr/bin/env python3
"""
Test build_combined_step.py against Veros.

For each restart file, the global 4-degree model is set up from that restart exactly
as Example_run_Veros.py does, one full Veros time step is run (model.step(), which
calls thermodynamics.thermodynamics among other things), and the resulting
temp[...,taup1] is compared with build_combined_step.temperature_step's output, built
entirely from the restart's own state (and dt_tracer, supplied only to temperature_step).

Run with the Python environment in which Veros is installed:

    python test_combined_step.py [restart_file ...]
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
RTOL = 1e-6

sys.path.insert(0, HERE)
from build_advection_matrix import SHAPE, get_mask_t  # noqa: E402
from build_combined_step import (  # noqa: E402
    combined_matrices,
    dsalt_taum1_from_restart,
    dtemp_taum1_from_restart,
    salinity_step,
    temperature_step,
)
from build_gm_matrix import eke_from_restart, nsqr_from_restart  # noqa: E402
from build_isoneutral_matrix import salinity_from_restart  # noqa: E402
from build_vertical_mixing_matrix import kappaH_from_restart, temperature_from_restart  # noqa: E402


def veros_model(restart_file):
    from global_4deg_no_cycle import GlobalFourDegreeSetup

    workdir = tempfile.mkdtemp(dir=INTERMEDIATE_DIR, prefix="veros_combined_test_")
    for name in ("forcing_4deg_global_annual_mean.nc", "ecmwf_4deg_monthly_nc4.nc"):
        os.symlink(os.path.join(INPUT_DIR, name), os.path.join(workdir, name))
    os.symlink(restart_file, os.path.join(workdir, "restart.h5"))
    os.chdir(workdir)

    model = GlobalFourDegreeSetup("combined_step_test", 86400.0, "./restart.h5", restart_mode=True)
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


def test_restart(restart_file):
    print(f"\n=== {os.path.basename(restart_file)} ===")
    model, workdir = veros_model(restart_file)
    state = model.state
    try:
        vs = state.variables
        dt = float(state.settings.dt_tracer)
        taup1 = int(vs.taup1)
        m = get_mask_t()
        ok = True

        T = temperature_from_restart(restart_file)
        S = salinity_from_restart(restart_file)
        from build_advection_matrix import uvw_from_restart

        uvw = uvw_from_restart(restart_file)
        kappaH = kappaH_from_restart(restart_file)
        Nsqr = nsqr_from_restart(restart_file)
        Ekin = eke_from_restart(restart_file)
        g_T = dtemp_taum1_from_restart(restart_file)
        g_S = dsalt_taum1_from_restart(restart_file)

        matrices = combined_matrices(T, S, uvw, kappaH, Nsqr, Ekin, mask_t=m)
        for name in ("M", "E", "D33", "F", "D", "R"):
            print(f"  {name}: {matrices[name].nnz} non-zeros")

        T_next = temperature_step(matrices, T, g_T, dt=dt)
        S_next = salinity_step(matrices, S, g_S, dt=dt)

        model.step(state)  # one full Veros step: set_forcing, eke, tke, momentum, thermodynamics, ...
        T_veros = interior(vs.temp[..., taup1]).ravel()
        S_veros = interior(vs.salt[..., taup1]).ravel()

        ok &= report("combined-matrix step matches Veros, temp", T_next, T_veros)
        ok &= report("combined-matrix step matches Veros, salt", S_next, S_veros)

        land = m.ravel() == 0
        ok &= check("T_next unchanged on land", np.allclose(T_next[land], T[land]))
        ok &= check("S_next unchanged on land", np.allclose(S_next[land], S[land]))

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
