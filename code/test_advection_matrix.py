#!/usr/bin/env python3
"""
Test that advection_matrix(uvw) reproduces the Veros advection tendency.

For each restart file, the global 4-degree model is set up from that restart
exactly as Example_run_Veros.py does. M = advection_matrix(uvw) is built from
the restart's u, v, w (current time level tau), and M @ T is compared with
Veros's own advect_tracer for
  1. the model temperature in the restart file, and
  2. several random tracer fields (random values on land and in the halos too,
     to check that land and boundary cells are handled correctly).
It also checks that the stored advection_matrix.npz equals advection_matrix()
for the restart it was built from.

Run with the Python environment in which Veros is installed:

    python test_advection_matrix.py [restart_file ...]

Each model setup runs in a temporary directory, so no output files are left in
spin-up-veros-config.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401

CONFIG_DIR = VEROS_RUN_DIR  # Veros runs here and writes its (empty) NetCDF output here
DEFAULT_RESTARTS = [
    T200_FILE,
    INIT_FILE,
]
RTOL = 1e-10  # max |error| allowed, relative to max |tendency|

sys.path.insert(0, HERE)
from build_advection_matrix import SHAPE, advection_matrix, get_mask_t, make_grid_z, uvw_from_restart  # noqa: E402


def veros_state(restart_file):
    """Set up the model from restart_file as Example_run_Veros.py does; return (state, workdir)."""
    from global_4deg_no_cycle import GlobalFourDegreeSetup  # sets VEROS_BACKEND=jax

    workdir = tempfile.mkdtemp(dir=INTERMEDIATE_DIR, prefix="veros_adv_test_")
    for name in ("forcing_4deg_global_annual_mean.nc", "ecmwf_4deg_monthly_nc4.nc"):
        os.symlink(os.path.join(INPUT_DIR, name), os.path.join(workdir, name))
    os.symlink(restart_file, os.path.join(workdir, "restart.h5"))
    os.chdir(workdir)

    model = GlobalFourDegreeSetup("adv_matrix_test", 86400.0, "./restart.h5", restart_mode=True)
    model.setup()
    return model.state, workdir


def veros_advection(state, tracer_full):
    """Veros's own advective tendency for a full (94, 44, 15) tracer field."""
    from veros import veros_routine
    from veros.core import thermodynamics
    from veros.core.operators import numpy as npx

    result = {}

    @veros_routine
    def run(state):
        result["dtr"] = np.asarray(thermodynamics.advect_tracer(state, npx.asarray(tracer_full)))

    run(state)
    return result["dtr"][2:-2, 2:-2, :]


def compare(name, M, tracer_full, state):
    expected = veros_advection(state, tracer_full)
    got = (M @ np.asarray(tracer_full)[2:-2, 2:-2, :].ravel()).reshape(SHAPE)
    err = np.abs(got - expected).max()
    scale = np.abs(expected).max()
    ok = err <= RTOL * scale
    print(f"  {name:30s} max|veros| = {scale:.3e}   max|M T - veros| = {err:.3e}   rel = {err / scale:.1e}   {'PASS' if ok else 'FAIL'}")
    return ok


def check(label, ok):
    print(f"  {label:30s} {'PASS' if ok else 'FAIL'}")
    return ok


def test_restart(restart_file):
    print(f"\n=== {os.path.basename(restart_file)} ===")
    state, workdir = veros_state(restart_file)
    try:
        vs = state.variables
        tau = int(vs.tau)
        ok = True

        # grid, masks and velocities must match those Veros uses
        ok &= check("maskT matches Veros", np.array_equal(np.asarray(vs.maskT)[2:-2, 2:-2, :], get_mask_t()))
        ok &= check("zt matches Veros", np.allclose(np.asarray(vs.zt), make_grid_z()[0]))
        uvw = uvw_from_restart(restart_file)
        uvw_veros = np.concatenate([np.asarray(getattr(vs, n)[2:-2, 2:-2, :, tau]).ravel() for n in ("u", "v", "w")])
        ok &= check("uvw matches Veros state", np.array_equal(uvw, uvw_veros))

        M = advection_matrix(uvw)
        print(f"  matrix: {M.shape[0]} x {M.shape[1]}, {M.nnz} non-zeros")

        ok &= compare("model temperature", M, np.asarray(vs.temp[..., tau]), state)

        rng = np.random.default_rng(0)
        for n in range(3):
            tr = rng.standard_normal(np.asarray(vs.temp[..., tau]).shape)
            tr[-2:] = tr[2:4]  # periodic halos in x, as Veros maintains them
            tr[:2] = tr[-4:-2]
            ok &= compare(f"random field {n + 1}", M, tr, state)

        land = get_mask_t().ravel() == 0
        ok &= check("land rows and columns zero", abs(M[:, land]).sum() == 0 and abs(M[land, :]).sum() == 0)
        return ok, M
    finally:
        os.chdir(HERE)
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    restarts = [os.path.abspath(f) for f in sys.argv[1:]] or DEFAULT_RESTARTS
    all_ok = True
    matrices = {}
    for restart_file in restarts:
        ok, M = test_restart(restart_file)
        all_ok &= ok
        matrices[os.path.basename(restart_file)] = M

    stored_path = os.path.join(INTERMEDIATE_DIR, "advection_matrix.npz")
    if os.path.exists(stored_path):
        source = str(np.load(os.path.join(INTERMEDIATE_DIR, "advection_matrix_info.npz"))["restart_file"])
        if source in matrices:
            stored = sp.load_npz(stored_path)
            diff = abs(stored - matrices[source]).max()
            print(f"\nstored advection_matrix.npz vs advection_matrix({source}): max|diff| = {diff:.1e}")
            all_ok &= diff == 0

    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
