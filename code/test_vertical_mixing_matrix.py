#!/usr/bin/env python3
"""
Test build_vertical_mixing_matrix.py against Veros.

For each restart file, the global 4-degree model is set up from that restart
exactly as Example_run_Veros.py does, and the following are compared with
Veros's own routines:
  1. the surface fields Q_net, Q_nec, T_clim;
  2. kappaH_from_restart against tke.set_tke_diffusivities;
  3. the surface flux, (R T + b) * dz_top, against forc_temp_surface from the
     setup's set_forcing, for the model temperature and for a colder surface
     (to switch on the ice mask in many cells);
  4. one implicit vertical mixing step, thermodynamics.vertmix_tempsalt.
     Veros solves (I - dt D) Y = X + dt (R T^n + b), so the test checks
         (Y - X) / dt  ==  D Y + R T^n + b
     for X = the model temperature and for random fields X (random values
     on land too, which must stay unchanged).

Run with the Python environment in which Veros is installed:

    python test_vertical_mixing_matrix.py [restart_file ...]
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
RTOL = 1e-10  # max |error| allowed, relative to max |expected|

sys.path.insert(0, HERE)
from build_advection_matrix import SHAPE, get_mask_t  # noqa: E402
from build_vertical_mixing_matrix import (  # noqa: E402
    DZT,
    ice_mask,
    kappaH_from_restart,
    surface_fields,
    surface_forcing_vector,
    surface_relaxation_matrix,
    temperature_from_restart,
    vertical_diffusion_matrix,
    vertical_mixing_matrix,
)


def veros_model(restart_file):
    """Set up the model from restart_file as Example_run_Veros.py does; return (model, workdir)."""
    from global_4deg_no_cycle import GlobalFourDegreeSetup  # sets VEROS_BACKEND=jax

    workdir = tempfile.mkdtemp(dir=INTERMEDIATE_DIR, prefix="veros_vmix_test_")
    for name in ("forcing_4deg_global_annual_mean.nc", "ecmwf_4deg_monthly_nc4.nc"):
        os.symlink(os.path.join(INPUT_DIR, name), os.path.join(workdir, name))
    os.symlink(restart_file, os.path.join(workdir, "restart.h5"))
    os.chdir(workdir)

    model = GlobalFourDegreeSetup("vmix_matrix_test", 86400.0, "./restart.h5", restart_mode=True)
    model.setup()
    return model, workdir


def interior(a):
    return np.asarray(a)[2:-2, 2:-2]


def report(label, got, expected, rtol=RTOL):
    err = np.abs(np.asarray(got) - np.asarray(expected)).max()
    scale = max(np.abs(np.asarray(expected)).max(), 1e-300)
    ok = err <= rtol * scale
    print(f"  {label:42s} max|expected| = {scale:.3e}   max|err| = {err:.3e}   rel = {err / scale:.1e}   {'PASS' if ok else 'FAIL'}")
    return ok


def set_temp_level(state, level, field):
    """Set temp[2:-2, 2:-2, :, level] = field (90, 40, 15), keeping the x-halos periodic."""
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


def veros_kappaH(state):
    from veros import veros_routine
    from veros.core import tke

    result = {}

    @veros_routine
    def run(state):
        tke.set_tke_diffusivities(state)
        result["kappaH"] = np.asarray(state.variables.kappaH)

    run(state)
    return interior(result["kappaH"])


def veros_vertmix(state, X):
    """Veros's implicit vertical mixing of temp[taup1] = X; returns the new temp[taup1] (90, 40, 15)."""
    from veros import veros_routine
    from veros.core import thermodynamics

    taup1 = int(state.variables.taup1)
    set_temp_level(state, taup1, X)
    result = {}

    @veros_routine
    def run(state):
        vs = state.variables
        vs.update(thermodynamics.vertmix_tempsalt(state))
        result["temp"] = np.asarray(vs.temp[..., taup1])

    run(state)
    return interior(result["temp"])


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

        # 1. surface fields
        q_net, q_nec, t_clim = surface_fields()
        ok &= report("Q_net", q_net, interior(vs.qnet)[:, :, 0])
        ok &= report("Q_nec", q_nec, interior(vs.qnec)[:, :, 0])
        ok &= report("T_clim", t_clim, interior(vs.sst_clim)[:, :, 0])

        # 2. kappaH (compare where it is used: faces between two ocean cells)
        kappa = kappaH_from_restart(restart_file)
        kappa_veros = veros_kappaH(state)
        used = np.zeros(SHAPE, bool)
        used[:, :, :-1] = (m[:, :, :-1] * m[:, :, 1:]) > 0
        ok &= report("kappaH (ocean faces)", kappa.reshape(SHAPE)[used], kappa_veros[used])
        ok &= report("kappaH (whole interior)", kappa.reshape(SHAPE), kappa_veros)
        kappa_veros = kappa_veros.ravel()

        T_model = temperature_from_restart(restart_file)
        ok &= report("temperature matches Veros", T_model, interior(vs.temp[..., tau]).ravel(), rtol=0)
        D = vertical_diffusion_matrix(kappa_veros)
        print(f"  D: {D.nnz} non-zeros")

        rng = np.random.default_rng(0)
        T_cold = T_model.reshape(SHAPE).copy()
        T_cold[:, :, -1] -= 3.0  # colder surface, to switch on the ice mask in many cells

        for name, T_n in (("model T", T_model), ("surface 3 K colder", T_cold.ravel())):
            print(f"  -- T at time level tau: {name}; ice-masked surface cells: {int(ice_mask(T_n).sum())}")
            set_temp_level(state, tau, T_n.reshape(SHAPE))
            model.set_forcing(state)
            R = surface_relaxation_matrix(T_n)
            b = surface_forcing_vector(T_n)
            flux = ((R @ T_n + b).reshape(SHAPE)[:, :, -1]) * DZT[-1]

            # 3. surface flux
            ok &= report("surface flux (R T + b) dz_top", flux, interior(vs.forc_temp_surface))
            ok &= check_zero_below_surface(R, b)

            # 4. one implicit vertical mixing step
            Xs = [("X = model T", T_model.reshape(SHAPE))]
            for n in range(2):
                Xs.append((f"X = random field {n + 1}", rng.standard_normal(SHAPE)))
            for xname, X in Xs:
                Y = veros_vertmix(state, X)
                expected = (Y - X).ravel() / dt
                got = D @ Y.ravel() + R @ T_n + b
                ok &= report(f"vertmix step, {xname}", got, expected)

        # V = D + R
        V = vertical_mixing_matrix(kappa_veros, T_model)
        ok &= check("V == D + R", abs(V - D - surface_relaxation_matrix(T_model)).max() <= 1e-15 * abs(V).max())
        land = m.ravel() == 0
        ok &= check("land rows and columns of V zero", abs(V[:, land]).sum() == 0 and abs(V[land, :]).sum() == 0)
        ok &= check("D conserves heat (column sums of dz D = 0)", conserves(D))
        return ok
    finally:
        os.chdir(HERE)
        shutil.rmtree(workdir, ignore_errors=True)


def check(label, ok):
    print(f"  {label:42s} {'PASS' if ok else 'FAIL'}")
    return ok


def check_zero_below_surface(R, b):
    below = np.ones(SHAPE, bool)
    below[:, :, -1] = False
    below = below.ravel()
    return check("R, b zero below the surface", abs(R[below, :]).sum() == 0 and np.all(b[below] == 0))


def conserves(D):
    vol = np.broadcast_to(DZT, SHAPE).ravel()
    col = np.asarray(D.multiply(vol[:, None]).sum(axis=0)).ravel()
    return np.abs(col).max() <= 1e-12 * np.abs(D.multiply(vol[:, None])).max()


def main():
    restarts = [os.path.abspath(f) for f in sys.argv[1:]] or DEFAULT_RESTARTS
    all_ok = True
    for restart_file in restarts:
        all_ok &= test_restart(restart_file)
    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
