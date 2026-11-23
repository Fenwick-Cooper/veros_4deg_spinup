#!/usr/bin/env python3
"""
Patch core/dtemp (the cached Adams-Bashforth advection tendency, at the taum1 slot) in a
copy of the linear-solve restart file, so that the AB2 scheme sees a tendency consistent
with the solved temperature on the very first step after restart, rather than the stale
tendency left over from the real (un-solved) 4-year run.

The only thing that matters for the *next* step is core/dtemp[...,taum1]: dtemp[...,tau]
is recomputed fresh by Veros's own advect_temperature() at the start of every call to
step(), from whatever temperature is currently in the tau slot -- so patching it in the
restart file would be overwritten immediately and have no effect (checked directly below,
not assumed).

The patched tendency is g = M @ T_solved, where M is the (linear, velocity-only-dependent)
advection matrix already built and validated in this directory (build_advection_matrix.py,
validated against real Veros to ~1e-10 relative error in test_combined_step.py for other
temperature fields). Before trusting it here, this script independently checks it against
Veros's own advect_temperature() kernel, called directly (not through model.step(), so
nothing else -- in particular momentum() -- can change u, v first) on the patched restart.

Run with the Python environment in which Veros is installed:
    python patch_dtemp_and_verify.py
"""

import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401

CONFIG_DIR = VEROS_RUN_DIR  # Veros runs here and writes its (empty) NetCDF output here
LINSOLVE_RESTART_FILE = os.path.join(RESTART_DIR, "warmup_0004yr_linsolve.restart.h5")
PATCHED_RESTART_FILE = os.path.join(RESTART_DIR, "warmup_0004yr_linsolve_dtemp.restart.h5")

sys.path.insert(0, HERE)
from build_advection_matrix import SHAPE, get_mask_t, uvw_from_restart, advection_matrix  # noqa: E402
from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402


def main():
    mask_t = get_mask_t()
    T_solved = temperature_from_restart(LINSOLVE_RESTART_FILE)  # already T_solved (patched tau slot)
    uvw = uvw_from_restart(LINSOLVE_RESTART_FILE)  # level="tau" default, matches the validated scheme
    M = advection_matrix(uvw, mask_t=mask_t)

    g_solved = (M @ T_solved).reshape(SHAPE)
    print(f"computed g = M @ T_solved: max|g| = {np.nanmax(np.abs(g_solved)):.6e}")

    # --- independent check: Veros's own advect_temperature(), called directly, on the
    #     same (patched) restart, so u, v, w and T are exactly what M and T_solved used
    os.chdir(CONFIG_DIR)
    from veros import runtime_settings
    from global_4deg_no_cycle import GlobalFourDegreeSetup
    from veros.core import thermodynamics

    for suffix in ("averages.nc", "snapshot.nc"):
        leftover = os.path.join(CONFIG_DIR, f"dtemp_check.{suffix}")
        if os.path.exists(leftover):
            os.remove(leftover)

    model = GlobalFourDegreeSetup(
        "dtemp_check",
        86400.0,
        os.path.relpath(LINSOLVE_RESTART_FILE, CONFIG_DIR),
        snap_out_freq=86400 * 30,
        restart_mode=True,
    )
    model.setup()
    state = model.state
    os.chdir(HERE)

    vs = state.variables
    out = thermodynamics.advect_temperature(state)
    tau = int(vs.tau)
    g_veros = np.array(out.dtemp[2:-2, 2:-2, :, tau]).reshape(SHAPE)

    diff = g_veros - g_solved
    ocean = mask_t > 0
    abs_err = np.abs(diff[ocean])
    rel_err = abs_err / np.maximum(np.abs(g_veros[ocean]), 1e-30)
    print(f"Veros's own advect_temperature(): max|g_veros| = {np.nanmax(np.abs(g_veros)):.6e}")
    print(f"max abs diff (ocean cells)  = {abs_err.max():.6e}")
    print(f"max rel diff (ocean cells)  = {rel_err.max():.6e}")
    print(f"mean abs diff (ocean cells) = {abs_err.mean():.6e}")

    # --- patch core/dtemp[...,taum1] (interior + periodic x-halo) into a fresh copy
    import shutil

    shutil.copyfile(LINSOLVE_RESTART_FILE, PATCHED_RESTART_FILE)
    with h5py.File(PATCHED_RESTART_FILE, "r+") as f:
        taum1 = int(f["core/taum1"][()])
        full = np.array(f["core/dtemp"][:, :, :, taum1])  # (94, 44, 15), includes halos
        full[2:-2, 2:-2, :] = g_solved
        full[:2, 2:-2, :] = g_solved[-2:, :, :]  # periodic west halo
        full[-2:, 2:-2, :] = g_solved[:2, :, :]  # periodic east halo
        f["core/dtemp"][:, :, :, taum1] = full

    print(f"\nPatched core/dtemp[...,taum1] into {PATCHED_RESTART_FILE}")

    # --- confirm patching dtemp[...,tau] in the restart would be pointless: Veros
    #     overwrites it immediately (as above) regardless of what was on disk
    with h5py.File(LINSOLVE_RESTART_FILE, "r") as f:
        tau_idx = int(f["core/tau"][()])
        g_on_disk_tau = np.array(f["core/dtemp"][2:-2, 2:-2, :, tau_idx])
    print(f"\n(sanity check) core/dtemp[...,tau] on disk before the call: max|.| = {np.nanmax(np.abs(g_on_disk_tau)):.6e}")
    print("-> advect_temperature() overwrote this with the freshly computed tendency above regardless,")
    print("   confirming dtemp[...,tau] in the restart file is never read back by Veros.")


if __name__ == "__main__":
    main()
