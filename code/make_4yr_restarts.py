#!/usr/bin/env python3
"""
Recreate the 4-year warm-up restart pair for plotting the temperature field before and
after the linear solve:

  warmup_0004yr_fresh.restart.h5     -- unpatched: Veros's own temperature after 4 years
  warmup_0004yr_linsolve.restart.h5  -- patched: same state, temperature (tau slot only)
                                         replaced by the A_T T = -b equilibrium estimate

(The temperature field itself is identical whether or not core/dtemp is separately
patched for a consistent restart -- that only affects how the model continues
integrating, not the field values -- so this intentionally omits the dtemp patch.)

Run with the Python environment in which Veros is installed:
    python make_4yr_restarts.py
"""

import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401

CONFIG_DIR = VEROS_RUN_DIR  # Veros runs here and writes its (empty) NetCDF output here
INIT_STATE_FILE = INIT_FILE
FRESH_RESTART_FILE = os.path.join(RESTART_DIR, "warmup_0004yr_fresh.restart.h5")
LINSOLVE_RESTART_FILE = os.path.join(RESTART_DIR, "warmup_0004yr_linsolve.restart.h5")
WARMUP_DAYS = 4 * 360

sys.path.insert(0, HERE)
from build_advection_matrix import SHAPE, get_mask_t  # noqa: E402
from solve_steady_state_temperature import build_A_T_and_b, solve_steady_state  # noqa: E402


def setup_model(identifier, restart_file, runlen_days):
    from global_4deg_no_cycle import GlobalFourDegreeSetup

    os.chdir(CONFIG_DIR)
    for suffix in ("averages.nc", "snapshot.nc"):
        leftover = os.path.join(CONFIG_DIR, f"{identifier}.{suffix}")
        if os.path.exists(leftover):
            os.remove(leftover)
    model = GlobalFourDegreeSetup(
        identifier,
        runlen_days * 86400.0,
        os.path.relpath(restart_file, CONFIG_DIR),
        snap_out_freq=86400 * 30,
        restart_mode=True,
    )
    model.setup()
    os.chdir(HERE)
    return model


def main():
    t0 = time.time()

    print(f"=== Running the {WARMUP_DAYS}-day warm-up from {os.path.basename(INIT_STATE_FILE)} ===")
    model = setup_model("warmup_4yr_remake", INIT_STATE_FILE, WARMUP_DAYS)
    state = model.state
    for _ in range(WARMUP_DAYS):
        model.step(state)

    from veros import restart

    with state.settings.unlock():
        state.settings.restart_output_filename = FRESH_RESTART_FILE
    restart.write_restart(state, force=True)
    print(f"wrote unpatched 4-year restart to {FRESH_RESTART_FILE}; wall = {time.time()-t0:.1f}s")

    print("\n=== Building A_T, b from the fresh restart and solving for the equilibrium temperature ===")
    mask_t = get_mask_t()
    A_T, b, T_restart = build_A_T_and_b(FRESH_RESTART_FILE)
    T_steady = solve_steady_state(A_T, b, mask_t)
    ocean = mask_t.ravel() > 0
    T_write = T_restart.copy()
    T_write[ocean] = T_steady[ocean]
    T_write = T_write.reshape(SHAPE)

    import shutil

    shutil.copyfile(FRESH_RESTART_FILE, LINSOLVE_RESTART_FILE)
    with h5py.File(LINSOLVE_RESTART_FILE, "r+") as f:
        tau = int(f["core/tau"][()])
        full = np.array(f["core/temp"][:, :, :, tau])  # (94, 44, 15), includes halos
        full[2:-2, 2:-2, :] = T_write
        full[:2, 2:-2, :] = T_write[-2:, :, :]  # periodic west halo
        full[-2:, 2:-2, :] = T_write[:2, :, :]  # periodic east halo
        f["core/temp"][:, :, :, tau] = full

    print(f"wrote linear-solve-patched 4-year restart to {LINSOLVE_RESTART_FILE}")
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
