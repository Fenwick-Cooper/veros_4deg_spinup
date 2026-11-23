#!/usr/bin/env python3
"""
Ablation study for the steady-state temperature equation A_T T = -b (see
solve_steady_state_temperature.py), A_T = M + E + D33 + F + D + R.

Built from the 4-year warm-up restart (warmup_0004yr_fresh.restart.h5), solve this
equation six times, each time leaving one of the six matrices out of the sum, plus once
with the full sum and once with none (Veros's own 4-year state, no solve at all). Compare
the top (surface) layer of each result to the true climatological equilibrium -- the
surface temperature in spinup_200_72000.restart.h5.

Run with the Python environment in which Veros is installed:
    python ablation_A_T_matrices.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401

CONFIG_DIR = VEROS_RUN_DIR  # Veros runs here and writes its (empty) NetCDF output here
FRESH_RESTART_FILE = os.path.join(RESTART_DIR, "warmup_0004yr_fresh.restart.h5")
CLIMATOLOGY_FILE = T200_FILE
OUT_NPZ = os.path.join(INTERMEDIATE_DIR, "ablation_A_T_matrices.npz")

sys.path.insert(0, HERE)

from build_advection_matrix import SHAPE, XT, YT, get_mask_t, uvw_from_restart  # noqa: E402
from build_combined_step import combined_matrices  # noqa: E402
from build_gm_matrix import eke_from_restart, nsqr_from_restart  # noqa: E402
from build_isoneutral_matrix import salinity_from_restart  # noqa: E402
from build_vertical_mixing_matrix import kappaH_from_restart, temperature_from_restart  # noqa: E402
from solve_steady_state_temperature import solve_steady_state  # noqa: E402

MATRIX_NAMES = ["M", "E", "D33", "F", "D", "R"]
MATRIX_LABELS = {
    "M": "advection",
    "E": "isoneutral (Redi), explicit",
    "D33": "isoneutral $K_{33}$, implicit",
    "F": "GM skew",
    "D": "vertical mixing",
    "R": "surface relaxation",
}


def main():
    mask_t = get_mask_t()
    top_mask = mask_t[:, :, -1] > 0

    print(f"Building all six matrices from {os.path.basename(FRESH_RESTART_FILE)}")
    T = temperature_from_restart(FRESH_RESTART_FILE)
    S = salinity_from_restart(FRESH_RESTART_FILE)
    uvw = uvw_from_restart(FRESH_RESTART_FILE)
    kappaH = kappaH_from_restart(FRESH_RESTART_FILE)
    Nsqr = nsqr_from_restart(FRESH_RESTART_FILE)
    Ekin = eke_from_restart(FRESH_RESTART_FILE)
    matrices = combined_matrices(T, S, uvw, kappaH, Nsqr, Ekin, mask_t=mask_t)
    b = matrices["b"]

    T_clim_top = np.where(top_mask, temperature_from_restart(CLIMATOLOGY_FILE).reshape(SHAPE)[:, :, -1], np.nan)
    T_4yr_top = np.where(top_mask, T.reshape(SHAPE)[:, :, -1], np.nan)

    print("Solving the full A_T . T = -b...")
    A_full = sum(matrices[k] for k in MATRIX_NAMES).tocsr()
    T_full_top = np.where(top_mask, solve_steady_state(A_full, b, mask_t).reshape(SHAPE)[:, :, -1], np.nan)

    leaveout_tops = {}
    for name in MATRIX_NAMES:
        print(f"Solving with {name} ({MATRIX_LABELS[name]}) left out...")
        A_leaveout = sum(matrices[k] for k in MATRIX_NAMES if k != name).tocsr()
        T_leaveout = solve_steady_state(A_leaveout, b, mask_t).reshape(SHAPE)
        leaveout_tops[name] = np.where(top_mask, T_leaveout[:, :, -1], np.nan)

    print("Solving with D33 and F left out...")
    A_leaveout_D33_F = sum(matrices[k] for k in MATRIX_NAMES if k not in ("D33", "F")).tocsr()
    T_leaveout_D33_F = solve_steady_state(A_leaveout_D33_F, b, mask_t).reshape(SHAPE)
    T_leaveout_D33_F_top = np.where(top_mask, T_leaveout_D33_F[:, :, -1], np.nan)

    np.savez(
        OUT_NPZ,
        xt=XT,
        yt=YT,
        T_clim_top=T_clim_top,
        T_4yr_top=T_4yr_top,
        T_full_top=T_full_top,
        T_leaveout_D33_F_top=T_leaveout_D33_F_top,
        **{f"T_leaveout_{name}_top": leaveout_tops[name] for name in MATRIX_NAMES},
    )
    print(f"Saved {OUT_NPZ}")


if __name__ == "__main__":
    main()
