#!/usr/bin/env python3
"""
Solve for the steady-state temperature with the circulation responding to temperature
(coupled_temperature_velocity.py), starting from a restart file, and compare the result
with the 200-year state, level by level, alongside the frozen-velocity linear solve
(solve_steady_state_temperature.py).

The solve marches in implicit pseudo-time steps whose length doubles after each
accepted step (pseudo_time_solve), starting from dt_first years. The closure fields
(kappaH, kappaM, K_gm, kappa_gm) are those of the restart file throughout, except that
kappaH is raised wherever the column becomes statically unstable.

Run with the Python environment in which Veros is installed:
    python solve_coupled_steady_state.py [restart_file [dt_first [max_steps]]]

defaults: warmup_0004yr_fresh.restart.h5, 1 year, 30 steps.
Saves coupled_steady_state.npz: T, u, v, w of the coupled solution, T_frozen, the history
(elapsed pseudo-time in years, dt, score) and T_steps, the temperature after every accepted
step (T_steps[0] is the start state), in the same order as history.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401

CONFIG_DIR = VEROS_RUN_DIR  # Veros runs here and writes its (empty) NetCDF output here
EQUILIBRIUM_FILE = T200_FILE
DEFAULT_RESTART = os.path.join(RESTART_DIR, "warmup_0004yr_fresh.restart.h5")

sys.path.insert(0, HERE)

import coupled_temperature_velocity as ctv  # noqa: E402
from build_advection_matrix import SHAPE, uvw_from_restart  # noqa: E402
from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402
from solve_steady_state_temperature import solve_steady_state  # noqa: E402


def rms_by_level(a, b, mask):
    d = np.where(mask, a.reshape(SHAPE) - b.reshape(SHAPE), 0.0)
    by_level = np.sqrt((d**2).sum(axis=(0, 1)) / np.maximum(mask.sum(axis=(0, 1)), 1))
    return np.sqrt((d**2).sum() / mask.sum()), by_level[::-1]


def main():
    restart_file = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESTART
    dt_first = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    problem = ctv.CoupledProblem.from_restart(restart_file)
    mask, oc = problem.mask_t > 0, problem.ocean
    T0 = temperature_from_restart(restart_file)
    T_eq = temperature_from_restart(EQUILIBRIUM_FILE)
    T_norm = np.linalg.norm(T0[oc]) + np.linalg.norm(temperature_from_restart(INIT_FILE)[oc])

    # frozen-velocity linear solve, for comparison (the restart's own velocities)
    L0, b0 = problem.mixing_matrices(T0)
    uvw_restart = tuple(uvw_from_restart(restart_file).reshape(3, *SHAPE))
    T_frozen = np.nan_to_num(solve_steady_state((problem.advection(uvw_restart) + L0).tocsr(), b0, problem.mask_t))

    T_steps = []

    def monitor(T):
        T_steps.append(T.reshape(SHAPE).copy())
        total, by_level = rms_by_level(T, T_eq, mask)
        return f"| rms(T - T_200yr) total {total:.3f}, surface {by_level[0]:.2f}, 935 m {by_level[6]:.2f}, bottom {by_level[-1]:.2f}"

    print(f"Coupled pseudo-time solve from {os.path.basename(restart_file)}")
    t0 = time.time()
    T_coupled, history = ctv.pseudo_time_solve(problem, T0, T_norm, dt_first=dt_first, max_steps=max_steps, monitor=monitor)
    print(f"   {len(history) - 1} steps in {time.time() - t0:.0f}s")

    print("\nRMS(T - T_200yr) [degC]: total, then by level from the surface to the bottom")
    for name, T in (("restart state", T0), ("frozen-velocity linear solve", T_frozen), ("coupled solve", T_coupled)):
        total, by_level = rms_by_level(T, T_eq, mask)
        print(f"   {name:30s} total={total:6.3f} | " + " ".join(f"{x:5.2f}" for x in by_level))

    u, v, w = problem.velocity(T_coupled)
    np.savez(
        os.path.join(INTERMEDIATE_DIR, "coupled_steady_state.npz"),
        T=T_coupled.reshape(SHAPE), T_frozen=T_frozen.reshape(SHAPE), u=u, v=v, w=w, history=np.array(history),
        T_steps=np.array(T_steps),
    )  # fmt: skip


if __name__ == "__main__":
    main()
