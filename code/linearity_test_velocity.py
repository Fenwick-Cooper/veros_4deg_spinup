#!/usr/bin/env python3
"""
Is the settled circulation a linear function of the temperature field?

Three runs, identical except for the temperature that is clamped (at all three time
levels, after every step) throughout:

    a    T = the 4-year warm-up temperature          (warmup_0004yr_fresh.restart.h5)
    b    T = the 200-year temperature                (spinup_200_72000.restart.h5)
    mid  T = (T_a + T_b) / 2

Every other variable starts from the 4-year warm-up state and is left free. After
n_days the settled u, v, w, psi, tke, eke and kappaH are saved to
linearity_test_velocity_<case>.npz. If the circulation is linear in T, the fields from
`mid` equal the average of those from `a` and `b` (compare_linearity_test.py).

Run with the Python environment in which Veros is installed:
    python linearity_test_velocity.py a|b|mid [n_days]
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
FRESH_RESTART_FILE = os.path.join(RESTART_DIR, "warmup_0004yr_fresh.restart.h5")
EQUILIBRIUM_FILE = T200_FILE
DEFAULT_N_DAYS = 3000
N_TAIL = 30  # also save the mean over the last N_TAIL days

sys.path.insert(0, HERE)

from equilibration_experiment_fixed import interior, l2_norm, make_clamper, setup_model  # noqa: E402

FREE_NAMES = ["u", "v", "w", "psi", "tke", "eke"]


def read_tau_slot(restart_file, name):
    with h5py.File(restart_file, "r") as f:
        tau = int(f["core/tau"][()])
        return np.array(f[f"core/{name}"][..., tau])


def main():
    case = sys.argv[1]
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_DAYS

    T_a = read_tau_slot(FRESH_RESTART_FILE, "temp")
    T_b = read_tau_slot(EQUILIBRIUM_FILE, "temp")
    T_clamp = {"a": T_a, "b": T_b, "mid": 0.5 * (T_a + T_b)}[case]
    frozen = {"temp": np.repeat(T_clamp[..., np.newaxis], 3, axis=-1)}

    identifier = f"linearity_test_{case}"
    t0 = time.time()
    # snapshot interval longer than the run: no snapshots are wanted from these runs
    model = setup_model(identifier, FRESH_RESTART_FILE, n_days, snap_out_freq=86400.0 * n_days * 10)
    state = model.state
    clamp = make_clamper(frozen)
    clamp(state)

    def current(name):
        tau = int(state.variables.tau)
        return interior(np.array(getattr(state.variables, name)[..., tau])).copy()

    prev = {name: current(name) for name in FREE_NAMES}
    tail_sum = {name: np.zeros_like(prev[name]) for name in FREE_NAMES}
    for day in range(1, n_days + 1):
        model.step(state)
        clamp(state)
        now = {name: current(name) for name in FREE_NAMES}
        if day % 250 == 0 or day == n_days:
            # one-day relative change, as a check that the circulation has settled
            change = {n: l2_norm(now[n] - prev[n]) / max(l2_norm(now[n]), 1e-300) for n in FREE_NAMES}
            print(f"[{case}] day {day:5d}  " + "  ".join(f"{n}={change[n]:.2e}" for n in FREE_NAMES)
                  + f"  wall={time.time() - t0:.0f}s", flush=True)
        if day > n_days - N_TAIL:
            for name in FREE_NAMES:
                tail_sum[name] += now[name]
        prev = now

    out = {name: prev[name] for name in FREE_NAMES}
    out.update({f"{name}_tailmean": tail_sum[name] / N_TAIL for name in FREE_NAMES})
    out["kappaH"] = interior(np.array(state.variables.kappaH)).copy()
    out["kappaM"] = interior(np.array(state.variables.kappaM)).copy()
    out["temp"] = interior(T_clamp)
    np.savez(os.path.join(INTERMEDIATE_DIR, f"linearity_test_velocity_{case}.npz"), **out)

    for suffix in ("averages.nc", "snapshot.nc"):
        leftover = os.path.join(CONFIG_DIR, f"{identifier}.{suffix}")
        if os.path.exists(leftover):
            os.remove(leftover)
    print(f"[{case}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
