#!/usr/bin/env python3
"""
Isolate the shock mechanism from jumpstart_variable_scores_dtemp.py: rather than letting
temperature evolve (and feed back on the circulation every step), clamp it permanently
to the linear-solve estimate (warmup_0004yr_linsolve.restart.h5's temperature) -- so it
never changes again -- and measure how long u, v, tke and eke take to equilibrate to
*that* (wrong) state, starting from the real 4-year warm-up's own values
(warmup_0004yr_fresh.restart.h5).

This is the mirror of equilibration_experiment.py's Experiment 1 (temperature fixed,
dynamics free), but using the linear-solve temperature and the 4-year warm-up's own
circulation as the starting point, rather than spinup_200_72000's equilibrium and
Init_state.restart.h5.

Clamping (not just substituting once) sidesteps the Adams-Bashforth / core/dtemp
subtlety entirely: since temperature is overwritten back to the same fixed field after
every step, regardless of what tendency Veros computes for it internally, there is
nothing to go stale.

Score (unweighted, X_0 fixed as each variable's own value in Init_state.restart.h5, as
throughout this document): same construction as equilibration_experiment.py.

Truncated to the same total model day at which the direct-integration baseline
(score_timeseries.csv) stopped.

Run with the Python environment in which Veros is installed:
    python jumpstart_clamped_wrong_temp_fixed.py
"""

import csv
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

BASELINE_CSV = os.path.join(INTERMEDIATE_DIR, "score_timeseries.csv")
OUT_CSV = os.path.join(INTERMEDIATE_DIR, "jumpstart_clamped_wrong_temp_fixed.csv")
OUT_LOG = os.path.join(INTERMEDIATE_DIR, "jumpstart_clamped_wrong_temp_fixed.log")

FREE_NAMES = ["u", "v", "tke", "eke"]

sys.path.insert(0, HERE)


def read_full(restart_file, name):
    with h5py.File(restart_file, "r") as f:
        return np.array(f[f"core/{name}"])


def read_full_broadcast_tau(restart_file, name):
    """Read only the `tau` slot (the one actually holding T_solved) and broadcast it
    into all 3 time levels, so the field is genuinely constant regardless of which raw
    slot Veros's rotating `tau` index happens to land on each day -- unlike reading the
    full stored array directly, whose taum1/taup1 slots still hold the original,
    un-solved 4-year temperature (see decompose_u_tendency.py / check_temp_tendency_zero
    investigation for how that mismatch was found)."""
    with h5py.File(restart_file, "r") as f:
        tau = int(f["core/tau"][()])
        one_slot = np.array(f["core/temp"][..., tau])
    return np.repeat(one_slot[..., np.newaxis], 3, axis=-1)


def read_interior_tau(restart_file, name):
    with h5py.File(restart_file, "r") as f:
        tau = int(f["core/tau"][()])
        arr = f[f"core/{name}"]
        if arr.ndim == 4:
            return np.array(arr[2:-2, 2:-2, :, tau])
        return np.array(arr[2:-2, 2:-2, tau])


def interior(a):
    a = np.asarray(a)
    if a.ndim == 3:
        return a[2:-2, 2:-2, :]
    return a[2:-2, 2:-2]


def l2_norm(x):
    return float(np.sqrt(np.sum(np.asarray(x) ** 2)))


def baseline_stop_day():
    with open(BASELINE_CSV) as f:
        rows = list(csv.DictReader(f))
    return int(rows[-1]["day"])


def make_clamper(frozen_values):
    from veros import veros_routine
    from veros.core.operators import at, update
    from veros.core.operators import numpy as npx

    names = list(frozen_values.keys())

    @veros_routine
    def _clamp(state):
        vs = state.variables
        for name in names:
            current = getattr(vs, name)
            setattr(vs, name, update(current, at[...], npx.asarray(frozen_values[name])))

    def clamp(state):
        _clamp(state)

    return clamp


def setup_model(identifier, restart_file, n_days):
    from global_4deg_no_cycle import GlobalFourDegreeSetup

    os.chdir(CONFIG_DIR)
    for suffix in ("averages.nc", "snapshot.nc"):
        leftover = os.path.join(CONFIG_DIR, f"{identifier}.{suffix}")
        if os.path.exists(leftover):
            os.remove(leftover)
    model = GlobalFourDegreeSetup(
        identifier,
        n_days * 86400.0,
        os.path.relpath(restart_file, CONFIG_DIR),
        snap_out_freq=86400 * 30,
        restart_mode=True,
    )
    model.setup()
    os.chdir(HERE)
    return model


def main():
    log = open(OUT_LOG, "w", buffering=1)

    def say(msg):
        print(msg)
        log.write(msg + "\n")

    total_stop_day = baseline_stop_day()
    n_days = total_stop_day - WARMUP_DAYS
    say(f"Baseline (direct integration) stopped at total model day {total_stop_day}.")
    say(f"Running {n_days} days from the 4-year warm-up state, temperature clamped to the "
        f"linear-solve estimate throughout (total day {WARMUP_DAYS}+{n_days}={total_stop_day}).")

    t0 = time.time()
    model = setup_model("jumpstart_clamped_wrong_temp", FRESH_RESTART_FILE, n_days)
    state = model.state

    frozen_values = {"temp": read_full_broadcast_tau(LINSOLVE_RESTART_FILE, "temp")}
    clamp = make_clamper(frozen_values)
    clamp(state)  # temperature starts at the linear-solve estimate, not the real 4yr value

    X0 = {name: read_interior_tau(INIT_STATE_FILE, name) for name in FREE_NAMES}
    norm_X0 = {name: l2_norm(X0[name]) for name in FREE_NAMES}

    csv_out = open(OUT_CSV, "w", buffering=1)
    csv_out.write("day_after_restart," + ",".join(f"score_{n}" for n in FREE_NAMES) + ",wall_time_s\n")

    X_star = {}
    for name in FREE_NAMES:
        tau = int(state.variables.tau)
        X_star[name] = interior(getattr(state.variables, name)[..., tau]).copy()

    for day in range(1, n_days + 1):
        model.step(state)
        clamp(state)

        tau_after = int(state.variables.tau)
        scores = {}
        for name in FREE_NAMES:
            X_phi = interior(getattr(state.variables, name)[..., tau_after]).copy()
            diff = X_phi - X_star[name]
            scores[name] = l2_norm(diff) / (l2_norm(X_star[name]) + norm_X0[name])
            X_star[name] = X_phi

        wall = time.time() - t0
        csv_out.write(f"{day}," + ",".join(f"{scores[n]:.8e}" for n in FREE_NAMES) + f",{wall:.2f}\n")

        if day % 200 == 0:
            score_str = "  ".join(f"{n}={scores[n]:.3e}" for n in FREE_NAMES)
            say(f"day_after_restart {day:6d}  {score_str}  wall = {wall:.1f}s")

    csv_out.close()
    wall_total = time.time() - t0
    say(f"Done: wall time = {wall_total:.1f}s = {wall_total/60:.2f} min")
    log.close()


if __name__ == "__main__":
    main()
