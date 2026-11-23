#!/usr/bin/env python3
"""
Re-run of jumpstart_variable_scores.py, but starting from the dtemp-patched restart
(warmup_0004yr_linsolve_dtemp.restart.h5), so that the Adams-Bashforth tendency is
consistent with the solved temperature from the very first step. All five variables
(u, v, temp, tke, eke) are tracked fresh in this run, since patching dtemp changes the
whole subsequent trajectory (unlike the single-field-reuse shortcut used previously).

Truncated to the same total model day at which the direct-integration baseline
(score_timeseries.csv) stopped.

Run with the Python environment in which Veros is installed:
    python jumpstart_variable_scores_dtemp.py
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
PATCHED_RESTART_FILE = os.path.join(RESTART_DIR, "warmup_0004yr_linsolve_dtemp.restart.h5")
WARMUP_DAYS = 4 * 360

BASELINE_CSV = os.path.join(INTERMEDIATE_DIR, "score_timeseries.csv")
OUT_CSV = os.path.join(INTERMEDIATE_DIR, "jumpstart_variable_scores_dtemp.csv")
OUT_LOG = os.path.join(INTERMEDIATE_DIR, "jumpstart_variable_scores_dtemp.log")

FREE_NAMES = ["u", "v", "temp", "tke", "eke"]

sys.path.insert(0, HERE)


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
    say(f"Running the dtemp-patched jump-started trajectory for {n_days} more days "
        f"(total day {WARMUP_DAYS}+{n_days}={total_stop_day}).")

    t0 = time.time()
    model = setup_model("jumpstart_scores_dtemp", PATCHED_RESTART_FILE, n_days)
    state = model.state

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
