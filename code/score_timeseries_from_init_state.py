#!/usr/bin/env python3
"""
Run Veros from Init_state.restart.h5 (exactly as Example_run_Veros.py does: same setup,
same restart, same fixed time step dt_tracer = 86400 s = 1 day), and at every model day
compute the normalised score

    score = norm( phi(T*) - T* ) / ( norm(T*) + norm(T_0) )

(see compute_normalized_score.py), both unweighted and latitude-weighted, where:
  T_0   is fixed: the temperature field of Init_state.restart.h5 itself.
  T*    is the temperature at the start of that day's step (so this changes every day,
        unlike T_0).
  phi(T*) is the temperature Veros computes one day later.

Stops (and reports, to the nearest day) the first model day on which each of the two
scores drops below 1e-5, then exits -- it does not keep running once both are found,
up to a safety cap of 72000 days (200 x 360-day years, the length Example_run_Veros.py
itself was run for to produce spinup_200_72000.restart.h5).

Progress (score every 100 days, and immediately on each threshold crossing) is appended
to score_timeseries.log; the full day-by-day series is written to score_timeseries.csv;
a one-line summary is written to score_crossing_summary.txt once both thresholds are
found (or the safety cap is reached).

Run with the Python environment in which Veros is installed, in the background (this can take tens of minutes):

    python score_timeseries_from_init_state.py
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401

CONFIG_DIR = VEROS_RUN_DIR  # Veros runs here and writes its (empty) NetCDF output here
INIT_STATE_FILE = INIT_FILE

THRESHOLD = 1e-5
MAX_DAYS = 72000
SNAP_OUT_FREQ = 86400 * 30  # same as Example_run_Veros.py

LOG_FILE = os.path.join(INTERMEDIATE_DIR, "score_timeseries.log")
CSV_FILE = os.path.join(INTERMEDIATE_DIR, "score_timeseries.csv")
SUMMARY_FILE = os.path.join(INTERMEDIATE_DIR, "score_crossing_summary.txt")

sys.path.insert(0, HERE)
from build_advection_matrix import COST, SHAPE, get_mask_t  # noqa: E402
from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402


def interior(a):
    return np.asarray(a)[2:-2, 2:-2, ...]


def l2_norm(x, weight=None):
    if weight is None:
        return float(np.sqrt(np.sum(x**2)))
    return float(np.sqrt(np.sum(weight * x**2)))


def main():
    t_script_start = time.time()
    log = open(LOG_FILE, "w", buffering=1)
    csv = open(CSV_FILE, "w", buffering=1)
    csv.write("day,score_unweighted,score_weighted,wall_time_s\n")

    def say(msg):
        print(msg)
        log.write(msg + "\n")

    say(f"Setting up Veros from {INIT_STATE_FILE} (identical to Example_run_Veros.py)")

    # run from this directory so relative restart/forcing paths resolve, as in the other
    # test scripts in this directory (no symlinking needed: we run in place, read-only,
    # and restart_frequency defaults to 0 so Veros will not overwrite Init_state.restart.h5)
    os.chdir(CONFIG_DIR)
    from global_4deg_no_cycle import GlobalFourDegreeSetup

    model = GlobalFourDegreeSetup(
        "score_timeseries",
        MAX_DAYS * 86400.0,
        INIT_FILE,
        snap_out_freq=SNAP_OUT_FREQ,
        restart_mode=True,
    )
    model.setup()
    state = model.state
    os.chdir(HERE)

    mask_t = get_mask_t()
    weight3 = np.broadcast_to(COST[np.newaxis, :, np.newaxis], SHAPE)

    T_0 = temperature_from_restart(INIT_STATE_FILE).reshape(SHAPE)
    norm_T0_unw = l2_norm(T_0)
    norm_T0_w = l2_norm(T_0, weight3)
    say(f"norm(T_0): unweighted = {norm_T0_unw:.6e}, latitude-weighted = {norm_T0_w:.6e}")

    found_unw = None
    found_w = None

    t_loop_start = time.time()
    tau = int(state.variables.tau)
    T_star = interior(state.variables.temp[..., tau]).copy()

    for day in range(1, MAX_DAYS + 1):
        model.step(state)
        tau_after = int(state.variables.tau)
        T_phi = interior(state.variables.temp[..., tau_after]).copy()

        diff = T_phi - T_star
        score_unw = l2_norm(diff) / (l2_norm(T_star) + norm_T0_unw)
        score_w = l2_norm(diff, weight3) / (l2_norm(T_star, weight3) + norm_T0_w)

        wall = time.time() - t_script_start
        csv.write(f"{day},{score_unw:.8e},{score_w:.8e},{wall:.2f}\n")

        if day % 100 == 0:
            say(f"day {day:6d}  score_unweighted = {score_unw:.6e}  score_weighted = {score_w:.6e}  wall = {wall:.1f}s")

        if found_unw is None and score_unw < THRESHOLD:
            found_unw = (day, wall)
            say(f"*** unweighted score first below {THRESHOLD:g} at day {day} (wall time {wall:.1f}s = {wall/60:.2f} min)")

        if found_w is None and score_w < THRESHOLD:
            found_w = (day, wall)
            say(f"*** latitude-weighted score first below {THRESHOLD:g} at day {day} (wall time {wall:.1f}s = {wall/60:.2f} min)")

        T_star = T_phi

        if found_unw is not None and found_w is not None:
            break

    total_wall = time.time() - t_script_start

    with open(SUMMARY_FILE, "w") as f:
        for label, found in (("unweighted", found_unw), ("latitude-weighted", found_w)):
            if found is None:
                line = f"{label}: threshold {THRESHOLD:g} not reached within {MAX_DAYS} days"
            else:
                day, wall = found
                line = (
                    f"{label}: score < {THRESHOLD:g} at model day {day} "
                    f"({day/360:.3f} x 360-day years), wall time {wall:.1f}s = {wall/60:.2f} min "
                    f"(script total, including setup)"
                )
            say(line)
            f.write(line + "\n")
        f.write(f"total wall time for this run: {total_wall:.1f}s = {total_wall/60:.2f} min\n")

    say(f"\nTotal wall time: {total_wall:.1f}s = {total_wall/60:.2f} min")
    log.close()
    csv.close()


if __name__ == "__main__":
    main()
