#!/usr/bin/env python3
"""
Two experiments on which part of the model equilibrates faster: the thermal field or
the dynamics (momentum, mixing energetics), when the *other* part is held fixed at its
spinup_200_72000.restart.h5 equilibrium value throughout the run.

Experiment 1 ("dynamics free"): temperature is clamped to its spinup_200_72000
equilibrium value every step -- never advected, diffused, or otherwise changed -- while
every other prognostic variable (u, v, w, salt, tke, eke, psi) is initialised from
Init_state.restart.h5 and left to evolve normally. Tracks how long each of those free
variables takes to equilibrate.

Experiment 2 ("temperature free"): the mirror image. u, v, w, salt, tke, eke and psi are
all clamped to their spinup_200_72000 equilibrium values every step; only temperature is
initialised from Init_state.restart.h5 and left to evolve normally. Tracks how long
temperature alone takes to equilibrate when circulation and mixing are already ideal.

Clamping is literal and total: after every real Veros step (which internally computes
whatever tendency it likes for a clamped variable), that variable's array is overwritten
back to the spinup_200_72000 value at every one of its three stored time levels, so it is
numerically indistinguishable from a variable that was never advected at all.

Score (same construction as the temperature score used earlier, with X in place of T):

    score_X = norm( phi(X*) - X* ) / ( norm(X*) + norm(X_0) )

X_0 is fixed per variable: its own value in Init_state.restart.h5 (not spinup_200's, and
not re-read at the point a variable happens to be clamped -- a clamped variable's score is
not computed at all, since it is trivially zero by construction). X* is that variable's
value at the start of each day; phi(X*) is Veros's result one day later. norm() is the
plain (unweighted) L2 norm; "appropriate normalisation" here means using each variable's
*own* X_0 and X* (so a velocity is judged against the scale of velocities and salinity
against the scale of salinities, etc.), not reusing temperature's scale for a different
physical quantity. psi (the barotropic streamfunction, no z dimension) is scored the same
way, just without a vertical sum.

Run with the Python environment in which Veros is installed:

    python equilibration_experiment_fixed.py exp1 [n_days]
    python equilibration_experiment_fixed.py exp2 [n_days]
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
EQUILIBRIUM_FILE = T200_FILE

DEFAULT_N_DAYS = 10000
SNAP_OUT_FREQ = 86400 * 30

# the variables Veros steps forward in time via their own tendency equations (3 stored
# time levels each); every other field (rho, Nsqr, Hd, kappaH, K_gm, ...) is a pure
# diagnostic recomputed every step from these, so there is nothing to freeze for them
PROGNOSTIC_VARS = ["temp", "salt", "u", "v", "w", "tke", "eke", "psi"]

sys.path.insert(0, HERE)


def read_full(restart_file, name):
    """The full stored array for `name` (all 3 time levels, with halos), straight from
    the restart file -- the exact shape and values Veros itself holds in vs.<name>."""
    with h5py.File(restart_file, "r") as f:
        return np.array(f[f"core/{name}"])


def read_full_broadcast_tau(restart_file, name):
    """Read only the `tau` slot (the one Veros actually advances from) and broadcast
    it into all of the variable's stored time levels, so the clamped field is
    genuinely constant regardless of which raw slot Veros's rotating `tau` index
    happens to land on each day. Reading the full stored array directly (read_full)
    instead leaves tiny but non-zero differences between the taum1/tau/taup1 slots of
    a real restart file, which then show up as a small, spurious period-3 ripple in
    the clamped variable's own value -- confirmed by comparing to the late-time tail
    of equilibration_exp1.csv / equilibration_exp2.csv."""
    with h5py.File(restart_file, "r") as f:
        tau = int(f["core/tau"][()])
        one_slot = np.array(f[f"core/{name}"][..., tau])
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


def make_clamper(frozen_values):
    """frozen_values: {name: full array (matching vs.<name>'s own shape exactly)}.
    Returns a function clamp(state) that overwrites every listed variable, at every
    time level, back to its frozen value -- called once before the loop starts (so the
    frozen variables start at equilibrium, not at Init_state) and once after every step
    (so they never move)."""
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


def setup_model(identifier, restart_file, n_days, snap_out_freq=SNAP_OUT_FREQ):
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
        snap_out_freq=snap_out_freq,
        restart_mode=True,
    )
    model.setup()
    os.chdir(HERE)
    return model


def run_experiment(label, identifier, frozen_names, free_names, n_days, csv_path, log_path):
    """frozen_names are clamped to EQUILIBRIUM_FILE throughout; free_names are
    initialised from Init_state.restart.h5 (as the whole model is) and left alone."""
    log = open(log_path, "w", buffering=1)

    def say(msg):
        print(msg)
        log.write(msg + "\n")

    say(f"=== {label} ===")
    say(f"frozen (clamped to spinup_200_72000 equilibrium): {frozen_names}")
    say(f"free (initialised from Init_state, left to evolve): {free_names}")

    t0 = time.time()
    model = setup_model(identifier, INIT_STATE_FILE, n_days)
    state = model.state

    frozen_values = {name: read_full_broadcast_tau(EQUILIBRIUM_FILE, name) for name in frozen_names}
    clamp = make_clamper(frozen_values)
    clamp(state)  # frozen variables start at equilibrium, not at Init_state's own value

    X0 = {name: read_interior_tau(INIT_STATE_FILE, name) for name in free_names}
    norm_X0 = {name: l2_norm(X0[name]) for name in free_names}

    csv = open(csv_path, "w", buffering=1)
    csv.write("day," + ",".join(f"score_{n}" for n in free_names) + ",wall_time_s\n")

    X_star = {}
    for name in free_names:
        tau = int(state.variables.tau)
        X_star[name] = interior(getattr(state.variables, name)[..., tau]).copy()

    for day in range(1, n_days + 1):
        try:
            model.step(state)
        except RuntimeError as exc:
            say(f"!!! stopped early at day {day}: {exc}")
            break

        clamp(state)

        tau_after = int(state.variables.tau)
        scores = {}
        for name in free_names:
            X_phi = interior(getattr(state.variables, name)[..., tau_after]).copy()
            diff = X_phi - X_star[name]
            scores[name] = l2_norm(diff) / (l2_norm(X_star[name]) + norm_X0[name])
            X_star[name] = X_phi

        wall = time.time() - t0
        csv.write(f"{day}," + ",".join(f"{scores[n]:.8e}" for n in free_names) + f",{wall:.2f}\n")

        if day % 200 == 0:
            score_str = "  ".join(f"{n}={scores[n]:.3e}" for n in free_names)
            say(f"day {day:6d}  {score_str}  wall = {wall:.1f}s")

    csv.close()
    wall_total = time.time() - t0
    say(f"Done: wall time = {wall_total:.1f}s = {wall_total/60:.2f} min")
    log.close()
    return wall_total


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "exp1"
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_DAYS

    if mode == "exp1":
        free_names = ["u", "v", "w", "salt", "tke", "eke", "psi"]
        run_experiment(
            "Experiment 1: temperature fixed at equilibrium, dynamics free from Init_state",
            "equil_exp1_fixed",
            frozen_names=["temp"],
            free_names=free_names,
            n_days=n_days,
            csv_path=os.path.join(INTERMEDIATE_DIR, "equilibration_exp1_fixed.csv"),
            log_path=os.path.join(INTERMEDIATE_DIR, "equilibration_exp1_fixed.log"),
        )
    elif mode == "exp2":
        frozen_names = ["u", "v", "w", "salt", "tke", "eke", "psi"]
        run_experiment(
            "Experiment 2: dynamics fixed at equilibrium, temperature free from Init_state",
            "equil_exp2_fixed",
            frozen_names=frozen_names,
            free_names=["temp"],
            n_days=n_days,
            csv_path=os.path.join(INTERMEDIATE_DIR, "equilibration_exp2_fixed.csv"),
            log_path=os.path.join(INTERMEDIATE_DIR, "equilibration_exp2_fixed.log"),
        )
    else:
        raise ValueError(f"unknown mode {mode!r}, expected 'exp1' or 'exp2'")


if __name__ == "__main__":
    main()
