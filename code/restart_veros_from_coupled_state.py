#!/usr/bin/env python3
"""
Restart Veros from a temperature field of the coupled temperature-velocity solve
(solve_coupled_steady_state.py) and measure the real, competition-style score

    score_X = norm( phi(X*) - X* ) / ( norm(X*) + norm(X_0) )       (one Veros day)

day by day, together with the RMS distance of the temperature from the 200-year state.

The model is set up from the 4-year restart (the state the coupled solve started from)
and patched in memory before the first step:

    mode "full"
        temp            the coupled solution (all three time levels)
        u, v, w         the steady circulation the momentum operator gives for it
        psi             the streamfunction of that circulation; dpsi, dpsin = 0
        rho, Nsqr, Hd   recomputed from the new temperature by Veros's own routine
        du, dv, dtemp   the Adams-Bashforth "previous" tendencies are set equal to the
                        tendencies of the patched state itself, so that the first step
                        is not driven by tendencies left over from the old state
    mode "temp_only"
        only temp (and rho, Nsqr, Hd, dtemp as above); the 4-year velocities are kept
    mode "none"
        nothing is patched (the baseline continuing from the 4-year state)

tke and eke (and so the mixing coefficients) are left as in the 4-year restart: the
coupled solve does not provide them.

Run with the Python environment in which Veros is installed:
    python restart_veros_from_coupled_state.py <label> [step [n_days [mode]]]

step   index into T_steps of coupled_steady_state.npz (default -1, the last one)
Writes veros_from_coupled_<label>.csv and .log (and .npz with the final temperature).
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
START_RESTART = os.path.join(RESTART_DIR, "warmup_0004yr_fresh.restart.h5")
COUPLED_FILE = os.path.join(INTERMEDIATE_DIR, "coupled_steady_state.npz")
WARMUP_DAYS = 4 * 360
THRESHOLD = 1e-5
SCORED = ["temp", "u", "v", "tke", "eke"]

sys.path.insert(0, HERE)

import build_momentum_matrix as bm  # noqa: E402
import coupled_temperature_velocity as ctv  # noqa: E402
from build_advection_matrix import COST, SHAPE, get_mask_t  # noqa: E402
from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402
from equilibration_experiment_fixed import interior, l2_norm, read_interior_tau, setup_model  # noqa: E402


from patch_state import patch_state, with_halo  # noqa: E402, F401  (the routine that patches the model state)


def main():
    label = sys.argv[1]
    step = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 7200
    mode = sys.argv[4] if len(sys.argv) > 4 else "full"

    log = open(os.path.join(INTERMEDIATE_DIR, f"veros_from_coupled_{label}.log"), "w", buffering=1)

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    mask = get_mask_t() > 0
    T_eq = temperature_from_restart(EQUILIBRIUM_FILE).reshape(SHAPE)
    weight = np.broadcast_to(COST[np.newaxis, :, np.newaxis], SHAPE)

    def distance(T):
        d = np.where(mask, T - T_eq, 0.0)
        by_level = np.sqrt((d**2).sum(axis=(0, 1)) / np.maximum(mask.sum(axis=(0, 1)), 1))
        return np.sqrt((d**2).sum() / mask.sum()), by_level[-1], by_level[0]

    t0 = time.time()
    identifier = f"veros_from_coupled_{label}"
    model = setup_model(identifier, START_RESTART, n_days, snap_out_freq=86400.0 * n_days * 10)
    state = model.state

    if mode != "none":
        coupled = np.load(COUPLED_FILE)
        T = coupled["T_steps"][step]
        elapsed = coupled["history"][step][0]
        say(f"[{label}] mode {mode}: coupled solve step {step} of {len(coupled['T_steps']) - 1}, {elapsed:.1f} years of pseudo-time")
        uvw = None
        if mode == "full":
            uvw = ctv.CoupledProblem.from_restart(START_RESTART).velocity(T)
        info = {}
        patch_state(state, model, T, uvw, info)
        if info:
            say(f"[{label}] streamfunction: {info}")
    else:
        say(f"[{label}] mode none: the 4-year state, unpatched")

    def current(name):
        return interior(np.array(getattr(state.variables, name)[..., int(state.variables.tau)])).copy()

    norm_0 = {name: l2_norm(read_interior_tau(INIT_FILE, name)) for name in SCORED}
    norm_T0_w = l2_norm(np.sqrt(weight) * read_interior_tau(INIT_FILE, "temp"))
    prev = {name: current(name) for name in SCORED}
    say(f"[{label}] start: rms(T - T_200yr) total {distance(prev['temp'])[0]:.3f}, surface {distance(prev['temp'])[1]:.3f}, bottom {distance(prev['temp'])[2]:.3f} degC")

    csv = open(os.path.join(INTERMEDIATE_DIR, f"veros_from_coupled_{label}.csv"), "w", buffering=1)
    csv.write("day," + ",".join(f"score_{n}" for n in SCORED) + ",score_temp_weighted,rms_total,rms_surface,rms_bottom\n")

    first_below, last_above = None, 0
    for day in range(1, n_days + 1):
        try:
            model.step(state)
        except RuntimeError as exc:
            say(f"[{label}] !!! stopped at day {day}: {exc}")
            break
        now = {name: current(name) for name in SCORED}
        score = {n: l2_norm(now[n] - prev[n]) / (l2_norm(prev[n]) + norm_0[n]) for n in SCORED}
        score_w = l2_norm(np.sqrt(weight) * (now["temp"] - prev["temp"])) / (l2_norm(np.sqrt(weight) * prev["temp"]) + norm_T0_w)
        dist = distance(now["temp"])
        csv.write(f"{day}," + ",".join(f"{score[n]:.8e}" for n in SCORED) + f",{score_w:.8e},{dist[0]:.5f},{dist[1]:.5f},{dist[2]:.5f}\n")
        if score["temp"] < THRESHOLD:
            if first_below is None:
                first_below = day
                say(f"[{label}] *** temperature score first below {THRESHOLD:g} on day {day} after restart (total day {WARMUP_DAYS + day})")
        else:
            last_above = day
        if day <= 3 or day % 360 == 0 or day == n_days:
            say(
                f"[{label}] day {day:5d}  " + "  ".join(f"{n}={score[n]:.3e}" for n in SCORED)
                + f"  | rms(T - T_200yr) total {dist[0]:.3f}, surface {dist[1]:.3f}, bottom {dist[2]:.3f}  wall={time.time() - t0:.0f}s"
            )
        prev = now

    csv.close()
    say(
        f"[{label}] temperature score: first below {THRESHOLD:g} on day {first_below} after restart"
        f" (total day {None if first_below is None else WARMUP_DAYS + first_below}); last day above: {last_above} (total day {WARMUP_DAYS + last_above})"
    )
    np.savez(os.path.join(INTERMEDIATE_DIR, f"veros_from_coupled_{label}.npz"), T=prev["temp"], u=prev["u"], v=prev["v"])
    for suffix in ("averages.nc", "snapshot.nc"):
        leftover = os.path.join(CONFIG_DIR, f"{identifier}.{suffix}")
        if os.path.exists(leftover):
            os.remove(leftover)
    say(f"[{label}] done in {time.time() - t0:.0f}s")
    log.close()


if __name__ == "__main__":
    main()
