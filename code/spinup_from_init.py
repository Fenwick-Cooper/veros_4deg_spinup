#!/usr/bin/env python3
"""
The whole accelerated spin-up, starting from data/input/Init_state.restart.h5 (day 0):

    1. run Veros conventionally for `--warmup` days,
    2. from that state, march the coupled temperature-velocity problem
       (coupled_temperature_velocity.py) through `--steps` implicit pseudo-time steps,
    3. put the result into the model (restart_veros_from_coupled_state.patch_state:
       temperature, the circulation that belongs to it, streamfunction, density,
       consistent Adams-Bashforth tendencies) and run Veros on until the temperature score

           score = norm( phi(T*) - T* ) / ( norm(T*) + norm(T_0) )      (one Veros day)

       has stayed below 1e-5 for `--confirm` consecutive days (or `--post-days` have passed).

`--steps` may be a comma-separated list (e.g. 3,5,7): the warm-up and the solve are then
done once, and step 3 is repeated from the warm-up state for each number of steps.
`--steps 0` runs Veros on without any solve (the baseline).

`--cycles days:steps:dt_first,days:steps:dt_first,...` instead repeats steps 1-2 several
times in one model run before step 3, so that later solves use turbulence energies that
have adjusted to the earlier solution. Writing the middle field as e.g. 60y asks for that
many pseudo-years instead of a number of steps (30:30y:0.125,30:60y:2).

Days are counted from the initial state; "total day" is warm-up days + days after the
solve. Wall times are measured here (warm-up, solve, run after the solve).

With --cycles, --restart-out FILE writes the model state on the first day below 1e-5 after
the last round to FILE as a Veros restart file (as competition_spin-up.py and
equilibrium_spin-up.py do).

Run with the Python environment in which Veros is installed, e.g.
    python code/spinup_from_init.py --warmup 360 --steps 3,5,7

Writes data/intermediate/from_init/w<warmup>_k<steps>[_<tag>].csv (daily scores and RMS
distance from the 200-year state T_200, which is NaN until the 2000-year reference run has
reached year 200) and appends one summary line per run to data/intermediate/from_init/summary.txt.
Restart files written for the solves go to data/veros_restart/.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from paths import INIT_FILE, INTERMEDIATE_DIR, RESTART_DIR, T200_FILE  # noqa: E402
from veros_path import ensure_veros  # noqa: E402

ensure_veros()

OUT_DIR = os.path.join(INTERMEDIATE_DIR, "from_init")
EQUILIBRIUM_FILE = T200_FILE  # T_200, from the 2000-year reference run; distances are NaN without it
THRESHOLD = 1e-5
BASELINE_DAY = 3714  # day on which conventional Veros first passes the threshold

# The Veros setup must be imported before anything else that imports veros.core (the
# solver modules below do): it selects the JAX backend, as the conventional runs use.
# Imported the other way round, Veros silently runs on its numpy backend instead, with
# the same results but about half the speed (measured: 0.13 s against 0.056 s per day).
import global_4deg_no_cycle  # noqa: E402, F401

import coupled_temperature_velocity as ctv  # noqa: E402
from build_advection_matrix import SHAPE, get_mask_t  # noqa: E402
from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402
from patch_state import patch_state  # noqa: E402
from veros_model import interior, l2_norm, read_interior_tau, remove_netcdf, setup_model  # noqa: E402


class Scorer:
    """Steps the model one day at a time and returns the temperature score of each day
    and the RMS distance (total, surface, bottom) from the 200-year state."""

    def __init__(self, model):
        self.model, self.state = model, model.state
        self.mask = get_mask_t() > 0
        self.T_eq = temperature_from_restart(EQUILIBRIUM_FILE).reshape(SHAPE) if os.path.exists(EQUILIBRIUM_FILE) else None
        self.norm_0 = l2_norm(read_interior_tau(INIT_FILE, "temp"))
        self.prev = self.temperature()

    def temperature(self):
        vs = self.state.variables
        return interior(np.array(vs.temp[..., int(vs.tau)])).copy()

    def distance(self, T=None):
        if self.T_eq is None:
            return np.nan, np.nan, np.nan
        d = np.where(self.mask, (self.prev if T is None else T) - self.T_eq, 0.0)
        by_level = np.sqrt((d**2).sum(axis=(0, 1)) / np.maximum(self.mask.sum(axis=(0, 1)), 1))
        return np.sqrt((d**2).sum() / self.mask.sum()), by_level[-1], by_level[0]

    def reset(self):
        self.prev = self.temperature()

    def step(self):
        self.model.step(self.state)
        now = self.temperature()
        score = l2_norm(now - self.prev) / (l2_norm(self.prev) + self.norm_0)
        self.prev = now
        return score


def warm_up(n_days, tag, say):
    """Veros from the initial state for n_days; writes the restart file the solve reads.
    Returns (restart file, daily scores, wall seconds)."""
    restart_file = os.path.join(RESTART_DIR, f"warmup_{n_days}d{tag}.restart.h5")
    identifier = f"from_init_warmup_{n_days}{tag}"
    t0 = time.time()
    model = setup_model(identifier, INIT_FILE, max(n_days, 1), snap_out_freq=86400.0 * 1e6)
    t_setup = time.time() - t0
    scorer = Scorer(model)
    t0 = time.time()
    scores = [scorer.step() for _ in range(n_days)]
    wall = time.time() - t0
    from veros import restart

    with model.state.settings.unlock():
        model.state.settings.restart_output_filename = restart_file
    restart.write_restart(model.state, force=True)
    remove_netcdf(identifier)
    dist = scorer.distance()
    say(f"warm-up {n_days} days: {wall:.1f}s (+ {t_setup:.1f}s model setup); rms(T - T_200yr) {dist[0]:.3f} degC" + (f"; last score {scores[-1]:.2e}" if scores else ""))
    return restart_file, scores, wall


def solve(restart_file, n_steps, dt_first, growth, max_change, max_change_convecting, closures, say, years=None):
    """The coupled pseudo-time solve from restart_file: n_steps accepted steps, or, if
    `years` is given, as many steps as it takes to cover that much pseudo-time. Returns (problem, T after each
    accepted step, cumulative wall seconds at each accepted step, history)."""
    t0 = time.time()
    problem = ctv.CoupledProblem.from_restart(restart_file, closures=closures)
    T0 = temperature_from_restart(restart_file)
    T_norm = np.linalg.norm(T0[problem.ocean]) + np.linalg.norm(temperature_from_restart(INIT_FILE)[problem.ocean])
    T_steps, wall = [], []

    def monitor(T):
        T_steps.append(T.reshape(SHAPE).copy())
        wall.append(time.time() - t0)
        return ""

    _, history = ctv.pseudo_time_solve(
        problem, T0, T_norm, dt_first=dt_first, growth=growth, max_change=max_change,
        max_change_convecting=max_change_convecting,
        max_steps=30 if years is not None else 3 * n_steps + 6, max_accepted=None if years is not None else n_steps,
        max_elapsed=years, monitor=monitor, log=lambda s: say("  " + s.strip()),
    )  # fmt: skip
    return problem, T_steps, wall, history


def run_after_solve(restart_file, warmup_days, problem, T, post_days, confirm, name, say):
    """Veros from the warm-up state with the coupled solution patched in (T = None: no
    patch). Returns a dict of results; writes the daily series to from_init/<name>.csv."""
    identifier = f"from_init_{name}"
    model = setup_model(identifier, restart_file, post_days, snap_out_freq=86400.0 * 1e6)
    t0 = time.time()
    if T is not None:
        problem.update_closures(T)
        patch_state(model.state, model, T, problem.velocity(T), {})
    wall_patch = time.time() - t0
    scorer = Scorer(model)
    out = {"rms_start": scorer.distance()[0], "first": None, "above_after_first": 0, "wall_to_first": None}
    csv = open(os.path.join(OUT_DIR, f"{name}.csv"), "w", buffering=1)
    csv.write("total_day,score_temp,rms_total,rms_surface,rms_bottom\n")
    t0 = time.time()
    run_below = 0
    for day in range(1, post_days + 1):
        try:
            score = scorer.step()
        except RuntimeError as exc:
            say(f"  !!! Veros stopped on day {day} after the solve: {exc}")
            out["failed"] = True
            break
        dist = scorer.distance()
        csv.write(f"{warmup_days + day},{score:.8e},{dist[0]:.5f},{dist[1]:.5f},{dist[2]:.5f}\n")
        if score < THRESHOLD:
            run_below += 1
            if out["first"] is None:
                out.update(first=warmup_days + day, wall_to_first=wall_patch + time.time() - t0, rms_first=dist[0])
        else:
            run_below = 0
            if out["first"] is not None:
                out["above_after_first"] += 1
        if run_below >= confirm:
            break
    out.update(last_day=warmup_days + day, rms_last=scorer.distance()[0], wall_post=wall_patch + time.time() - t0)
    csv.close()
    remove_netcdf(identifier)
    return out


def run_cycles(cycles, args, tag, say):
    """
    Several rounds of [Veros for n days, then a coupled solve of k steps], in one model
    run: cycles is a list of (n_days, k_steps, dt_first). Each solve takes its closure
    fields from the model state at that moment, so later solves see turbulence energies
    that have adjusted to the earlier solution. After the last solve Veros runs on until
    the threshold is held. Writes from_init/cycles<tag>.csv and a summary line.
    """
    from veros import restart

    t_start = time.time()
    name = f"cycles{tag}"
    restart_file = os.path.join(RESTART_DIR, f"{name}.restart.h5")
    model = setup_model(f"from_init_{name}", INIT_FILE, sum(c[0] for c in cycles) + args.post_days, snap_out_freq=86400.0 * 1e6)
    scorer = Scorer(model)
    csv = open(os.path.join(OUT_DIR, f"{name}.csv"), "w", buffering=1)
    csv.write("total_day,score_temp,rms_total,rms_surface,rms_bottom\n")
    day, wall_veros, wall_solve, n_solves, pseudo_years = 0, 0.0, 0.0, 0, []
    states = {}  # --save-states: temperature after each patch, at the crossing, yearly afterwards, and at the end
    restart_written = []

    def advance(n, stop_when_held=False):
        nonlocal day, wall_veros
        first, run_below, above_after = None, 0, 0
        t0 = time.time()
        for _ in range(n):
            score = scorer.step()
            day += 1
            dist = scorer.distance()
            csv.write(f"{day},{score:.8e},{dist[0]:.5f},{dist[1]:.5f},{dist[2]:.5f}\n")
            if score < THRESHOLD:
                run_below += 1
                if first is None:
                    first = (day, wall_veros + time.time() - t0, dist[0])
                    if args.save_states:
                        states[f"crossing_day{day:05d}"] = scorer.temperature()
                    if stop_when_held and args.restart_out:
                        with model.state.settings.unlock():
                            model.state.settings.restart_output_filename = args.restart_out
                        restart.write_restart(model.state, force=True)
                        restart_written.append(day)
                        say(f"day {day}: score {score:.2e} < 1e-5; restart file written to {args.restart_out}")
            else:
                run_below = 0
                above_after += first is not None
            if args.save_states and day % 360 == 0:
                states[f"day{day:05d}"] = scorer.temperature()
            if stop_when_held and run_below >= args.confirm:
                break
        wall_veros += time.time() - t0
        return first, above_after

    for n_days, k, dt_first in cycles:
        years = float(k[:-1]) if k.endswith("y") else None
        advance(n_days)
        with model.state.settings.unlock():
            model.state.settings.restart_output_filename = restart_file
        restart.write_restart(model.state, force=True)
        say(f"day {day}: coupled solve, {k} {'(pseudo-years)' if years else 'steps'} from {dt_first:g} years; rms(T - T_200yr) before {scorer.distance()[0]:.3f}")
        problem, T_steps, wall, history = solve(
            restart_file, 0 if years else int(k), dt_first, args.growth, args.max_change, args.max_change_convecting, args.closures, say, years=years
        )
        n_solves += len(wall) - 1
        pseudo_years.append(history[-1][0])
        wall_solve += wall[-1]
        problem.update_closures(T_steps[-1])
        patch_state(model.state, model, T_steps[-1], problem.velocity(T_steps[-1]), {})
        scorer.reset()
        say(f"day {day}: patched; rms(T - T_200yr) after {scorer.distance()[0]:.3f}")
        if args.save_states:
            states[f"patched_day{day:05d}"] = scorer.temperature()
    first, above_after = advance(args.post_days, stop_when_held=True)
    csv.close()
    if args.save_states:
        states[f"end_day{day:05d}"] = scorer.temperature()
        np.savez_compressed(os.path.join(OUT_DIR, f"{name}_states.npz"), **{k: np.asarray(v, dtype=np.float32) for k, v in states.items()})
        say(f"saved {len(states)} temperature fields to {name}_states.npz")
    remove_netcdf(f"from_init_{name}")
    spec = ",".join(f"{n}:{k}:{d:g}" for n, k, d in cycles)
    line = (
        f"cycles {spec}  closures {args.closures}  growth {args.growth:g}  first below 1e-5: total day {first[0] if first else None}  "
        f"(days above afterwards: {above_after}, run to day {day})  rms(T - T_200yr) at crossing {first[2] if first else float('nan'):.3f}  "
        f"wall: Veros {first[1] if first else float('nan'):.0f}s + solves {wall_solve:.0f}s  (Veros {wall_veros / day:.3f} s/day; "
        f"pseudo-years reached {'/'.join(f'{y:.0f}' for y in pseudo_years)}; accepted steps {n_solves}); total wall time {time.time() - t_start:.0f}s"
    )
    say(line)
    with open(os.path.join(OUT_DIR, "summary.txt"), "a") as f:
        f.write(line + (f"  tag {args.tag}" if args.tag else "") + "\n")
    if args.restart_out and not restart_written:
        say(f"!!! the score did not fall below 1e-5 within {args.post_days} days after the last round; no restart file written")
    return {"first": first, "restart_written": bool(restart_written), "wall_total": time.time() - t_start}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warmup", type=int, default=0, help="days of conventional Veros before the solve")
    ap.add_argument("--cycles", default="", help="several rounds instead of one: 'days:steps:dt_first,...' (e.g. 60:9:0.125,40:6:1)")
    ap.add_argument("--steps", default="5", help="accepted pseudo-time steps; comma-separated list allowed; 0 = no solve")
    ap.add_argument("--dt-first", type=float, default=1.0, help="first pseudo-time step, years")
    ap.add_argument("--growth", type=float, default=2.0, help="factor by which the step grows")
    ap.add_argument("--max-change", type=float, default=3.0, help="largest temperature change accepted in one step outside convecting columns, degC")
    ap.add_argument("--max-change-convecting", type=float, default=20.0, help="the same, inside convecting columns, degC")
    ap.add_argument("--closures", default="coefficients", choices=["coefficients", "energies"], help="what the solve holds fixed: the mixing coefficients, or the turbulence energies")
    ap.add_argument("--post-days", type=int, default=2500, help="cap on the days run after the solve")
    ap.add_argument("--confirm", type=int, default=30, help="consecutive days below the threshold before stopping")
    ap.add_argument("--tag", default="", help="suffix for the output names")
    ap.add_argument("--restart-out", default="", help="with --cycles: write the state on the first day below 1e-5 after the last round to this restart file")
    ap.add_argument("--save-states", action="store_true", help="with --cycles: save the temperature after each patch, at the first crossing, every 360 days and at the end (from_init/cycles<tag>_states.npz)")
    args = ap.parse_args(argv)
    if args.restart_out:
        args.restart_out = os.path.abspath(args.restart_out)
    steps = [int(k) for k in args.steps.split(",")]
    tag = f"_{args.tag}" if args.tag else ""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESTART_DIR, exist_ok=True)

    def say(msg):
        print(f"[w{args.warmup}{tag}] {msg}", flush=True)

    if args.cycles:
        cycles = [(int(n), k, float(d)) for n, k, d in (c.split(":") for c in args.cycles.split(","))]
        return run_cycles(cycles, args, tag, lambda msg: print(f"[cycles{tag}] {msg}", flush=True))

    restart_file, _, wall_warmup = warm_up(args.warmup, tag, say)
    sec_per_day = wall_warmup / args.warmup if args.warmup >= 100 else None

    problem, T_steps, wall_solve = None, [None], [0.0]
    if max(steps) > 0:
        say(f"coupled solve: {max(steps)} steps, first step {args.dt_first:g} years, growth {args.growth:g}")
        problem, T_steps, wall_solve, history = solve(restart_file, max(steps), args.dt_first, args.growth, args.max_change, args.max_change_convecting, args.closures, say)
        np.savez(os.path.join(OUT_DIR, f"solve_w{args.warmup}{tag}.npz"), T_steps=np.array(T_steps), history=np.array(history), wall=np.array(wall_solve))

    for k in steps:
        if k >= len(T_steps):
            say(f"steps {k}: the solve only reached {len(T_steps) - 1} accepted steps, skipped")
            continue
        name = f"w{args.warmup}_k{k}{tag}"
        r = run_after_solve(restart_file, args.warmup, problem, T_steps[k] if k > 0 else None, args.post_days, args.confirm, name, say)
        wall_total = None if r["first"] is None else wall_warmup + wall_solve[k] + r["wall_to_first"]
        if sec_per_day is None and r["wall_post"] and r["last_day"] > args.warmup:
            sec_per_day = r["wall_post"] / (r["last_day"] - args.warmup)
        line = (
            f"warmup {args.warmup:5d}  steps {k:2d}  dt_first {args.dt_first:g}  growth {args.growth:g}  "
            f"first below 1e-5: total day {r['first']}  (days above afterwards: {r['above_after_first']}, run to day {r['last_day']})  "
            f"rms(T - T_200yr): after solve {r['rms_start']:.3f}, at crossing {r.get('rms_first', float('nan')):.3f}  "
            f"wall: warm-up {wall_warmup:.0f}s + solve {wall_solve[k]:.0f}s + after {r['wall_to_first'] or float('nan'):.0f}s = "
            f"{wall_total or float('nan'):.0f}s  (conventional {BASELINE_DAY} days at this speed: {BASELINE_DAY * sec_per_day:.0f}s)"
        )
        say(line)
        with open(os.path.join(OUT_DIR, "summary.txt"), "a") as f:
            f.write(line + (f"  tag {args.tag}" if args.tag else "") + "\n")


if __name__ == "__main__":
    main()
