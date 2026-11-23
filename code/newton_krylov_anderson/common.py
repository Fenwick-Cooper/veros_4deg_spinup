"""
Shared machinery for the comparison of spin-up methods (Anderson acceleration, Newton-Krylov,
with and without the coupled emulator of the repository's code/ directory).

Run   one Veros model (the competition's 4-degree setup, production settings: JAX backend,
      SciPy streamfunction solver) plus the bookkeeping every method needs:

        days_total  every Veros day integrated, including the runs a method throws away
                    (Jacobian-vector products, residual evaluations). This is the cost.
        days_main   the days of the one continuous model trajectory (what a conventional
                    run, or the competition, would see). The emulator's work and thrown-away
                    runs are not in it.
        wall        seconds since the method started (model set-up and JIT compilation of
                    the first step are excluded, as for the earlier timings).

      The temperature score of every main day is computed exactly as in the competition,

          score = || T_today - T_yesterday || / ( || T_yesterday || + || T_init || ),

      and the competition criterion counts as met on the first main day with score < 1e-5
      that is followed by CONFIRM more main days below 1e-5 without the state being changed
      by the method in between (a patch resets the count).

      snapshot() / restore() keep and put back the complete model state (all Veros variables
      are immutable JAX arrays, so a snapshot is a dict of references and costs nothing).

Distances are RMS(T - T_2000) over ocean cells, T_2000 = year 2000 of the reference run.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# nka_paths also puts code/ on the Python path; it imports nothing from Veros
from nka_paths import INIT_FILE, RESULTS_DIR, T2000_FILE  # noqa: E402
from nka_paths import NKA_RESTART_DIR as RESTART_DIR  # noqa: E402
from nka_paths import make_nka_dirs as make_dirs  # noqa: E402

import global_4deg_no_cycle  # noqa: E402,F401  (selects the JAX backend; must come first)

import numpy as np  # noqa: E402

from build_advection_matrix import SHAPE, get_mask_t  # noqa: E402
from build_vertical_mixing_matrix import temperature_from_restart  # noqa: E402
from patch_state import patch_state  # noqa: E402
from veros_model import interior, l2_norm, read_interior_tau, remove_netcdf, setup_model  # noqa: E402

THRESHOLD = 1e-5
CONFIRM = 30
DAYS_PER_YEAR = 360


class Run:
    def __init__(self, name, max_days, restart_file=INIT_FILE, log_every=1):
        make_dirs()
        self.name = name
        self.model = setup_model(f"nka_{name}", restart_file, max_days, snap_out_freq=86400.0 * 1e6)
        self.state = self.model.state
        self.vs = self.state.variables
        # Veros's NetCDF diagnostics are not used here; switched off, as they would also keep
        # references to fields between steps (which breaks jax.jvp through the step)
        for diagnostic in self.state.diagnostics.values():
            diagnostic.sampling_frequency = 0.0
            diagnostic.output_frequency = 0.0
        self.mask = get_mask_t().reshape(SHAPE) > 0
        self.ocean = self.mask.ravel()
        self.T2000 = temperature_from_restart(T2000_FILE).reshape(SHAPE)
        self.norm_0 = l2_norm(read_interior_tau(INIT_FILE, "temp"))
        self.days_total = self.days_main = 0
        self.prev = self.temperature()
        self.below, self.candidate, self.crossing = 0, None, None
        self.milestones = {}  # rms level -> (days_total, days_main, wall) when first reached on the main trajectory
        self.log_every = log_every
        self.csv = open(os.path.join(RESULTS_DIR, f"{name}.csv"), "w", buffering=1)
        self.csv.write("wall,days_total,days_main,event,score,rms_T2000\n")
        self.notes = []
        # compile the step outside the timed part: one step, then put the state back
        snap = self.snapshot()
        self.model.step(self.state)
        self.restore(snap)
        self.t0 = time.time()

    # ---- state ----------------------------------------------------------------------
    def temperature(self):
        return interior(np.array(self.vs.temp[..., int(self.vs.tau)])).copy()

    def rms(self, T=None):
        T = self.temperature() if T is None else T.reshape(SHAPE)
        d = np.where(self.mask, T - self.T2000, 0.0)
        return float(np.sqrt((d**2).sum() / self.mask.sum()))

    def snapshot(self):
        return dict(self.vs.items())

    def restore(self, snap):
        with self.vs.unlock():
            for k, v in snap.items():
                setattr(self.vs, k, v)

    def wall(self):
        return time.time() - self.t0

    # ---- stepping -------------------------------------------------------------------
    def step_main(self):
        self.model.step(self.state)
        self.days_total += 1
        self.days_main += 1
        now = self.temperature()
        if not np.all(np.isfinite(now[self.mask])):
            raise FloatingPointError(f"{self.name}: non-finite temperature on main day {self.days_main}")
        score = l2_norm(now - self.prev) / (l2_norm(self.prev) + self.norm_0)
        self.prev = now
        rms = self.rms(now)
        wall = self.wall()
        if score < THRESHOLD:
            if self.below == 0:
                self.candidate = (self.days_total, self.days_main, wall, rms)
            self.below += 1
            if self.below > CONFIRM and self.crossing is None:
                self.crossing = self.candidate
        else:
            self.below = 0
        for level in (0.10, 0.07, 0.06, 0.05):
            if rms <= level and level not in self.milestones:
                self.milestones[level] = (self.days_total, self.days_main, wall)
        if self.days_main % self.log_every == 0 or score < THRESHOLD * 3:
            self.csv.write(f"{wall:.2f},{self.days_total},{self.days_main},day,{score:.6e},{rms:.5f}\n")
        return score

    def run_main(self, n, stop_when_crossed=False):
        scores = []
        for _ in range(n):
            scores.append(self.step_main())
            if stop_when_crossed and self.crossing is not None:
                break
        return scores

    def run_aux(self, n):
        """n days that are thrown away afterwards (the caller restores a snapshot)."""
        for _ in range(n):
            self.model.step(self.state)
        self.days_total += n
        return self.temperature()

    def set_temperature(self, T, uvw=None, event="patch"):
        """Patch T (and, if given, the circulation (u, v, w)) into the model; see patch_state."""
        patch_state(self.state, self.model, np.asarray(T).reshape(SHAPE), uvw, {})
        self.prev = self.temperature()
        self.below = 0
        self.csv.write(f"{self.wall():.2f},{self.days_total},{self.days_main},{event},,{self.rms():.5f}\n")

    def note(self, text):
        line = f"[{self.name} {self.wall():7.1f}s, days {self.days_total}/{self.days_main}] {text}"
        print(line, flush=True)
        self.notes.append(line)

    def write_restart(self, path):
        from veros import restart

        with self.state.settings.unlock():
            self.state.settings.restart_output_filename = path
        restart.write_restart(self.state, force=True)

    # ---- the end ----------------------------------------------------------------------
    def close(self):
        """Close the CSV and remove Veros's (empty) NetCDF output, without a summary line."""
        self.csv.close()
        remove_netcdf(f"nka_{self.name}")

    def finish(self, method, params, post_days=int(os.environ.get("NKA_POST_DAYS", 3600))):
        """Run plain Veros on from the final state until the criterion holds (at most
        post_days), then write the summary line."""
        if self.crossing is None:
            self.note("final state: running plain Veros until the criterion holds")
            self.run_main(post_days, stop_when_crossed=True)
        restart_file = os.path.join(RESTART_DIR, f"{self.name}_final.restart.h5")
        self.write_restart(restart_file)
        c = self.crossing
        out = {
            "name": self.name,
            "method": method,
            "params": params,
            "criterion_days_total": c[0] if c else None,
            "criterion_days_main": c[1] if c else None,
            "criterion_wall_s": round(c[2], 1) if c else None,
            "rms_at_criterion": round(c[3], 4) if c else None,
            "final_days_total": self.days_total,
            "final_days_main": self.days_main,
            "final_wall_s": round(self.wall(), 1),
            "final_rms": round(self.rms(), 4),
            "milestones": {f"{k:.2f}": {"days_total": v[0], "days_main": v[1], "wall_s": round(v[2], 1)} for k, v in sorted(self.milestones.items(), reverse=True)},
        }
        self.note("summary " + json.dumps(out))
        with open(os.path.join(RESULTS_DIR, "summary.jsonl"), "a") as f:
            f.write(json.dumps(out) + "\n")
        self.csv.close()
        remove_netcdf(f"nka_{self.name}")
        return out


def ocean_vector(run, T):
    return np.asarray(T).reshape(SHAPE)[run.mask]


def full_field(run, x, fill):
    T = np.array(fill, dtype=float).reshape(SHAPE).copy()
    T[run.mask] = x
    return T
