#!/usr/bin/env python3
"""
Wall-clock time of a conventional Veros run from Init_state, from process start (imports,
model set-up and JIT compilation included) to the end of the last day:

    python code/newton_krylov_anderson/time_conventional.py --days 3714 --no-output   # as the method runs
    python code/newton_krylov_anderson/time_conventional.py --days 3714               # Veros's own output on

Appends one JSON line to conventional_timing.jsonl in RESULTS_DIR
(data/intermediate/newton_krylov_anderson/). compare.py and plot_compare.py use the last
line with the output off. Run when nothing else is running.
"""

import time

T_START = time.time()

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nka_paths import INIT_FILE, RESULTS_DIR, VEROS_RUN_DIR, make_nka_dirs  # noqa: E402  (puts code/ on the path)

import global_4deg_no_cycle  # noqa: E402,F401  (JAX backend; must come before veros.core)

from veros_model import remove_netcdf, setup_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=3714)
    ap.add_argument("--no-output", action="store_true", help="switch Veros's NetCDF diagnostics off")
    args = ap.parse_args()
    make_nka_dirs()
    ident = "nka_timing_" + ("off" if args.no_output else "on")
    model = setup_model(ident, INIT_FILE, args.days)
    if args.no_output:
        for diagnostic in model.state.diagnostics.values():
            diagnostic.sampling_frequency = 0.0
            diagnostic.output_frequency = 0.0
    t_setup = time.time() - T_START
    cwd = os.getcwd()
    os.chdir(VEROS_RUN_DIR)  # Veros writes its output and final restart into the current directory
    try:
        model.run()
    finally:
        os.chdir(cwd)
    total = time.time() - T_START
    out = {"days": args.days, "output": not args.no_output, "setup_s": round(t_setup, 1), "total_s": round(total, 1),
           "run_s_per_day": round((total - t_setup) / args.days, 5)}  # fmt: skip
    print(json.dumps(out))
    with open(os.path.join(RESULTS_DIR, "conventional_timing.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    remove_netcdf(ident)
    for f in glob.glob(os.path.join(VEROS_RUN_DIR, f"{ident}*.restart.h5")):
        os.remove(f)


if __name__ == "__main__":
    main()
