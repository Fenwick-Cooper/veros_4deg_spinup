#!/usr/bin/env python3
"""
Our method (rounds of [Veros n days, coupled pseudo-time solve, patch]; spinup_from_init.py),
optionally with Anderson acceleration over the rounds (method D).

Round r starts from the temperature x_r patched in at the end of round r-1 (x_0: the initial
state) and returns y_r, the temperature after its solve. The plain method takes
x_{r+1} = y_r. With --aa-m m > 0, from round --aa-start on (counted from 1) the next
temperature is the Anderson mixture of the rounds so far (krylov.Anderson, residual
g_r = y_r - x_r), patched in with the emulator's circulation for it.

    python code/newton_krylov_anderson/anderson_outer.py --tag L3                      # our competition spin-up
    python code/newton_krylov_anderson/anderson_outer.py --tag L2 --cycles 20:30y:0.125,20:60y:2,20:60y:4,20:2000y:32
    python code/newton_krylov_anderson/anderson_outer.py --tag D_plain --extra 3       # L3 + 3 more long rounds
    python code/newton_krylov_anderson/anderson_outer.py --tag D_aa3 --extra 3 --aa-m 3
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import RESTART_DIR, Run  # noqa: E402  (first: selects the JAX backend)
import numpy as np  # noqa: E402

import spinup_from_init as sfi  # noqa: E402
from build_advection_matrix import SHAPE  # noqa: E402
from krylov import Anderson  # noqa: E402

L3 = "10:30y:0.125,10:60y:2,10:60y:4,10:2000y:32"
LONG = "10:2000y:32"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cycles", default=L3)
    ap.add_argument("--extra", type=int, default=0, help="further long rounds appended")
    ap.add_argument("--long", default=LONG, help="the round appended by --extra")
    ap.add_argument("--aa-m", type=int, default=0, help="Anderson memory (0: plain rounds)")
    ap.add_argument("--aa-start", type=int, default=4, help="first round (from 1) whose output is mixed")
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args(argv)

    specs = args.cycles.split(",") + [args.long] * args.extra
    cycles = [(int(n), float(y.rstrip("y")), float(d)) for n, y, d in (c.split(":") for c in specs)]
    run = Run(args.tag, sum(c[0] for c in cycles) + 4000)
    run.note(f"rounds: {args}")
    aa = Anderson(m=args.aa_m) if args.aa_m > 0 else None
    x = run.temperature()  # temperature at the start of the round
    rounds = []
    for r, (n_days, years, dt_first) in enumerate(cycles, start=1):
        run.run_main(n_days)
        restart_file = os.path.join(RESTART_DIR, f"{args.tag}_round.restart.h5")
        run.write_restart(restart_file)
        problem, T_steps, wall, history = sfi.solve(restart_file, 0, dt_first, 2.0, 3.0, 20.0, "energies", lambda s: None, years=years)
        y = T_steps[-1].reshape(SHAPE)
        info = ""
        if aa is not None and r >= args.aa_start:
            x_new, d = aa.step(x[run.mask], y[run.mask])
            x_next = y.copy()
            x_next[run.mask] = x_new
            info = f"; Anderson (m used {d['m_used']}{', restarted' if d['restarted'] else ''}): rms of mixture {run.rms(x_next):.4f}"
        else:
            x_next = y
        problem.update_closures(x_next)
        run.set_temperature(x_next, problem.velocity(x_next))
        rounds.append({"round": r, "days_main": run.days_main, "wall": round(run.wall(), 1), "solves": len(history) - 1, "rms_solve": round(run.rms(y), 4), "rms_patched": round(run.rms(), 4)})
        run.note(f"round {r}: {years:g} pseudo-years, {len(history) - 1} steps; rms of solution {run.rms(y):.4f}{info}; patched {run.rms():.4f}")
        x = x_next
    params = vars(args)
    params["rounds"] = rounds
    return run.finish("ours" + (" + Anderson over rounds" if aa else ""), params)


if __name__ == "__main__":
    main()
