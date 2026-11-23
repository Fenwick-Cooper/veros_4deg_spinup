#!/usr/bin/env python3
"""
Method A: Anderson acceleration of Veros alone (as in Khatiwala 2024, Sci. Adv. 10, eadn2839).

The fixed-point map is f(T) = the temperature after tau days of Veros, started from the
current model state with T patched in (patch_state, temperature only: density, N^2 and the
Adams-Bashforth tendencies follow T; the circulation and turbulence are those the model has).
Each iteration integrates tau days on the main trajectory, then patches in the Anderson
mixture of the iterations so far. No emulator, no matrices.

    python code/newton_krylov_anderson/anderson_veros.py --tag A_m10 --tau 360 --m 10
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import DAYS_PER_YEAR, Run  # noqa: E402
import numpy as np  # noqa: E402

from krylov import Anderson  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tau", type=int, default=360, help="days of Veros per iteration")
    ap.add_argument("--m", type=int, default=10, help="Anderson memory")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=0, help="plain Veros days before the first iteration")
    ap.add_argument("--max-years", type=float, default=300.0)
    ap.add_argument("--max-wall", type=float, default=4500.0, help="seconds")
    ap.add_argument("--clip", type=float, default=5.0, help="largest change of any cell by one Anderson step beyond f(x), degC")
    args = ap.parse_args(argv)

    max_days = int(args.max_years * DAYS_PER_YEAR)
    run = Run(args.tag, max_days + 4000, log_every=10)
    run.note(f"method A: {args}")
    run.run_main(args.warmup)
    aa = Anderson(m=args.m, beta=args.beta)
    x = run.temperature()[run.mask]
    k = 0
    while run.days_total + args.tau <= max_days and run.wall() < args.max_wall:
        run.run_main(args.tau)
        fx = run.temperature()[run.mask]
        x_new, d = aa.step(x, fx)
        excess = x_new - fx
        big = np.abs(excess).max()
        if big > args.clip:  # keep the extrapolation within a few degrees of the model's own state
            x_new = fx + excess * (args.clip / big)
        T = run.temperature()
        T[run.mask] = x_new
        run.set_temperature(T)
        k += 1
        run.note(
            f"iteration {k}: |f(x) - x| {d['residual']:.3e}, m used {d['m_used']}{', restarted' if d['restarted'] else ''}, "
            f"max |x_new - f(x)| {big:.2f}; rms f(x) {run.rms(_full(run, fx)):.4f}, rms x_new {run.rms():.4f}"
        )
        x = x_new
    return run.finish("A: Anderson on Veros alone", vars(args))


def _full(run, x):
    T = np.zeros(run.mask.shape)
    T[run.mask] = x
    return T


if __name__ == "__main__":
    main()
