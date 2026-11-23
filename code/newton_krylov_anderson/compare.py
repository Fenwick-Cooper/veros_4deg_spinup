#!/usr/bin/env python3
"""
The comparison table: reads summary.jsonl (one line per method run) and the conventional
curve (conventional_curve.npz) from RESULTS_DIR (data/intermediate/newton_krylov_anderson/),
writes comparison.md there and prints it.

Conventional Veros: criterion on day 3714 (from Init_state); distance to T_2000 from
conventional_curve.npz. All wall times are full process times: the methods'
harness times plus their start-up (startup_s.json, from startup_times.py); conventional Veros, the
measured full time of a 3714-day run with output off (conventional_timing.jsonl) extended at
the measured seconds per day (veros_speed.json).

    python code/newton_krylov_anderson/compare.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nka_paths import RESULTS_DIR  # noqa: E402

BASELINE_DAY = 3714
LEVELS = ("0.06", "0.05")
ORDER = ["L3", "L2", "D_plain", "D_aa3", "C_k0", "C_k5f", "A_m10", "B_k20"]
NAMES = {
    "L3": "Leap-and-settle, competition (L3)",
    "L2": "Leap-and-settle, equilibrium (L2)",
    "D_plain": "Leap-and-settle, L3 + 3 more long rounds",
    "D_aa3": "Leap-and-settle, L3 + 3 long rounds, Anderson (m=3)",
    "C_k0": "Leap-and-settle, Veros residual, defect correction",
    "C_k5f": "Leap-and-settle, Veros residual, NK (5 GMRES its)",
    "A_m10": "Anderson on Veros alone (tau 1 yr, m=10)",
    "B_k20": "Newton-Krylov on Veros alone (tau 1 yr, 20 GMRES)",
}


def fmt_wall(s):
    if s is None:
        return "-"
    return f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.1f} h"


def fmt_days(d):
    if d is None:
        return "-"
    return f"{d:,}" if d < 36000 else f"{d:,} ({d / 360:.0f} y)"


def main():
    speed = json.load(open(os.path.join(RESULTS_DIR, "veros_speed.json")))["seconds_per_day"]
    startup = json.load(open(os.path.join(RESULTS_DIR, "startup_s.json")))["startup_s"]
    timing = [json.loads(line) for line in open(os.path.join(RESULTS_DIR, "conventional_timing.jsonl"))]
    t_off = [t for t in timing if not t["output"]][-1]
    conv_startup = t_off["total_s"] - speed * t_off["days"]

    def conv_wall(days):
        return conv_startup + speed * days

    runs = {}
    for line in open(os.path.join(RESULTS_DIR, "summary.jsonl")):
        r = json.loads(line)
        runs[r["name"]] = r
    conv = np.load(os.path.join(RESULTS_DIR, "conventional_curve.npz"))
    years, r2000 = conv["days"] / 360, conv["rms"]

    head = (
        "| Method | criterion: main-trajectory day | criterion: Veros days integrated | criterion: wall | rms(T-T2000) at criterion "
        "| days integrated / wall to 0.06 °C | days integrated / wall to 0.05 °C | final rms |\n|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    cells = [f"{BASELINE_DAY:,}", f"{BASELINE_DAY:,}", fmt_wall(conv_wall(BASELINE_DAY)), f"≈{np.interp(BASELINE_DAY / 360, years, r2000):.2f}"]
    for lev in LEVELS:
        i = int(np.argmax(r2000 <= float(lev)))
        d = int(years[i] * 360)
        cells.append(f"{fmt_days(d)} / {fmt_wall(conv_wall(d))}")
    cells.append("0 (year 2000 itself)")
    rows.append("| Conventional Veros | " + " | ".join(cells) + " |")
    for name in ORDER + sorted(set(runs) - set(ORDER)):
        if name not in runs:
            continue
        r = runs[name]
        s0 = startup.get(name, 0.0)
        cells = [fmt_days(r["criterion_days_main"]), fmt_days(r["criterion_days_total"]), fmt_wall(r["criterion_wall_s"] + s0 if r["criterion_wall_s"] is not None else None),
                 f"{r['rms_at_criterion']:.3f}" if r["rms_at_criterion"] is not None else "-"]  # fmt: skip
        for lev in LEVELS:
            m = r["milestones"].get(lev)
            cells.append(f"{fmt_days(m['days_total'])} / {fmt_wall(m['wall_s'] + s0)}" if m else f"not reached ({fmt_wall(r['final_wall_s'] + s0)})")
        cells.append(f"{r['final_rms']:.3f}")
        rows.append(f"| {NAMES.get(name, name)} | " + " | ".join(cells) + " |")
    text = (
        f"Plain Veros: {speed:.4f} s per model day, start-up {conv_startup:.0f} s (measured, output off as in the method runs). "
        "Wall times are full process times, start-up included.\n\n" + head + "\n".join(rows) + "\n"
    )
    with open(os.path.join(RESULTS_DIR, "comparison.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
