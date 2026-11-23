#!/usr/bin/env python3
"""Seconds per model day of plain Veros (production settings, diagnostics off), from
Init_state, over 720 days; written to veros_speed.json in RESULTS_DIR (data/intermediate/newton_krylov_anderson/). Run when nothing else
is running."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import RESULTS_DIR, Run  # noqa: E402

run = Run("speed", 800)
run.run_main(720)
out = {"seconds_per_day": run.wall() / 720, "days": 720}
print(out)
json.dump(out, open(os.path.join(RESULTS_DIR, "veros_speed.json"), "w"))
run.close()
os.remove(os.path.join(RESULTS_DIR, "speed.csv"))
