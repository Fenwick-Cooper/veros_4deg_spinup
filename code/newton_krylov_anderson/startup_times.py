#!/usr/bin/env python3
"""
Start-up time of each method run: the time a run's process takes that the harness clock
(common.Run, which starts after model set-up and compilation of the first step) does not
count. It is the run's process duration, from the start times run_all.sh writes to
run_all.log, minus the final_wall_s in summary.jsonl, and so also includes the final restart
write (about a second). compare.py and plot_compare.py add it to every wall time of the run,
so that all wall times are full process times.

Writes startup_s.json in RESULTS_DIR (data/intermediate/newton_krylov_anderson/).

    python code/newton_krylov_anderson/startup_times.py
"""

import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nka_paths import RESULTS_DIR  # noqa: E402


def main():
    starts = []
    for line in open(os.path.join(RESULTS_DIR, "run_all.log")):
        m = re.match(r"=== (\S+) (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})$", line.strip())
        if m:
            starts.append((m.group(1), datetime.datetime.fromisoformat(m.group(2))))
    final = {}
    for line in open(os.path.join(RESULTS_DIR, "summary.jsonl")):
        r = json.loads(line)
        final[r["name"]] = r["final_wall_s"]
    startup = {}
    for (name, t0), (_, t1) in zip(starts, starts[1:]):
        if name in final:
            startup[name] = round((t1 - t0).total_seconds() - final[name], 1)
    out = {"note": "process duration (from run_all.log) minus final_wall_s (from summary.jsonl)", "startup_s": startup}
    with open(os.path.join(RESULTS_DIR, "startup_s.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(startup))


if __name__ == "__main__":
    main()
