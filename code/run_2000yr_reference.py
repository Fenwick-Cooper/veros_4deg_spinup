#!/usr/bin/env python3
"""
The 2000-year conventional reference run of the 4-degree model
(Description/coupled_spinup.tex, Section "The 2000-year reference run").
Starts from data/input/Init_state.restart.h5 (day 0) and steps Veros for 2000 years of
360 days. Output, kept well below 3 GB in total:
  in data/intermediate/reference_2000yr/
    daily.csv                       every day: the temperature score and the volume-mean
                                    temperature
    snapshots/day_<day>.npz         every 10 years: temp, u, v, w, psi, tke, eke (interior,
                                    float32; about 1.3 MB each, 200 files)
    run.log, checkpoint.day
  in data/veros_restart/reference_2000yr/
    reference_y<year>.restart.h5    every 100 years: a full Veros restart file (12 MB, 20
                                    files); reference_y2000.restart.h5 is T_2000 and
                                    reference_y0200.restart.h5 is T_200
    checkpoint.restart.h5           the latest 10-year checkpoint, overwritten

The run can be interrupted at any time (or the computer restarted): the same command
continues from the last checkpoint, losing at most 10 years of computing (about
2 minutes). Veros's own netCDF output is switched off. It takes about 6 hours on an
Apple M3 Pro; on a Mac, run it under `caffeinate -i` so that the computer does not sleep:
    caffeinate -i python code/run_2000yr_reference.py

--float32 runs the same in single precision (Description/single_precision_spinup.tex),
with output in data/intermediate/reference_2000yr_float32/ and
data/veros_restart/reference_2000yr_float32/. It sets VEROS_FLOAT_TYPE=float32 and
VEROS_USE_SPECIAL_TDMA=0, because Veros's compiled tridiagonal solver rejects single
precision. compare_2000yr_float32.py and plot_2000yr_float32.py compare the two runs.

Options for testing: --years, --checkpoint-days, --keep-days.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DAYS_PER_YEAR = 360

sys.path.insert(0, HERE)

# single precision must be chosen before anything imports veros
FLOAT32 = "--float32" in sys.argv[1:]
if FLOAT32:
    os.environ["VEROS_FLOAT_TYPE"] = "float32"
    os.environ["VEROS_USE_SPECIAL_TDMA"] = "0"

from paths import INIT_FILE, REFERENCE_DIR, REFERENCE_FLOAT32_DIR, REFERENCE_FLOAT32_RESTART_DIR, REFERENCE_RESTART_DIR  # noqa: E402
from veros_path import ensure_veros  # noqa: E402

ensure_veros()

# the setup first: it selects the JAX backend before anything imports veros.core
import global_4deg_no_cycle  # noqa: E402, F401

from build_advection_matrix import COST, DZT, get_mask_t  # noqa: E402
from veros_model import interior, l2_norm, read_interior_tau, remove_netcdf, setup_model  # noqa: E402


def write_restart(model, path):
    from veros import restart

    with model.state.settings.unlock():
        model.state.settings.restart_output_filename = path
    restart.write_restart(model.state, force=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=float, default=2000)
    ap.add_argument("--checkpoint-days", type=int, default=10 * DAYS_PER_YEAR)
    ap.add_argument("--keep-days", type=int, default=100 * DAYS_PER_YEAR)
    ap.add_argument("--float32", action="store_true", help="single precision (see above)")
    args = ap.parse_args()
    last_day = int(round(args.years * DAYS_PER_YEAR))
    out, restart_dir = (REFERENCE_FLOAT32_DIR, REFERENCE_FLOAT32_RESTART_DIR) if args.float32 else (REFERENCE_DIR, REFERENCE_RESTART_DIR)
    os.makedirs(os.path.join(out, "snapshots"), exist_ok=True)
    os.makedirs(restart_dir, exist_ok=True)
    csv_path = os.path.join(out, "daily.csv")
    checkpoint = os.path.join(restart_dir, "checkpoint.restart.h5")
    day_file = os.path.join(out, "checkpoint.day")
    log = open(os.path.join(out, "run.log"), "a", buffering=1)

    def say(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")

    day = int(open(day_file).read()) if os.path.exists(checkpoint) and os.path.exists(day_file) else 0
    if day >= last_day:
        say(f"already at day {day}")
        return
    if day == 0:
        with open(csv_path, "w") as f:
            f.write("day,score_temp,mean_T\n")
    else:
        rows = open(csv_path).read().splitlines()
        with open(csv_path, "w") as f:  # drop days written after the checkpoint
            f.write("\n".join(rows[: day + 1]) + "\n")
    start_file = checkpoint if day else INIT_FILE
    say(f"start at day {day} from {os.path.basename(start_file)}; run to day {last_day}")

    identifier = "reference_2000yr"
    model = setup_model(identifier, start_file, last_day - day, snap_out_freq=86400.0 * 1e7)
    state = model.state
    vs = state.variables
    say(f"precision {vs.temp.dtype}")

    mask = get_mask_t() > 0
    volume = np.where(mask, COST[np.newaxis, :, np.newaxis] * DZT[np.newaxis, np.newaxis, :], 0.0)
    norm_T0 = l2_norm(read_interior_tau(INIT_FILE, "temp"))

    def temperature():
        return interior(np.array(vs.temp[..., int(vs.tau)])).copy()

    def snapshot(d):
        tau = int(vs.tau)
        fields = {name: interior(np.array(getattr(vs, name)[..., tau])).astype(np.float32) for name in ("temp", "u", "v", "w", "tke", "eke")}
        fields["psi"] = interior(np.array(vs.psi[..., tau])).astype(np.float32)
        np.savez_compressed(os.path.join(out, "snapshots", f"day_{d:06d}.npz"), day=d, **fields)

    if day == 0:
        snapshot(0)
    prev = temperature()
    csv = open(csv_path, "a", buffering=1)
    t0, t_last, day0 = time.time(), time.time(), day
    while day < last_day:
        model.step(state)
        day += 1
        now = temperature()
        score = l2_norm(now - prev) / (l2_norm(prev) + norm_T0)
        mean_T = (now * volume).sum() / volume.sum()
        csv.write(f"{day},{score:.8e},{mean_T:.8f}\n")
        prev = now
        if day % args.checkpoint_days == 0 or day == last_day:
            csv.flush()
            snapshot(day)
            write_restart(model, checkpoint + ".tmp")
            os.replace(checkpoint + ".tmp", checkpoint)
            with open(day_file + ".tmp", "w") as f:
                f.write(str(day))
            os.replace(day_file + ".tmp", day_file)
            if day % args.keep_days == 0 or day == last_day:
                kept = os.path.join(restart_dir, f"reference_y{day // DAYS_PER_YEAR:04d}.restart.h5" if day % DAYS_PER_YEAR == 0 else f"reference_d{day:06d}.restart.h5")
                with open(checkpoint, "rb") as src, open(kept, "wb") as dst:
                    dst.write(src.read())
            rate = (time.time() - t0) / (day - day0)
            say(
                f"day {day:6d} (year {day / DAYS_PER_YEAR:7.1f}): score {score:.3e}, "
                f"mean T {mean_T:.5f}; {rate:.4f} s/day, "
                f"{(last_day - day) * rate / 3600:.1f} h to go"
            )
            t_last = time.time()
    csv.close()
    remove_netcdf(identifier)
    say(f"done: day {day}")


if __name__ == "__main__":
    main()
