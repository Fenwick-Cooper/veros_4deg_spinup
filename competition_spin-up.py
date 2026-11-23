#!/usr/bin/env python3
"""
Competition spin-up of the Veros 4-degree competition setup (configuration L3 of
Description/coupled_spinup.tex), aimed at the earliest crossing of the competition criterion.

Starting from data/input/Init_state.restart.h5, Veros is run for 10 days followed by a
round of implicit pseudo-time stepping of the coupled temperature-velocity problem; this
is done four times (rounds of 30, 60, 60 and 2000 pseudo-years), and Veros is then run
until the temperature score first falls below 1e-5 (model day 63 in the documents).
The model state on that day is written to competition.restart.h5 next to this script.

    python competition_spin-up.py

Needs only this repository and the Veros source code, whose location is read from
veros_path.txt (you are asked for it if it is not set). Intermediate files go to
data/intermediate and data/veros_restart.
"""

import os
import sys
import time

T_START = time.time()
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_DIR, "code"))

from paths import ECMWF_FILE, FORCING_FILE, INIT_FILE, make_dirs  # noqa: E402
from veros_path import ensure_veros  # noqa: E402

CYCLES = "10:30y:0.125,10:60y:2,10:60y:4,10:2000y:32"
OUTPUT = os.path.join(REPO_DIR, "competition.restart.h5")


JAX_WARNING = (
    "WARNING: {}. Veros will run on its NumPy backend instead: the results should be the "
    "same, but each Veros time step takes about twice as long, so the wall time will be "
    "longer than reported. Install JAX in this Python environment (e.g. pip install "
    "veros[jax]) to use the JAX backend."
)


def check_jax_installed():
    """Warn if JAX is not installed (Veros then falls back to NumPy)."""
    import importlib.util

    if importlib.util.find_spec("jax") is None:
        print(JAX_WARNING.format("JAX is not installed"), flush=True)


def check_jax_in_use():
    """Warn if Veros, once imported, is not running on the JAX backend (when JAX is
    installed; check_jax_installed has already warned otherwise)."""
    import importlib.util

    from veros import runtime_settings

    if importlib.util.find_spec("jax") is None:
        return

    if runtime_settings.backend != "jax":
        print(JAX_WARNING.format(f"Veros is using its {runtime_settings.backend} backend, not JAX"), flush=True)


def main():
    check_jax_installed()
    for f in (INIT_FILE, FORCING_FILE, ECMWF_FILE):
        if not os.path.isfile(f):
            sys.exit(f"missing input file {f}")
    veros_dir = ensure_veros()
    print(f"Veros source code: {veros_dir}", flush=True)
    make_dirs()
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)

    import spinup_from_init  # noqa: E402  (imports Veros)

    check_jax_in_use()

    result = spinup_from_init.main(
        ["--cycles", CYCLES, "--closures", "energies", "--post-days", "3600", "--confirm", "1",
         "--restart-out", OUTPUT, "--tag", "competition"]
    )  # fmt: skip
    wall = time.time() - T_START
    if not result["restart_written"]:
        sys.exit(f"the score did not fall below 1e-5; no restart file written (wall time {wall / 60:.1f} minutes)")
    print(
        f"\nCompetition spin-up done: score below 1e-5 on model day {result['first'][0]}; "
        f"state written to {OUTPUT}; total wall time {wall:.0f} s ({wall / 60:.1f} minutes)",
        flush=True,
    )


if __name__ == "__main__":
    main()
