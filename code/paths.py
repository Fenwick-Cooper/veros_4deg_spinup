"""
Locations of every file this repository reads or writes, relative to the repository root
(the directory that holds competition_spin-up.py). Nothing outside the repository is used,
except the Veros source code named in veros_path.txt (see veros_path.py).
"""

import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(CODE_DIR)

DATA_DIR = os.path.join(REPO_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")  # the competition's files, never written to
INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")  # CSV, logs, NPZ; Veros's own NetCDF output
RESTART_DIR = os.path.join(DATA_DIR, "veros_restart")  # restart files written along the way
PLOTS_DIR = os.path.join(DATA_DIR, "plots")  # figures not used by the LaTeX documents
DESCRIPTION_DIR = os.path.join(REPO_DIR, "Description")  # figures used by the LaTeX documents go here

INIT_FILE = os.path.join(INPUT_DIR, "Init_state.restart.h5")
FORCING_FILE = os.path.join(INPUT_DIR, "forcing_4deg_global_annual_mean.nc")
ECMWF_FILE = os.path.join(INPUT_DIR, "ecmwf_4deg_monthly_nc4.nc")

# Veros is run in this directory, because it writes its NetCDF diagnostics into the current
# directory (they are empty here and are removed after each run)
VEROS_RUN_DIR = INTERMEDIATE_DIR

# the 2000-year reference run (run_2000yr_reference.py)
REFERENCE_DIR = os.path.join(INTERMEDIATE_DIR, "reference_2000yr")  # daily.csv, snapshots/, analysis.npz
REFERENCE_RESTART_DIR = os.path.join(RESTART_DIR, "reference_2000yr")  # reference_y<year>.restart.h5, checkpoint
# T_200, the state after 200 years of conventional Veros: year 200 of the reference run
# (identical to the 200-year run of the competition's Example_run_Veros.py)
T200_FILE = os.path.join(REFERENCE_RESTART_DIR, "reference_y0200.restart.h5")
T2000_FILE = os.path.join(REFERENCE_RESTART_DIR, "reference_y2000.restart.h5")

# the same run in single precision (run_2000yr_reference.py --float32; Description/single_precision_spinup.tex)
REFERENCE_FLOAT32_DIR = os.path.join(INTERMEDIATE_DIR, "reference_2000yr_float32")  # daily.csv, snapshots/, comparison.npz
REFERENCE_FLOAT32_RESTART_DIR = os.path.join(RESTART_DIR, "reference_2000yr_float32")


def make_dirs():
    for d in (INTERMEDIATE_DIR, RESTART_DIR, PLOTS_DIR):
        os.makedirs(d, exist_ok=True)


def _add_veros_to_path():
    """Put the Veros source code named in veros_path.txt on the Python path, if that file
    exists and is valid (silently; veros_path.ensure_veros() asks for it otherwise)."""
    import sys

    try:
        with open(os.path.join(REPO_DIR, "veros_path.txt")) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    except OSError:
        return
    if lines:
        path = os.path.abspath(os.path.expanduser(lines[0]))
        if os.path.isfile(os.path.join(path, "veros", "__init__.py")) and path not in sys.path:
            sys.path.insert(0, path)


_add_veros_to_path()
