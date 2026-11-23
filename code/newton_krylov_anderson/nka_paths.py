"""
Locations used by the Newton-Krylov / Anderson comparison. Every location comes from the
repository's code/paths.py; this module only adds the two subdirectories the comparison
writes to, and puts code/ on the Python path so that the scripts here can import the
repository's modules (global_4deg_no_cycle, patch_state, spinup_from_init, ...).

    RESULTS_DIR      data/intermediate/newton_krylov_anderson/   CSV, logs, summaries, NPZ, JSON
    NKA_RESTART_DIR  data/veros_restart/newton_krylov_anderson/  restart files written on the way

Inputs read: the competition's files (data/input/) and the 2000-year reference run
(T_2000 = data/veros_restart/reference_2000yr/reference_y2000.restart.h5 and
data/intermediate/reference_2000yr/analysis.npz, from code/run_2000yr_reference.py and
code/analyse_2000yr_reference.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from paths import (  # noqa: E402,F401
    DESCRIPTION_DIR,
    INIT_FILE,
    INTERMEDIATE_DIR,
    PLOTS_DIR,
    REFERENCE_DIR,
    RESTART_DIR,
    T2000_FILE,
    VEROS_RUN_DIR,
    make_dirs,
)

RESULTS_DIR = os.path.join(INTERMEDIATE_DIR, "newton_krylov_anderson")
NKA_RESTART_DIR = os.path.join(RESTART_DIR, "newton_krylov_anderson")


def make_nka_dirs():
    make_dirs()
    for d in (RESULTS_DIR, NKA_RESTART_DIR):
        os.makedirs(d, exist_ok=True)
