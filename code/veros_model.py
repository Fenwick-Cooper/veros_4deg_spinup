"""
Setting up the Veros 4-degree model from a restart file, and small helpers for reading
restart files (taken from equilibration_experiment_fixed.py of the original project).

Veros writes its NetCDF diagnostics (<identifier>.averages.nc, <identifier>.snapshot.nc)
into the current directory, so the model is set up in data/intermediate; the snapshot
frequency passed by the callers is so long that these files stay empty, and they are
removed afterwards with remove_netcdf().
"""

import os

import h5py
import numpy as np

from paths import INTERMEDIATE_DIR

SNAP_OUT_FREQ = 86400 * 30


def read_interior_tau(restart_file, name):
    """The interior (no halos) of variable `name` at the current time level `tau`."""
    with h5py.File(restart_file, "r") as f:
        tau = int(f["core/tau"][()])
        arr = f[f"core/{name}"]
        if arr.ndim == 4:
            return np.array(arr[2:-2, 2:-2, :, tau])
        return np.array(arr[2:-2, 2:-2, tau])


def interior(a):
    a = np.asarray(a)
    if a.ndim == 3:
        return a[2:-2, 2:-2, :]
    return a[2:-2, 2:-2]


def l2_norm(x):
    return float(np.sqrt(np.sum(np.asarray(x) ** 2)))


def remove_netcdf(identifier):
    for suffix in ("averages.nc", "snapshot.nc"):
        leftover = os.path.join(INTERMEDIATE_DIR, f"{identifier}.{suffix}")
        if os.path.exists(leftover):
            os.remove(leftover)


def setup_model(identifier, restart_file, n_days, snap_out_freq=SNAP_OUT_FREQ):
    """A GlobalFourDegreeSetup started from restart_file, set up for n_days."""
    from global_4deg_no_cycle import GlobalFourDegreeSetup

    restart_file = os.path.abspath(restart_file)  # before changing directory
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(INTERMEDIATE_DIR)
    remove_netcdf(identifier)
    model = GlobalFourDegreeSetup(
        identifier,
        n_days * 86400.0,
        restart_file,
        snap_out_freq=snap_out_freq,
        restart_mode=True,
    )
    model.setup()
    os.chdir(cwd)
    return model
