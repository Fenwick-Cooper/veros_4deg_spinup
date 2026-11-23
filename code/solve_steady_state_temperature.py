#!/usr/bin/env python3
"""
Combine every temperature matrix in this directory -- explicit (advection M, isoneutral
E, GM skew F, surface relaxation R) and implicit (isoneutral K_33, D_33; vertical mixing
D) -- into a single matrix A_T, build the surface forcing vector b, and solve

    A_T . T = -b

for the steady-state temperature T, i.e. the temperature at which the combined tendency
dT/dt = A_T T + b is exactly zero everywhere.

This is a genuinely different equation from build_combined_step.py's one time-step
equation: "implicit" there meant *how Veros steps the term in time* (solved alongside an
identity matrix for numerical stability), not a different physical term. The underlying
continuous tendency sums every term linearly, so

    A_T = M + E + D_33 + F + D + R

is the frozen linear operator for dT/dt (given this restart's velocities, K_33, K_gm,
kappaH and the ice mask, all held fixed, exactly as in every other module here), and b is
the part of the surface forcing that does not depend on T (surface_forcing_vector).
A_T and b do not depend on a time step; dt never appears in this file.

The matrices are singular on land (every entry of a land row and column is zero), so A_T
itself cannot be inverted; the system is restricted to ocean cells and solved there, and
the land cells of the saved field are left as NaN (there is no equation to solve for them,
and no physically meaningful temperature to report).

Note that solving A_T . T = -b gives the steady state of A_T and b themselves, which are
frozen at whatever this restart's velocities, diffusivities and ice mask happen to be --
not the steady state of the full, evolving model. For a well-spun-up restart the two are
close (see spinup_200_72000.restart.h5); for a restart taken from early in a spin-up, the
solution is just "where this instant's frozen operator would settle", not a meaningful
physical equilibrium.

Output
------
NetCDF (via h5netcdf), with the solved 3D temperature field "temp" (degC) on the
(xt, yt, zt) grid used throughout this directory, plus the ocean mask "maskT" and the
residual "residual" (A_T T + b, which should be ~0 in ocean cells).

Command line
------------
    python solve_steady_state_temperature.py [restart_file [output_file]]

defaults to T_200 (data/veros_restart/reference_2000yr/reference_y0200.restart.h5) and
data/intermediate/steady_state_temperature.nc.
"""

import os
import sys

import h5netcdf
import numpy as np
from scipy.sparse.linalg import spsolve

from build_advection_matrix import NX, NY, NZ, SHAPE, XT, YT, get_mask_t, make_grid_z, uvw_from_restart
from build_combined_step import combined_matrices
from build_gm_matrix import eke_from_restart, nsqr_from_restart
from build_isoneutral_matrix import salinity_from_restart
from build_vertical_mixing_matrix import kappaH_from_restart, temperature_from_restart

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import INTERMEDIATE_DIR, T200_FILE  # noqa: E402

DEFAULT_RESTART_FILE = T200_FILE
DEFAULT_OUTPUT_FILE = os.path.join(INTERMEDIATE_DIR, "steady_state_temperature.nc")


def build_A_T_and_b(restart_file):
    """
    A_T (sparse CSR, 54000 x 54000) and b (vector, length 54000) for dT/dt = A_T T + b,
    built from the given restart's state. Neither depends on a time step.
    """
    T = temperature_from_restart(restart_file)
    S = salinity_from_restart(restart_file)
    uvw = uvw_from_restart(restart_file)
    kappaH = kappaH_from_restart(restart_file)
    Nsqr = nsqr_from_restart(restart_file)
    Ekin = eke_from_restart(restart_file)

    matrices = combined_matrices(T, S, uvw, kappaH, Nsqr, Ekin)
    A_T = matrices["M"] + matrices["E"] + matrices["D33"] + matrices["F"] + matrices["D"] + matrices["R"]
    b = matrices["b"]
    return A_T.tocsr(), b, T


def solve_steady_state(A_T, b, mask_t):
    """
    T solving A_T T = -b on ocean cells (A_T is singular over the full domain, since
    every land row and column is zero); land cells are left unsolved (returned as nan).
    """
    ocean = mask_t.ravel() > 0
    A_oc = A_T[ocean][:, ocean].tocsc()
    b_oc = -b[ocean]
    T_oc = spsolve(A_oc, b_oc)

    T_full = np.full(A_T.shape[0], np.nan)
    T_full[ocean] = T_oc
    return T_full


def save_netcdf(path, T, residual, mask_t, restart_file):
    zt, _ = make_grid_z()
    T3 = T.reshape(SHAPE)
    residual3 = residual.reshape(SHAPE)

    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"xt": NX, "yt": NY, "zt": NZ}

        v = f.create_variable("xt", ("xt",), dtype="f8", data=XT)
        v.attrs["units"] = "degrees_east"
        v.attrs["long_name"] = "longitude of T points"

        v = f.create_variable("yt", ("yt",), dtype="f8", data=YT)
        v.attrs["units"] = "degrees_north"
        v.attrs["long_name"] = "latitude of T points"

        v = f.create_variable("zt", ("zt",), dtype="f8", data=zt)
        v.attrs["units"] = "m"
        v.attrs["long_name"] = "depth of T points (0 = bottom, increasing upwards)"

        v = f.create_variable("temp", ("xt", "yt", "zt"), dtype="f8", data=T3)
        v.attrs["units"] = "degC"
        v.attrs["long_name"] = "steady-state temperature solving A_T . T = -b"

        v = f.create_variable("maskT", ("xt", "yt", "zt"), dtype="i1", data=mask_t.astype(np.int8))
        v.attrs["long_name"] = "ocean mask (1 = ocean, 0 = land)"

        v = f.create_variable("residual", ("xt", "yt", "zt"), dtype="f8", data=residual3)
        v.attrs["units"] = "degC/s"
        v.attrs["long_name"] = "A_T . temp + b (should be ~0 in ocean cells)"

        f.attrs["description"] = (
            "Solution of A_T . T = -b, A_T = M + E + D33 + F + D + R (advection, "
            "isoneutral explicit/implicit, GM skew, vertical mixing implicit/explicit), "
            f"built from {os.path.basename(restart_file)}."
        )


def main():
    restart_file = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESTART_FILE
    output_file = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_FILE

    print(f"Building A_T and b from {restart_file}")
    A_T, b, T_restart = build_A_T_and_b(restart_file)
    mask_t = get_mask_t()
    print(f"A_T: {A_T.shape[0]} x {A_T.shape[1]}, {A_T.nnz} non-zeros")

    print("Solving A_T . T = -b on ocean cells...")
    T_steady = solve_steady_state(A_T, b, mask_t)

    ocean = mask_t.ravel() > 0
    residual = np.full(A_T.shape[0], np.nan)
    residual[ocean] = (A_T @ np.where(ocean, T_steady, 0.0) + b)[ocean]
    print(f"max|residual| (ocean cells) = {np.nanmax(np.abs(residual)):.3e} degC/s")
    print(f"T_steady range (ocean): [{np.nanmin(T_steady):.3f}, {np.nanmax(T_steady):.3f}] degC")
    print(f"T_restart range (ocean): [{T_restart[ocean].min():.3f}, {T_restart[ocean].max():.3f}] degC")

    save_netcdf(output_file, T_steady, residual, mask_t, restart_file)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
