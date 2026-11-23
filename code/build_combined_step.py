#!/usr/bin/env python3
"""
Combine the matrices and forcing from every other module in this directory into a
single equation for one full temperature (and salinity) time step of the global
4-degree setup (spin-up-veros-config/global_4deg_no_cycle.py), and check it against
Veros's own "thermodynamics" step.

None of the matrix- or vector-building functions used here (advection_matrix,
isoneutral_explicit_matrix, isoneutral_implicit_matrix, gm_skew_matrix,
vertical_diffusion_matrix, surface_relaxation_matrix, surface_forcing_vector) take a
time step: each is a frozen, instantaneous operator or vector built only from the
current state (velocities, T, S, kappaH, N^2, EKE). dt_tracer is introduced for the
first time in temperature_step below, exactly where Veros itself introduces it: in
assembling the explicit Euler/Adams-Bashforth updates and the two implicit solves.

The full step (thermodynamics.thermodynamics, with this setup's settings: advection,
Redi diffusion and GM skew diffusion enabled, no horizontal/biharmonic diffusion, no
tracer sources) is, in order:

  1. Advection (Adams-Bashforth 2):
       T' = T^n + dt [ (1.5 + eps) M T^n - (0.5 + eps) g ]
     M is the advection matrix (build_advection_matrix.advection_matrix); g is the
     advection tendency from the *previous* step, M_{n-1} T^{n-1} -- a stored field,
     not reconstructible from this step's matrices, since the previous step's
     velocities are not part of this restart's state. eps = settings.AB_eps = 0.1.

  2. Isoneutral (Redi) diffusion, explicit part then implicit K_33 solve:
       T'' = (I - dt D_33)^{-1} ( T' + dt E T^n )
     E = isoneutral_explicit_matrix(T^n, S^n), D_33 = isoneutral_implicit_matrix(T^n, S^n).

  3. GM skew diffusion (purely explicit):
       T''' = T'' + dt F T^n
     F = gm_skew_matrix(T^n, S^n, N^2, EKE).

  4. Vertical mixing, surface flux (explicit) then implicit solve:
       T^{n+1} = (I - dt D)^{-1} ( T''' + dt (R T^n + b) )
     D = vertical_diffusion_matrix(kappaH), R = surface_relaxation_matrix(T^n),
     b = surface_forcing_vector(T^n).

Collecting these into one equation (the "one equation" of this module):

    T^{n+1} = (I - dt D)^{-1} [
                (I - dt D_33)^{-1} ( T^n + dt[(1.5+eps) M - E... ] T^n - dt(0.5+eps) g + dt E T^n )
                + dt F T^n + dt R T^n + dt b
              ]

or, written without re-expanding the two implicit solves (clearer, and how
temperature_step below actually computes it):

    T^{n+1} = (I - dt D)^{-1} [ (I - dt D_33)^{-1} [ T^n + dt((1.5+eps) M T^n - (0.5+eps) g + E T^n) ]
                                 + dt (F + R) T^n + dt b ]

Salinity follows exactly the same equation, with S in place of T and its own history
g_S = dsalt_{n-1}, but with R = 0 and b = 0: this setup has no surface salt flux (the
salinity equation's boundary condition is zero flux everywhere, not just at the sides
and bottom), so step 4 is a plain implicit solve with no explicit addition beforehand.
M, E, D_33, F and D are exactly the same matrices as for temperature -- none of them is
specific to the tracer they are applied to -- so they are built once and reused.

Main functions
--------------
    matrices = combined_matrices(T, S, uvw, kappaH, Nsqr, Ekin)   # dict, no dt anywhere
    T_next = temperature_step(matrices, T, g_T, dt)                # dt used only here
    S_next = salinity_step(matrices, S, g_S, dt)                   # dt used only here

Command line
------------
    python build_combined_step.py [restart_file]
"""

import os
import sys

import h5py
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

from build_advection_matrix import N, advection_matrix, get_mask_t, uvw_from_restart
from build_gm_matrix import eke_from_restart, gm_skew_matrix, nsqr_from_restart
from build_isoneutral_matrix import isoneutral_explicit_matrix, isoneutral_implicit_matrix, salinity_from_restart
from build_vertical_mixing_matrix import (
    kappaH_from_restart,
    surface_forcing_vector,
    surface_relaxation_matrix,
    temperature_from_restart,
    vertical_diffusion_matrix,
)

AB_EPS = 0.1  # settings.AB_eps
from paths import T200_FILE  # noqa: E402

DEFAULT_RESTART = T200_FILE


def combined_matrices(T, S, uvw, kappaH, Nsqr, Ekin, mask_t=None):
    """
    Every matrix and vector needed for one temperature time step, built from the
    current state only -- none of them depend on dt_tracer.

    Returns a dict with keys "M" (advection), "E" (isoneutral explicit), "D33"
    (isoneutral implicit), "F" (GM skew, explicit), "D" (vertical diffusion,
    implicit), "R" (surface relaxation, explicit) and "b" (surface forcing vector).
    """
    m = get_mask_t() if mask_t is None else mask_t
    return {
        "M": advection_matrix(uvw, mask_t=m),
        "E": isoneutral_explicit_matrix(T, S, mask_t=m),
        "D33": isoneutral_implicit_matrix(T, S, mask_t=m),
        "F": gm_skew_matrix(T, S, Nsqr, Ekin, mask_t=m),
        "D": vertical_diffusion_matrix(kappaH, mask_t=m),
        "R": surface_relaxation_matrix(T, mask_t=m),
        "b": surface_forcing_vector(T, mask_t=m),
    }


def temperature_step(matrices, T, g, dt):
    """
    One full temperature time step (advection + Redi + GM skew + vertical mixing),
    following Veros's own order exactly (thermodynamics.thermodynamics). dt (the time
    step) and AB_EPS appear only in this function -- nowhere in `matrices`.

    T : temperature vector at tau (length 54000), the state the matrices were built from.
    g : the previous step's advection tendency, M_{n-1} @ T^{n-1} (length 54000), read
        from the restart (build_combined_step.dtemp_taum1_from_restart); not computable
        from this step's own matrices.
    dt : the time step (s); settings.dt_tracer = 86400 for this setup.
    """
    M, E, D33, F, D, R, b = (matrices[k] for k in ("M", "E", "D33", "F", "D", "R", "b"))
    I = identity(len(T), format="csc")

    # 1. advection (Adams-Bashforth 2)
    T1 = T + dt * ((1.5 + AB_EPS) * (M @ T) - (0.5 + AB_EPS) * g)

    # 2. isoneutral (Redi) diffusion: explicit, then implicit K_33 solve
    T2 = spsolve((I - dt * D33).tocsc(), T1 + dt * (E @ T))

    # 3. GM skew diffusion: explicit only
    T3 = T2 + dt * (F @ T)

    # 4. vertical mixing: explicit surface flux, then implicit solve
    T4 = spsolve((I - dt * D).tocsc(), T3 + dt * (R @ T) + dt * b)

    return T4


def salinity_step(matrices, S, g, dt):
    """
    One full salinity time step (advection + Redi + GM skew + vertical mixing), the
    same equation as temperature_step but with no surface forcing (R = b = 0 for
    salinity in this setup: see the module docstring). dt appears only here.

    S : salinity vector at tau (length 54000), the state the matrices were built from.
    g : the previous step's advection tendency for salinity, length 54000, read from
        the restart (build_combined_step.dsalt_taum1_from_restart).
    dt : the time step (s); settings.dt_tracer = 86400 for this setup.
    """
    M, E, D33, F, D = (matrices[k] for k in ("M", "E", "D33", "F", "D"))
    I = identity(len(S), format="csc")

    S1 = S + dt * ((1.5 + AB_EPS) * (M @ S) - (0.5 + AB_EPS) * g)
    S2 = spsolve((I - dt * D33).tocsc(), S1 + dt * (E @ S))
    S3 = S2 + dt * (F @ S)
    S4 = spsolve((I - dt * D).tocsc(), S3)

    return S4


def dtemp_taum1_from_restart(restart_file):
    """The previous step's advection tendency for temperature (length 54000)."""
    with h5py.File(restart_file, "r") as f:
        taum1 = int(f["core/taum1"][()])
        return np.array(f["core/dtemp"][2:-2, 2:-2, :, taum1]).ravel()


def dsalt_taum1_from_restart(restart_file):
    """The previous step's advection tendency for salinity (length 54000)."""
    with h5py.File(restart_file, "r") as f:
        taum1 = int(f["core/taum1"][()])
        return np.array(f["core/dsalt"][2:-2, 2:-2, :, taum1]).ravel()


def main():
    restart_file = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESTART)

    T = temperature_from_restart(restart_file)
    S = salinity_from_restart(restart_file)
    uvw = uvw_from_restart(restart_file)
    kappaH = kappaH_from_restart(restart_file)
    Nsqr = nsqr_from_restart(restart_file)
    Ekin = eke_from_restart(restart_file)
    g_T = dtemp_taum1_from_restart(restart_file)
    g_S = dsalt_taum1_from_restart(restart_file)

    matrices = combined_matrices(T, S, uvw, kappaH, Nsqr, Ekin)
    T_next = temperature_step(matrices, T, g_T, dt=86400.0)
    S_next = salinity_step(matrices, S, g_S, dt=86400.0)
    print(f"Built all matrices and stepped temperature and salinity from {restart_file}")
    print(f"max|T_next - T| = {np.abs(T_next - T).max():.4f} degC")
    print(f"max|S_next - S| = {np.abs(S_next - S).max():.4f} g/kg")


if __name__ == "__main__":
    main()
