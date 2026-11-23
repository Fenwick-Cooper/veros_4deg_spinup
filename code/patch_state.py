"""
Putting a solution of the coupled temperature-velocity problem into a running Veros model
(taken from restart_veros_from_coupled_state.py of the original project; see
Description/coupled_spinup.tex, Section "Putting the solution into the model").

patch_state overwrites, in memory and before the next step:
    temp            the solution (all three time levels)
    u, v, w         the steady circulation the momentum operator gives for it
    psi             the streamfunction of that circulation; dpsi, dpsin = 0
    rho, Nsqr, Hd   recomputed from the new temperature by Veros's own routine
    du, dv, dtemp   the Adams-Bashforth "previous" tendencies are set equal to the
                    tendencies of the patched state itself, so that the first step
                    is not driven by tendencies left over from the old state
tke and eke (and so the mixing coefficients) are left as they are.
"""

import numpy as np

import build_momentum_matrix as bm
from veros import veros_routine


def with_halo(old_level, field, wet):
    """A copy of one time level of a Veros array (with halos) whose interior is replaced
    by `field` where `wet`, and whose east-west halos are refilled periodically."""
    out = np.array(old_level).copy()
    out[2:-2, 2:-2] = np.where(wet, field, out[2:-2, 2:-2])
    out[:2] = out[-4:-2]
    out[-2:] = out[2:4]
    return out


@veros_routine
def patch_state(state, model, T, uvw, info):
    """Overwrite the model state in memory (see the module docstring). T: (90, 40, 15);
    uvw: (u, v, w) each (90, 40, 15), or None to leave the circulation alone. Checks on
    the streamfunction are written into the dict `info`."""
    from veros.core import momentum, thermodynamics
    from veros.core.external import solve_stream
    from veros.core.operators import at, update
    from veros.core.operators import numpy as npx

    vs = state.variables
    tau, taum1 = int(vs.tau), int(vs.taum1)

    def set_all_levels(name, field, wet):
        arr = np.array(getattr(vs, name)).copy()
        level = with_halo(arr[..., tau], field, wet)
        for n in range(3):
            arr[..., n] = level
        setattr(vs, name, update(getattr(vs, name), at[...], npx.asarray(arr)))

    def copy_tau_to_taum1(name):
        arr = getattr(vs, name)
        setattr(vs, name, update(arr, at[..., taum1], arr[..., tau]))

    wet_t = np.array(vs.maskT)[2:-2, 2:-2] > 0
    set_all_levels("temp", T, wet_t)
    for n in range(3):
        vs.update(thermodynamics.calc_eq_of_state(state, n))

    if uvw is not None:
        u, v, w = uvw
        set_all_levels("u", u, np.array(vs.maskU)[2:-2, 2:-2] > 0)
        set_all_levels("v", v, np.array(vs.maskV)[2:-2, 2:-2] > 0)
        set_all_levels("w", w, np.array(vs.maskW)[2:-2, 2:-2] > 0)

        # streamfunction: only the corners next to a wet face matter (Veros uses psi
        # differences across wet faces only); psi is shifted so that it is zero where
        # Veros holds it at zero (the boundary of its reference island)
        psi, info["psi_residual"] = bm.streamfunction(u, v)
        wet_u = np.array(vs.maskU)[2:-2, 2:-2, -1] > 0
        wet_v = np.array(vs.maskV)[2:-2, 2:-2, -1] > 0
        touch = wet_u | wet_v | np.roll(wet_v, -1, axis=0)
        touch[:, :-1] |= wet_u[:, 1:]
        psi_old = np.array(vs.psi)[2:-2, 2:-2, tau]
        reference = touch & (psi_old == 0.0)
        info["psi_reference_corners"] = int(reference.sum())
        info["psi_reference_spread"] = float(np.ptp(psi[reference]))
        psi = psi - psi[reference].mean()
        psi[reference] = 0.0  # exactly, as Veros holds it, so that a later patch finds the same corners
        set_all_levels("psi", psi, touch)
        vs.dpsi = update(vs.dpsi, at[...], 0.0)
        vs.dpsin = update(vs.dpsin, at[...], 0.0)

        # momentum tendencies of the patched state (the first half of momentum.momentum),
        # stored as the previous Adams-Bashforth level
        model.set_forcing(state)
        vs.update(momentum.tend_coriolisf(state))
        vs.update(momentum.tend_tauxyf(state))
        vs.update(momentum.momentum_advection(state))
        state_update, _ = solve_stream.prepare_forcing(state)
        vs.update(state_update)
        copy_tau_to_taum1("du")
        copy_tau_to_taum1("dv")

    vs.update(thermodynamics.advect_temp_salt_enthalpy(state))
    for name in ("dtemp", "dHd"):
        copy_tau_to_taum1(name)
