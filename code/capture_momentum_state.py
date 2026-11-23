#!/usr/bin/env python3
"""
Capture a settled circulation, and Veros's own value of every term of the u, v momentum
equations in it, for testing build_momentum_matrix.py.

Temperature is clamped (all three time levels, after every step) to the temperature of
a restart file while everything else runs free for n_days, so the circulation settles to
the steady state belonging to that temperature. Then the momentum part of one more step
is replayed term by term (the same call sequence as veros.core.momentum.momentum), and
each term is saved, together with the state, the friction coefficients and the grid, to
momentum_state_<case>.npz. Arrays are saved as Veros holds them (with halos).

    a    temperature and start state from warmup_0004yr_fresh.restart.h5
    b    temperature and start state from spinup_200_72000.restart.h5

Run with the Python environment in which Veros is installed:
    python capture_momentum_state.py a|b [n_days]
"""

import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401

CONFIG_DIR = VEROS_RUN_DIR  # Veros runs here and writes its (empty) NetCDF output here
RESTARTS = {
    "a": os.path.join(RESTART_DIR, "warmup_0004yr_fresh.restart.h5"),
    "b": T200_FILE,
}
DEFAULT_N_DAYS = {"a": 3000, "b": 1000}

sys.path.insert(0, HERE)

from equilibration_experiment_fixed import make_clamper, setup_model  # noqa: E402
from veros import veros_routine  # noqa: E402

GRID_NAMES = [
    "maskT", "maskU", "maskV", "maskW", "kbot", "dzt", "dzw", "zt", "zw", "dxt", "dxu", "dyt", "dyu",
    "cost", "cosu", "tantr", "coriolis_t", "hur", "hvr", "ht", "area_t", "area_u", "area_v",
]  # fmt: skip
STATE_3D = ["kappaM", "kappaH", "kappa_gm", "K_gm", "K_diss_v", "p_hydro"]
STATE_TAU = ["u", "v", "w", "temp", "salt", "rho", "Nsqr", "tke", "eke", "psi"]


@veros_routine
def probe(state, model, out):
    """Replay the momentum part of veros.veros.Veros.step() on the current state, saving
    each term. Leaves the state partly advanced, so nothing must be stepped afterwards."""
    from veros.core import eke, friction, momentum, tke
    from veros.core.external import solve_stream

    vs = state.variables
    tau = int(vs.tau)

    def snap(a):
        return np.array(a).copy()

    model.set_forcing(state)
    eke.set_eke_diffusivities(state)
    tke.set_tke_diffusivities(state)

    for name in STATE_TAU:
        out[name] = snap(getattr(vs, name)[..., tau])
    for name in STATE_3D[:-1]:
        out[name] = snap(getattr(vs, name))
    out["surface_taux"] = snap(vs.surface_taux)
    out["surface_tauy"] = snap(vs.surface_tauy)

    vs.update(momentum.tend_coriolisf(state))
    out["du_cor"], out["dv_cor"] = snap(vs.du_cor), snap(vs.dv_cor)
    du0, dv0 = snap(vs.du[..., tau]), snap(vs.dv[..., tau])

    vs.update(momentum.tend_tauxyf(state))
    du1, dv1 = snap(vs.du[..., tau]), snap(vs.dv[..., tau])
    out["du_wind"], out["dv_wind"] = du1 - du0, dv1 - dv0

    vs.update(momentum.momentum_advection(state))
    out["du_adv"], out["dv_adv"] = snap(vs.du_adv), snap(vs.dv_adv)
    du2, dv2 = snap(vs.du[..., tau]), snap(vs.dv[..., tau])

    friction.friction(state)
    out["du_mix"], out["dv_mix"] = snap(vs.du_mix), snap(vs.dv_mix)

    state_update, (forc, uloc, vloc) = solve_stream.prepare_forcing(state)
    vs.update(state_update)
    out["du_pg"], out["dv_pg"] = snap(vs.du[..., tau]) - du2, snap(vs.dv[..., tau]) - dv2
    out["p_hydro"] = snap(vs.p_hydro)
    out["psi_forc"], out["uloc"], out["vloc"] = snap(forc), snap(uloc), snap(vloc)


def main():
    case = sys.argv[1]
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_DAYS[case]
    restart = RESTARTS[case]

    with h5py.File(restart, "r") as f:
        tau = int(f["core/tau"][()])
        T_clamp = np.array(f["core/temp"][..., tau])
    frozen = {"temp": np.repeat(T_clamp[..., np.newaxis], 3, axis=-1)}

    identifier = f"momentum_state_{case}"
    t0 = time.time()
    model = setup_model(identifier, restart, n_days + 1, snap_out_freq=86400.0 * (n_days + 1) * 10)
    state = model.state
    clamp = make_clamper(frozen)
    clamp(state)

    for day in range(1, n_days + 1):
        u_before = np.array(state.variables.u[..., int(state.variables.tau)])
        model.step(state)
        clamp(state)
        if day % 250 == 0 or day == n_days:
            u_after = np.array(state.variables.u[..., int(state.variables.tau)])
            change = np.linalg.norm(u_after - u_before) / np.linalg.norm(u_after)
            print(f"[{case}] day {day:5d}  one-day relative change in u = {change:.2e}  wall={time.time() - t0:.0f}s", flush=True)

    out = {}
    probe(state, model, out)
    for name in GRID_NAMES:
        out[name] = np.array(getattr(state.variables, name))
    s = state.settings
    for name in ("A_h", "dt_mom", "rho_0", "grav", "AB_eps", "hor_friction_cosPower", "kappaM_min", "kappaM_max"):
        out[name] = float(getattr(s, name))
    np.savez(os.path.join(INTERMEDIATE_DIR, f"momentum_state_{case}.npz"), **out)

    for suffix in ("averages.nc", "snapshot.nc"):
        leftover = os.path.join(CONFIG_DIR, f"{identifier}.{suffix}")
        if os.path.exists(leftover):
            os.remove(leftover)
    print(f"[{case}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
