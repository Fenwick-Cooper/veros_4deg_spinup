"""
Exact directional derivatives (Jacobian-vector products) of the Veros tau-day map, with
jax.jvp through Veros itself.

Import this module BEFORE common.py (or anything else that imports Veros): it switches Veros
to the two solvers JAX can differentiate through,

    VEROS_LINEAR_SOLVER=scipy_jax      streamfunction by jax.scipy.sparse.linalg.bicgstab
                                       (instead of SciPy on the host, which JAX cannot trace)
    VEROS_USE_SPECIAL_TDMA=0           vertical tridiagonal solves in plain JAX
                                       (instead of a Cython custom call without a derivative)

Both solve the same equations; the results differ from the production settings at the level
of the solver tolerances.

    tm = TangentMap(run, tau)              # run: common.Run, at the base state
    out = tm.evaluate(fields)              # Phi_tau(P(fields)), interior temperature after tau days
    out, dout = tm.jvp(fields, tangents)   # and its derivative in the direction `tangents`

fields / tangents: dicts with "temp" and, optionally, "u", "v", "w", "psi" (interior arrays,
(90, 40, 15), psi (90, 40)). P(fields) is the patch of patch_state.py written in JAX: the
fields are put into every time level (where wet), density, N^2 and Hd are recomputed, the
streamfunction tendencies are zeroed, and the Adams-Bashforth "previous" tendencies are set
to those of the patched state. Every evaluation starts from the base state stored when the
TangentMap was made, so the map is a function of `fields` alone.
"""

import os

os.environ["VEROS_LINEAR_SOLVER"] = "scipy_jax"
os.environ["VEROS_USE_SPECIAL_TDMA"] = "0"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Veros takes square roots of quantities clipped at zero, e.g. sqrt(max(0, N^2)) and sqrt(eke)
# (zero on land). At zero the derivative of sqrt is infinite and is multiplied by a zero
# tangent, which gives NaN. sqrt is therefore given the derivative 0 at 0 (a valid choice at a
# point where the function is not differentiable; the values themselves are unchanged). Done by
# replacing jax.numpy.sqrt before Veros's kernels are traced.
_sqrt = jnp.sqrt


@jax.custom_jvp
def _safe_sqrt(x):
    return _sqrt(x)


@_safe_sqrt.defjvp
def _safe_sqrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = _sqrt(x)
    positive = x > 0
    return y, jnp.where(positive, 0.5 / jnp.where(positive, y, 1.0), 0.0) * dx


def _sqrt_patched(x, *args, **kwargs):
    if args or kwargs or not isinstance(x, (jax.Array, jax.core.Tracer)) or not jnp.issubdtype(jnp.result_type(x), jnp.floating):
        return _sqrt(x, *args, **kwargs)
    return _safe_sqrt(x)


jnp.sqrt = _sqrt_patched


class TangentMap:
    def __init__(self, run, tau, freeze_closures=False):
        self.run, self.tau = run, int(tau)
        # freeze_closures: the derivative does not see the closures' response to the change of
        # N^2 in the patch (stop_gradient on N^2), as the emulator's Jacobian does not; the
        # values are unchanged. Exact for tau = 1; for tau > 1 later steps recompute N^2.
        self.freeze_closures = freeze_closures
        self.base = run.snapshot()
        vs = run.vs
        self.i_tau, self.i_taum1 = int(vs.tau), int(vs.taum1)
        self.wet = {
            "temp": np.array(vs.maskT)[2:-2, 2:-2] > 0,
            "u": np.array(vs.maskU)[2:-2, 2:-2] > 0,
            "v": np.array(vs.maskV)[2:-2, 2:-2] > 0,
            "w": np.array(vs.maskW)[2:-2, 2:-2] > 0,
        }
        wet_u = self.wet["u"][:, :, -1]
        wet_v = self.wet["v"][:, :, -1]
        touch = wet_u | wet_v | np.roll(wet_v, -1, axis=0)
        touch[:, :-1] |= wet_u[:, 1:]
        self.wet["psi"] = touch
        psi_old = np.array(vs.psi)[2:-2, 2:-2, self.i_tau]
        self.psi_reference = touch & (psi_old == 0.0)

    # the streamfunction is fixed up exactly as patch_state does (linear in psi)
    def fix_psi(self, psi):
        psi = psi - psi[self.psi_reference].mean()
        psi[self.psi_reference] = 0.0
        return psi

    def _apply(self, fields):
        """The patch P and tau days of Veros, as a function JAX can trace."""
        import jax.numpy as jnp
        from veros.core import momentum, thermodynamics
        from veros.core.external import solve_stream

        run = self.run
        state, model, vs = run.state, run.model, run.vs
        run.restore(self.base)
        tau, taum1 = self.i_tau, self.i_taum1

        def with_halo(name, field):
            base = getattr(vs, name)[..., tau]
            interior = jnp.where(self.wet[name], field, base[2:-2, 2:-2])
            level = base.at[2:-2, 2:-2].set(interior)
            level = level.at[:2].set(level[-4:-2])
            level = level.at[-2:].set(level[2:4])
            return jnp.stack([level, level, level], axis=-1)

        with vs.unlock():
            for name, field in fields.items():
                setattr(vs, name, with_halo(name, field))
            for n in range(3):
                vs.update(thermodynamics.calc_eq_of_state(state, n))
            if self.freeze_closures:
                vs.Nsqr = jax.lax.stop_gradient(vs.Nsqr)
            if "u" in fields:
                vs.dpsi = jnp.zeros_like(vs.dpsi)
                vs.dpsin = jnp.zeros_like(vs.dpsin)
                model.set_forcing(state)
                vs.update(momentum.tend_coriolisf(state))
                vs.update(momentum.tend_tauxyf(state))
                vs.update(momentum.momentum_advection(state))
                state_update, _ = solve_stream.prepare_forcing(state)
                vs.update(state_update)
                vs.du = vs.du.at[..., taum1].set(vs.du[..., tau])
                vs.dv = vs.dv.at[..., taum1].set(vs.dv[..., tau])
            vs.update(thermodynamics.advect_temp_salt_enthalpy(state))
            vs.dtemp = vs.dtemp.at[..., taum1].set(vs.dtemp[..., tau])
            vs.dHd = vs.dHd.at[..., taum1].set(vs.dHd[..., tau])
        for _ in range(self.tau):
            model.step(state)
        return vs.temp[2:-2, 2:-2, :, int(vs.tau)]

    def evaluate(self, fields):
        import jax.numpy as jnp

        out = np.array(self._apply({k: jnp.asarray(v) for k, v in fields.items()}))
        self.run.restore(self.base)
        self.run.days_total += self.tau
        return out

    def jvp(self, fields, tangents):
        import jax
        import jax.numpy as jnp

        keys = list(fields)
        primals = tuple(jnp.asarray(fields[k]) for k in keys)
        dirs = tuple(jnp.asarray(tangents.get(k, np.zeros_like(fields[k]))) for k in keys)
        out, dout = jax.jvp(lambda *a: self._apply(dict(zip(keys, a))), primals, dirs)
        out, dout = np.array(out), np.array(dout)
        self.run.restore(self.base)
        self.run.days_total += self.tau
        return out, dout


def velocity_fields(problem, tm, T):
    """The circulation the emulator gives for T, as patch fields (u, v, w, psi)."""
    import build_momentum_matrix as bm

    u, v, w = problem.velocity(T)
    psi, _ = bm.streamfunction(u, v)
    return {"u": u, "v": v, "w": w, "psi": tm.fix_psi(psi)}


def velocity_tangents(problem, tm, T, dT):
    """Its derivative in the direction dT (friction coefficients held fixed): the momentum
    balance is linear in the density, so d(u, v) = K^-1 PG drho/dT dT."""
    import build_momentum_matrix as bm
    from build_advection_matrix import SHAPE

    mom = problem.momentum
    drho = (bm.drho_dT(T, problem.S, problem.mask_t).ravel() * np.asarray(dT).ravel())
    du, dv, _ = mom.solve_forcing(mom.A["PG_u"] @ drho, mom.A["PG_v"] @ drho)
    dw = mom.W_u @ du + mom.W_v @ dv
    dpsi, _ = bm.streamfunction(du, dv)
    dpsi = dpsi - dpsi[tm.psi_reference].mean()
    dpsi[tm.psi_reference] = 0.0
    return {"u": du.reshape(SHAPE), "v": dv.reshape(SHAPE), "w": dw.reshape(SHAPE), "psi": dpsi}
