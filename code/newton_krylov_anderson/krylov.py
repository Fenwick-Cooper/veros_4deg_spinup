"""
Small, explicit iterative pieces shared by the solvers: restarted-free GMRES (right
preconditioned, at most k iterations) and Anderson acceleration (type II, Walker & Ni 2011).
"""

import numpy as np


def gmres(apply_A, b, apply_Minv=None, k_max=20, rtol=0.1, log=None):
    """
    Solve A x = b approximately with at most k_max GMRES iterations, right preconditioned:
    the Krylov space is built for A M^-1 and x = M^-1 y. Starting guess x = 0.
    With k_max = 0, returns x = M^-1 b (one preconditioned Richardson step).
    Returns (x, relative residual estimate, iterations).
    """
    Minv = apply_Minv if apply_Minv is not None else (lambda r: r)
    if k_max == 0:
        return Minv(b), np.nan, 0
    beta = np.linalg.norm(b)
    V = [b / beta]
    H = np.zeros((k_max + 1, k_max))
    rel, k = 1.0, 0
    for j in range(k_max):
        w = apply_A(Minv(V[j]))
        for i in range(j + 1):  # modified Gram-Schmidt, twice for safety
            H[i, j] = V[i] @ w
            w = w - H[i, j] * V[i]
        for i in range(j + 1):
            c = V[i] @ w
            H[i, j] += c
            w = w - c * V[i]
        H[j + 1, j] = np.linalg.norm(w)
        k = j + 1
        e1 = np.zeros(k + 1)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(H[: k + 1, :k], e1, rcond=None)
        rel = np.linalg.norm(H[: k + 1, :k] @ y - e1) / beta
        if log:
            log(f"      GMRES {k:2d}: relative residual {rel:.3e}")
        if rel < rtol or H[j + 1, j] < 1e-14 * beta:
            break
        V.append(w / H[j + 1, j])
    y_full = sum(y[i] * V[i] for i in range(k))
    return Minv(y_full), rel, k


class Anderson:
    """
    Anderson acceleration of the fixed-point iteration x <- f(x), with residual g = f(x) - x:

        x_{k+1} = x_k + beta g_k - (dX_k + beta dG_k) gamma_k,
        gamma_k = argmin || g_k - dG_k gamma ||,

    dX, dG holding the last m differences of x and g. With m = 0 it is the plain (or, with
    beta < 1, damped) iteration. The memory is cleared if the residual grows by more than
    `restart_growth` over the smallest seen since the last clearing.
    """

    def __init__(self, m=5, beta=1.0, restart_growth=10.0):
        self.m, self.beta, self.restart_growth = m, beta, restart_growth
        self.x_prev = self.g_prev = None
        self.dX, self.dG = [], []
        self.best = np.inf

    def clear(self):
        self.x_prev = self.g_prev = None
        self.dX, self.dG = [], []
        self.best = np.inf

    def step(self, x, fx):
        g = fx - x
        norm = np.linalg.norm(g)
        restarted = False
        if norm > self.restart_growth * self.best:
            self.clear()
            restarted = True
        self.best = min(self.best, norm)
        if self.x_prev is not None and self.m > 0:
            self.dX.append(x - self.x_prev)
            self.dG.append(g - self.g_prev)
            if len(self.dX) > self.m:
                self.dX.pop(0)
                self.dG.pop(0)
        self.x_prev, self.g_prev = x.copy(), g.copy()
        if not self.dG:
            return x + self.beta * g, {"m_used": 0, "restarted": restarted, "residual": norm}
        dX, dG = np.array(self.dX).T, np.array(self.dG).T
        gamma, *_ = np.linalg.lstsq(dG, g, rcond=None)
        x_new = x + self.beta * g - (dX + self.beta * dG) @ gamma
        return x_new, {"m_used": len(self.dX), "restarted": restarted, "residual": norm, "gamma_max": float(np.abs(gamma).max())}
