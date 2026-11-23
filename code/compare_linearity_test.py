#!/usr/bin/env python3
"""
Compare the three runs of linearity_test_velocity.py. If the settled circulation is a
linear function of temperature, X(mid) = (X(a) + X(b)) / 2 for every field X.

For each field, reports
    signal       = rms(X_b - X_a)                      how much the field responds to T
    nonlinearity = rms(X_mid - (X_a + X_b)/2)          departure from linearity
    ratio        = nonlinearity / (signal / 2)         relative to the half-step response
(rms over cells where the field is non-zero in any run), overall and by model level.

Any Python with numpy:
    python compare_linearity_test.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import DESCRIPTION_DIR, INIT_FILE, INPUT_DIR, INTERMEDIATE_DIR, PLOTS_DIR, REFERENCE_DIR, RESTART_DIR, T200_FILE, VEROS_RUN_DIR  # noqa: E402, F401
FIELDS = ["u", "v", "w", "psi", "tke", "eke", "kappaH", "kappaM"]


def rms(x, m):
    return float(np.sqrt(np.mean(x[m] ** 2))) if m.any() else 0.0


def main():
    runs = {c: np.load(os.path.join(INTERMEDIATE_DIR, f"linearity_test_velocity_{c}.npz")) for c in ("a", "b", "mid")}
    print(f"{'field':8s} {'rms(a)':>10s} {'signal':>10s} {'nonlin':>10s} {'ratio':>7s}   tail-mean ratio")
    for name in FIELDS:
        a, b, mid = (runs[c][name] for c in ("a", "b", "mid"))
        wet = (a != 0) | (b != 0) | (mid != 0)
        signal = rms(b - a, wet)
        nonlin = rms(mid - 0.5 * (a + b), wet)
        line = f"{name:8s} {rms(a, wet):10.3e} {signal:10.3e} {nonlin:10.3e} {nonlin / (0.5 * signal):7.3f}"
        key = f"{name}_tailmean"
        if key in runs["a"]:
            at, bt, mt = (runs[c][key] for c in ("a", "b", "mid"))
            line += f"   {rms(mt - 0.5 * (at + bt), wet) / (0.5 * rms(bt - at, wet)):7.3f}"
        print(line)

    print("\nratio by level, surface (k=14) ... bottom (k=0):")
    for name in ("u", "v", "w", "kappaH"):
        a, b, mid = (runs[c][name] for c in ("a", "b", "mid"))
        wet = (a != 0) | (b != 0) | (mid != 0)
        vals = []
        for k in range(a.shape[2] - 1, -1, -1):
            m = wet[:, :, k]
            s = rms((b - a)[:, :, k], m)
            vals.append(rms((mid - 0.5 * (a + b))[:, :, k], m) / (0.5 * s) if s > 0 else np.nan)
        print(f"{name:8s} " + " ".join(f"{v:5.2f}" for v in vals))

    # how far is the 4-year circulation from the 200-year one? (cf. Cooper 2017, sec. 2.3)
    print("\nsettled circulation under T_4yr versus under T_200yr: rms difference / rms of the latter")
    for name in ("u", "v", "w", "psi"):
        a, b = runs["a"][name], runs["b"][name]
        wet = (a != 0) | (b != 0)
        print(f"{name:8s} {rms(b - a, wet) / rms(b, wet):6.3f}")


if __name__ == "__main__":
    main()
