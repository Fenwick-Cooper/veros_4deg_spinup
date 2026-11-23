# Veros 4° ocean spin-up

Our aim here is to spin up the Veros ocean model using the 4° configuration at <https://past2future.org/2026/09/11/ocean-model-spin-up-competition/>. The longer term goal is to find a method to spin up higher resolution models.

We use the method put forward in [Cooper (2017)](#references) and adapted for this problem. Taking advantage of the fact that the velocity equilibrates at a faster rate than the temperature field, the basic method in Cooper (2017) is:

1. Run the ocean model for a short time to spin up to some extent and to bring fields into balance.
2. Linearise the temperature tendency about the current state and implicitly solve for the equilibrium.
3. Repeat until spin-up is achieved.

The changes here include:

- Unlike in Cooper (2017), here the temperature feeds back strongly onto the velocity. As suggested in that paper, this was handled by inclusion of a linear equation for velocity, instantaneously slaved to the temperature.
- The equilibrium solve used LU factorisation (affordable at 4°) instead of Bi-CGSTAB (used in Cooper 2017 for higher resolution), combined with implicit pseudo-time stepping (Kelley and Keyes, 1998) of a simplified non-linear analogue of Veros with fixed turbulence energy fields.
- At 4° resolution eddies are not resolved, enabling use of a linear equation for velocity. In an eddying flow, as in Cooper (2017), time mean equations need to be used.

## Results

| Method | Model day that score < 10⁻⁵ | Wall-clock time (Apple M3 Pro) | RMS distance to T<sub>2000</sub> |
|---|---|---|---|
| Veros 3714 day run | 3714 | 2.2 minutes | 0.65 °C |
| Veros 200 year run | - | 37 minutes | 0.21 °C |
| Veros 1000 year run | - | \~3 hours | 0.05 °C |
| Competition spin-up | 63 | 10 minutes | 0.06 °C |
| Equilibrium spin-up | 93 | 13 minutes | 0.05 °C |
| Anderson acceleration | 3420 | 1.9 minutes | 0.67 °C |
| Newton–Krylov | 2025 | 40 minutes | 0.57 °C |

**Competition spin-up:** Veros is run for 10 days followed by a round of implicit time stepping. This is repeated four times and then Veros is run until the score < 10⁻⁵. **Equilibrium spin-up:** The same but with Veros run for 20 days between four rounds of implicit time stepping. T<sub>2000</sub> is the model temperature at the end of a 2000 year integration.

The competition criterion is met in 63 model days and stays met for at least the next ten years. However, this neglects the cost of four rounds of implicit time stepping, which makes the method slower than Veros alone. The wall-clock time includes the implicit time stepping, but does not reflect that the optimal algorithm and relative run times change with resolution. The spun-up state can be improved by increasing the number of Veros days between rounds of implicit time stepping from 10 to 20. In 13 minutes Veros was brought as close to its equilibrium as a direct integration taking approximately 3 hours (14× faster).

## Code

All source code and extra documentation can be found at <https://github.com/Fenwick-Cooper/veros_4deg_spinup>. Clone the repo to use the code.

To run the "Competition spin-up" method run `python competition_spin-up.py`. The output is a Veros restart file `competition.restart.h5` containing the spun-up state.

To run the "Equilibrium spin-up" method run `python equilibrium_spin-up.py`. The output is a Veros restart file `equilibrium.restart.h5` with the state equivalent to an approximately 1000 year Veros integration.

## Note

Creating the exact sparse matrices specific to each term in an ocean model is labour intensive. However, AI assistants can now automate this step and it is relatively easy to check that a matrix is correct. Once the matrices have been extracted, it is much easier to play around with various time-stepping schemes. We first used Claude to reverse engineer and document the precise Veros integration scheme ([`veros_discretisation.pdf`](Description/veros_discretisation.pdf)). Then after preliminary tests established that the velocity must respond to the temperature during the solve ([`equilibration_experiments.pdf`](Description/equilibration_experiments.pdf)), we applied the technique to Veros (see [`coupled_spinup.pdf`](Description/coupled_spinup.pdf) for a detailed write-up). Finally we tested against basic Anderson acceleration and Newton–Krylov methods ([`method_comparison.pdf`](Description/method_comparison.pdf)).

| File | Description |
|---|---|
| [`Description/veros_discretisation.pdf`](Description/veros_discretisation.pdf) | Veros's precise numerical scheme. |
| [`Description/equilibration_experiments.pdf`](Description/equilibration_experiments.pdf) | Initial experiments. |
| [`Description/coupled_spinup.pdf`](Description/coupled_spinup.pdf) | Detailed documentation of the method and the results. |
| [`Description/method_comparison.pdf`](Description/method_comparison.pdf) | Test against Anderson and Newton–Krylov. |
| [`competition_spin-up.py`](competition_spin-up.py) | Spin up Veros in 63 days. |
| [`equilibrium_spin-up.py`](equilibrium_spin-up.py) | Spin up Veros close to equilibrium. |

The effort involved in all of these tasks was substantially reduced through the use of Claude; however, we directed the work, maintained oversight, understand what we are doing, have read and revised the documents many times and have checked the main results independently. This document was written by a human and checked for errors by Claude.

## References

Cooper, F. C., 2017: A potential method to accelerate spin up of turbulent ocean models. *Ocean Modelling*, **120**, 79–82. <https://doi.org/10.1016/j.ocemod.2017.10.008>
