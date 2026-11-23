# Newton–Krylov and Anderson acceleration compared with leap-and-settle

The scripts in this directory produce the results of `Description/method_comparison.pdf`.
They compare our spin-up method, **leap-and-settle**, with Anderson acceleration and
Newton–Krylov, in model days and in wall-clock time, to the competition criterion and to the
spun-up state.

Leap-and-settle works in rounds:

- **leap:** the coupled emulator (temperature, with the circulation slaved to it) is marched
  through tens to thousands of pseudo-years by linearly implicit pseudo-time steps, and the
  result is patched into Veros;
- **settle:** Veros runs for a few days, which brings the circulation, the turbulent energies
  and the fast modes back into line with the model before the next leap.

It is described in detail in `Description/coupled_spinup.pdf`. The competition and
equilibrium spin-ups of this repository (`competition_spin-up.py`, `equilibrium_spin-up.py`)
are its configurations L3 and L2.

## The runs

| Tag | Method | Uses the emulator? |
|---|---|---|
| `L3`, `L2` | Leap-and-settle, competition and equilibrium configurations | yes |
| `D_plain`, `D_aa3` | L3 with 3 extra long rounds: plain, or with Anderson acceleration over the rounds | yes |
| `A_m10` | Anderson acceleration of Veros alone (one-year map, memory 10) | no |
| `B_k20` | Newton–Krylov on Veros alone (one-year map, exact JAX derivatives, no preconditioner) | no |
| `C_k0`, `C_k5f` | Leap-and-settle, but the leap marches **Veros's own residual**, preconditioned by the emulator: defect correction (k = 0), or Newton–Krylov with 5 GMRES iterations using exact JAX derivatives. These are the "variants not shown" of the document. | yes, as a preconditioner |

For every run the harness records:

- the model days to the competition criterion (score < 10⁻⁵ held for 30 days), counted two ways:
  - days on the one continuous model trajectory;
  - all Veros days integrated, including the runs that are thrown away;
- the wall-clock time;
- the RMS distance to T₂₀₀₀ (year 2000 of the reference run), and the cost of reaching 0.06 °C and 0.05 °C.

## Running

All scripts run from the repository root with the Python environment in which Veros is
installed (JAX backend), except `plot_compare.py`, which needs only matplotlib. As for the
rest of `code/`, the location of the Veros source is read from `veros_path.txt` in the
repository root.

1. **The 2000-year reference run**, if it is not there already (about 6 hours):

   ```bash
   python code/run_2000yr_reference.py
   python code/analyse_2000yr_reference.py
   ```

2. **Every run, the timings and the table** (about 5 hours; nothing else should run on the
   machine meanwhile, so that wall-clock times are comparable). Set `VEROS_PYTHON` if the
   Veros environment's Python is not the `python` on the path:

   ```bash
   code/newton_krylov_anderson/run_all.sh
   ```

   It runs, one after another: the eight method runs (about 3.7 hours, Anderson and
   Newton–Krylov on Veros alone stop after one hour each), the speed of plain Veros, two
   full-length timings of a 3714-day conventional run (output off and on), the conventional
   curve (200 years of Veros, about 37 minutes), the start-up times, and the table
   `comparison.md`.

3. **The figure of the document**, written into `Description/`:

   ```bash
   python code/newton_krylov_anderson/plot_compare.py --exclude C_k0 C_k5f --out comparison_without_C.png --description
   ```

   Without `--exclude`, `--out` and `--description` it draws every run into `data/plots/comparison.png`.

## Where things go

| Path | Contents |
|---|---|
| `data/input/` | The competition's files. Read only. |
| `data/intermediate/newton_krylov_anderson/` | Written here: each run's log and daily CSV, `summary.jsonl` (one line per run), `run_all.log`, `veros_speed.json`, `conventional_timing.jsonl`, `startup_s.json`, `conventional_curve.npz`, `comparison.md`. Veros runs in `data/intermediate/`. |
| `data/veros_restart/newton_krylov_anderson/` | Restart files written on the way, and each run's final state (`<tag>_final.restart.h5`). |
| `data/intermediate/reference_2000yr/`, `data/veros_restart/reference_2000yr/` | The 2000-year reference run (read): `analysis.npz` and T₂₀₀₀ = `reference_y2000.restart.h5`. |
| `data/plots/` | Figures not used by the documents. |
| `Description/` | The documents, and the figure of `method_comparison.tex`. |

Every location comes from `code/paths.py`; `nka_paths.py` adds the two
`newton_krylov_anderson/` subdirectories.

## Files

| File | What it does | Needed for |
|---|---|---|
| `nka_paths.py` | File locations (from `code/paths.py`). | everything |
| `common.py` | The harness: runs Veros (production settings, NetCDF output off), counts days (main trajectory and all integrated), wall time and the criterion, saves and restores the model state, patches temperature (and circulation) in, writes the CSV and the summary line. | every run |
| `jax_tangent.py` | Exact Jacobian–vector products of the τ-day Veros map by `jax.jvp` through Veros, and the patch written in JAX. | B, C |
| `krylov.py` | GMRES (right-preconditioned) and Anderson acceleration. | A, B, C, D |
| `anderson_outer.py` | Leap-and-settle, optionally with Anderson acceleration over the rounds. | L3, L2, D |
| `anderson_veros.py` | Anderson acceleration on Veros alone. | A |
| `nk_veros.py` | Newton–Krylov on Veros alone. | B |
| `nk_emulator.py` | Leap-and-settle on Veros's own residual, with the emulator as preconditioner. | C (the "variants not shown") |
| `measure_speed.py` | Seconds per model day of plain Veros (0.030 s). | conventional wall times |
| `time_conventional.py` | Full wall-clock time of a 3714-day conventional run from process start (2.2 min; 2.3 min with output on). | conventional wall times |
| `conventional_curve.py` | The conventional Veros curve: a 200-year run sampled every 30 days, then the 2000-year run's decadal values. | conventional curve |
| `startup_times.py` | Each method run's start-up time (process time not on the harness clock). | all wall times |
| `compare.py` | The table, `comparison.md`. | table |
| `plot_compare.py` | The figure. | figure |
| `run_all.sh` | Runs all of the above in order. | — |
| `jvp_check.py` | Checks of `jax_tangent.py`: the JAX patch equals `patch_state` exactly; `jax.jvp` against finite differences (1–50 % apart, because convection switches); the cost of a jvp. | the statement that finite differences cannot be used |

## Notes

- **The JAX derivative needs differentiable solvers.** `jax.jvp` works through Veros once two
  settings are changed: `VEROS_LINEAR_SOLVER=scipy_jax` and `VEROS_USE_SPECIAL_TDMA=0`
  (`jax_tangent.py` sets both; import it before anything that imports Veros). The derivative of
  `sqrt` also has to be set to 0 at 0; without that, `sqrt(max(0, N²))` and `sqrt(eke)` give
  NaN. Runs that use the derivative (B, C) therefore run Veros at about 0.05 s/day instead of 0.03 s/day.
- **Finite-difference directional derivatives are unusable here.** Convection switches on and
  off where N² ≈ 0, so they disagree with the exact derivative by 1–50 % (`jvp_check.py`).
- **Method C's preconditioner is the emulator's τ-day backward-Euler map**,
  M⁻¹ = [(1+τ/dt) J_e − I/dt]⁻¹ (I − τ J_e). Using (J_e − I/dt) alone does not work: Veros
  mixes convecting columns implicitly, so the stiff modes of its residual saturate at −1/τ.
- **Wall-clock times are full process times**, from starting the program to the end, with
  Veros's NetCDF output off in every run: each method's harness time plus its start-up
  (13–14 s), and for conventional Veros the measured start-up (about 17 s) plus 0.030 s per
  model day.

## Results (Apple M3 Pro)

Criterion: main-trajectory day / all Veros days integrated / wall-clock time.

| Method | Criterion | Distance at criterion (°C) | To 0.06 °C: days integrated / wall | To 0.05 °C | Final distance (°C) |
|---|---|---|---|---|---|
| Conventional Veros | 3,714 / 3,714 / 2.2 min | 0.65 | 313,200 (870 y) / 2.7 h | 349,200 (970 y) / 3.0 h | — |
| Leap-and-settle, L3 | 63 / 63 / 10.0 min | 0.060 | 67 / 10.0 min | not reached | 0.059 |
| Leap-and-settle, L2 | 93 / 93 / 13.2 min | 0.048 | 81 / 13.2 min | 81 / 13.2 min | 0.048 |
| L3 + 3 long rounds | 78 / 78 / 17.0 min | 0.043 | 51 / 12.3 min | 51 / 12.3 min | 0.042 |
| Same, Anderson over rounds (m = 3) | 78 / 78 / 17.0 min | 0.043 | 51 / 12.3 min | 51 / 12.3 min | 0.043 |
| C: Veros residual, defect correction | 122 / 150 / 9.6 min | 0.067 | not reached | not reached | 0.066 |
| C: the same, NK with 5 GMRES its | 3,075 / 3,694 / 38.2 min | 0.594 | not reached | not reached | 0.593 |
| A: Anderson on Veros alone | 3,420 / 3,420 / 1.9 min | 0.667 | not reached (300 y, 53 min) | not reached | 0.253 |
| B: Newton–Krylov on Veros alone | 2,025 / 26,145 / 39.9 min | 0.566 | not reached (41,040 days, 63 min) | not reached | 0.443 |

- **Competition criterion.** Plain Veros meets it in 2.2 minutes, leap-and-settle (L3) in 10.
  Every run meets it after 6–10 model years whatever its distance from equilibrium, so it says
  little about spin-up.
- **Spun-up state.** Leap-and-settle is 13–16× faster in wall time and over 4000× cheaper in
  model days. Most of its wall time is the emulator solves.
- **Anderson acceleration over the rounds gains nothing.** The rounds reach their fixed point
  in about two rounds (0.061 → 0.048 → 0.043 → 0.043 °C); that fixed point is offset from T₂₀₀₀
  by the mismatch between the emulator and Veros, which Anderson acceleration cannot remove.
- **Anderson acceleration on Veros alone is worse than plain Veros** (0.253 against 0.168 °C
  after 300 years). Khatiwala (2024) applies it to passive biogeochemical tracers on a
  circulation already spun up; here temperature sets the density, so patching it alone puts the
  circulation out of balance.
- **Newton–Krylov on Veros alone is far worse** (0.443 °C after 41,040 Veros days; plain Veros
  reaches 0.246 with the same days and 0.153 in the same wall time).
- **Method C** with defect correction is no better than L3; with 5 GMRES iterations half of its
  steps are rejected and it ends 0.59 °C from T₂₀₀₀.
