# Code

All scripts run with the Python environment in which Veros is installed (with its JAX
backend, if available); the plotting scripts also need matplotlib. Run them from the
repository root, e.g. `python code/run_2000yr_reference.py`. The location of the Veros
source code is read from `veros_path.txt` in the repository root.

Which script produces which result is listed in the "Reproducing the results" sections of
`Description/coupled_spinup.pdf` and `Description/equilibration_experiments.pdf`.

| Path | Contents |
|---|---|
| `data/input/` | The competition's files: `Init_state.restart.h5` and the forcing files. Read only. |
| `data/intermediate/` | Written by the scripts: daily scores (CSV), saved fields and matrices (NPZ), logs, the 2000-year run's snapshots and analysis. Veros also runs here. |
| `data/veros_restart/` | Restart files written by the scripts, among them the 2000-year run's restart every 100 years (`reference_2000yr/reference_y0200.restart.h5` is T<sub>200</sub>, `reference_y2000.restart.h5` is T<sub>2000</sub>). |
| `data/plots/` | Figures not used by the documents. |
| `Description/` | The documents; the scripts write the figures they use next to them. |

`paths.py` holds every file location. T<sub>200</sub> (`spinup_200_72000.restart.h5` in the
documents) is identical to year 200 of the 2000-year reference run, so the scripts that use
it need that run (`run_2000yr_reference.py`) to have reached year 200.
