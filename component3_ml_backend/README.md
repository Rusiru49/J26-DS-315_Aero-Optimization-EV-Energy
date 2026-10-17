# Component 3: ML and Backend

**Physics-Grounded Drag Prediction and Cycle-Aware Shape Optimisation for Electric Vehicles**
J26-DS-315 · Owner: L.L. Chandanayaka (IT23320314)

Component 3 takes the CFD runs from Component 2 and the driving cycles from Component 1, and gives Component 4 (dashboard) the drag, energy, range and cost of a car shape, plus a better shape for a chosen driving cycle when one exists.

It has two workflows:

| | Workflow 1: build the drag model (offline) | Workflow 2: answer a request (live) |
|---|---|---|
| Input | Member 2's CFD runs (contract format) | 5 shape values, vehicle details, driving cycle |
| Steps | data check → split by design → POD → physics labeller → label gate → 1,200 physics-labelled synthetic shapes → XGBoost / Random Forest / ANN → stability over 10 hold-out draws | drag model → energy model → Bayesian optimiser (Optuna) → "near-optimal" or "better shape found" |
| Output | `models/drag_surrogate_*.joblib`, `models/frontal_area_model.joblib` | Wh/km, range, cost for the baseline and the best shape |

## Folder structure

```
component3_ml_backend/
├── src/component3/            the code (a Python package)
│   ├── config.py              all settings and constants
│   ├── data/                  loader.py (reads Member 2's format), checker.py (22 checks),
│   │                          dummy_generator.py (dummy data in the same format)
│   ├── workflow1/             pod.py (patterns + exact drag integration), helper.py (GP helper,
│   │                          labeller choice by cross-validation), augmentation.py, training.py
│   └── workflow2/             energy_model.py, optimizer.py
├── scripts/                   things you run: run_workflow1.py, check_dataset.py, make_dummy_data.py,
│                              run_workflow2_demo.py, plot_pressure_prediction.py, scale_test.py
├── experiments/               research tests, kept apart from the product code
│   ├── drivaerml/             tests on the public DrivAerML CFD data (CC BY-SA 4.0)
│   └── cfd2d_standin/         2D OpenFOAM stand-in in Member 2's format (Google Colab notebook)
├── tests/                     pytest (52 tests)
├── docs/research_papers/      papers used in the proposal
├── data/                      datasets (NOT in git)
├── models/                    trained models (NOT in git, rebuilt by run_workflow1.py)
└── outputs/                   small result summaries (JSON, figures)
```

## Setup

```
python -m venv ../component3_venv
../component3_venv/Scripts/pip install -r requirements.txt     (Windows; use bin/ on Linux/macOS)
```

Optional: `pip install -e .` makes `import component3` work from anywhere. Without it, the scripts and tests add `src/` to the path themselves.

## How to run

All commands from this folder, with the venv's Python:

| Task | Command |
|---|---|
| Make the dummy dataset | `python scripts/make_dummy_data.py` |
| Check a dataset folder | `python scripts/check_dataset.py <folder>` |
| Build the drag model | `python scripts/run_workflow1.py <placeholder \| real \| folder> <speed_kmh> [gate_counts]` |
| Energy model on the 5 driving cycles | `python scripts/run_workflow2_demo.py` |
| Predicted vs real pressure figure | `python scripts/plot_pressure_prediction.py <folder>` |
| Size test at Member 2's file size | `python scripts/scale_test.py` |
| Tests | `python -m pytest tests` |

Example on the 2D CFD stand-in: `python scripts/run_workflow1.py data/cfd2d_sedan 100 15`

## Data

- **Member 2's data** arrives in the format of the data contract (`run_table.csv`, `design_table.csv`, one NPZ per design and speed). Put it in a folder and set `config.REAL_DATA_DIR`, or pass the folder to the scripts.
- **Datasets and models are not in git.** They are too big and can be rebuilt. Shared datasets live in the team's Google Drive.
- Third-party data used in `experiments/drivaerml/`: DrivAerML (Ashton et al., 2024), CC BY-SA 4.0. See each `SOURCE.md`.

## Status (October 2026)

Workflow 1 and the energy model and optimiser are built and tested. On the 2D CFD stand-in, averaged over 10 hold-out draws, physics-labelled augmentation lowered the drag-model error by 50-65% (for example XGBoost 28.9 → 14.6 drag counts). These numbers come from 2D stand-in data, not Member 2's 3D runs. Still to do: the FastAPI service for Component 4.
