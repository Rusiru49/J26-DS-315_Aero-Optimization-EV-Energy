# Component 3: ML and Backend

**Physics-Grounded Drag Prediction and Cycle-Aware Shape Optimisation for Electric Vehicles**
J26-DS-315 · Owner: L.L. Chandanayaka (IT23320314)

Component 3 takes the CFD runs from Component 2 and the driving cycles from Component 1, and gives Component 4 (dashboard) the drag, energy, range and cost of a car shape, plus a better shape for a chosen driving cycle when one exists.

It has two workflows:

| | Workflow 1: build the drag model (offline) | Workflow 2: answer a request (live) |
|---|---|---|
| Input | Member 2's CFD runs (contract format) | 5 shape values, vehicle details, driving cycle |
| Steps | data check → POD → cross-validation by design (in each fold: physics labeller → 1,200 physics-labelled synthetic shapes → XGBoost / Random Forest / ANN, with and without augmentation, scored on real designs only) → pick the lowest-error model → retrain it on all designs | drag model → energy model → Bayesian optimiser (Optuna) → "near-optimal" or "better shape found" |
| Output | `models/<dataset>/`: `drag_model.joblib` (inputs: 5 shape values + speed), `frontal_area_model.joblib`, `model_card.json` (model, inputs, design box, cross-validated error per speed) | Wh/km, range, cost for the baseline and the best shape |

## Folder structure

```
component3_ml_backend/
├── src/component3/            the code (a Python package)
│   ├── config.py              all settings and constants
│   ├── data/                  loader.py (reads Member 2's format), checker.py (22 checks),
│   │                          dummy_generator.py (dummy data in the same format)
│   ├── workflow1/             pod.py (patterns + exact drag integration), helper.py (GP helper,
│   │                          labeller choice by cross-validation), augmentation.py, training.py,
│   │                          evaluation.py (cross-validation by design, model choice)
│   └── workflow2/             energy_model.py, optimizer.py
├── scripts/                   things you run: run_workflow1.py, check_dataset.py, make_dummy_data.py,
│                              run_workflow2_demo.py, plot_pressure_prediction.py, scale_test.py
├── experiments/               research tests, kept apart from the product code
│   ├── drivaerml/             tests on the public DrivAerML CFD data (CC BY-SA 4.0)
│   └── cfd2d_standin/         2D OpenFOAM stand-in in Member 2's format (Google Colab notebook)
├── tests/                     pytest (69 tests)
├── docs/research_papers/      papers used in the proposal
├── data/                      datasets used (in git): placeholder_sedan, cfd2d_sedan, scale_test_sedan
├── models/                    trained drag and frontal-area models, one folder per dataset and speed
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
| Build the drag model (all speeds, one model) | `python scripts/run_workflow1.py <placeholder \| real \| folder> [speed ...]` |
| Energy model on the 5 driving cycles | `python scripts/run_workflow2_demo.py` |
| Predicted vs real pressure figure | `python scripts/plot_pressure_prediction.py <folder>` |
| Size test at Member 2's file size | `python scripts/scale_test.py` |
| PP1 figures (from saved results) | `python scripts/make_pp1_figures.py` |
| Tests | `python -m pytest tests` |

Example on Member 2's data: `python scripts/run_workflow1.py real`

## Data

- **Member 2's data** arrives in the format of the data contract (`run_table.csv`, `design_table.csv`, `parameter_ranges.csv`, one NPZ per design and speed in `npz/`). Put it in a folder and set `config.REAL_DATA_DIR`, or pass the folder to the scripts. How the loader reads it:
  - **label:** `Cd_label` (Member 2's choice: the re-mesh value for low-confidence morphs, otherwise `Cd_mean`);
  - **test set:** low-confidence designs (`morph_confidence = low`) are used for training but never tested on;
  - **part deliveries:** designs listed in `design_table.csv` but not run yet are only a warning;
  - **design box:** a folder's own `parameter_ranges.csv` sets the box and baseline for the whole pipeline. The older test datasets carry the earlier guessed box this way.
- **Trained models** are saved per dataset, in `models/<dataset folder>/`: one drag model for all speeds.

## How Workflow 1 tests itself

With about 20 designs, a single hold-out of 4 designs is too noisy to trust. So Workflow 1 uses **cross-validation by design**:
- the testable designs (not the baseline, the extreme corners or the low-confidence morphs) are cut into 4 groups, and each group is the exam once while everything else trains;
- this is repeated 3 times with new groups (`config.N_CV_FOLDS`, `config.N_CV_REPEATS`);
- every candidate is scored on **real CFD designs only**, never on synthetic shapes.

The candidates are XGBoost, Random Forest and ANN, each trained with the 1,200 physics-labelled shapes and on the real designs only, plus two ANN variants that stop the few real designs being drowned by the synthetic ones: real rows weighted to half the say, and pre-train on all rows then fine-tune on the real ones. **The lowest cross-validated error wins**, so augmentation is used exactly when it helps. The winner is then retrained on all designs, and its cross-validated error is the uncertainty given with it. A diagnostic also scores the augmented models on synthetic shapes they never saw: if that error is small while the real-design error is large, the helper (labeller) is the weak link.
- **All datasets used in this component are in git** (about 1.6 GB in total), so every result can be reproduced from the repo:

  | Folder | What it is | Size |
  |---|---|---|
  | `data/member2_sedan_pp1/` | **Member 2's real CFD data, PP1 delivery: 30 sedan shapes x 2 speeds (60 runs)** | 108 MB |
  | `data/cfd2d_sedan/` | 2D OpenFOAM stand-in, Member 2's format (142 runs) | 1.8 MB |
  | `data/placeholder_sedan/` | dummy sedan, Member 2's format (invented physics) | 40 MB |
  | `data/scale_test_sedan/` | dummy at Member 2's file size (260,000 faces per file) | 1.2 GB |
  | `experiments/drivaerml/data/` | DrivAerML small files, 80 sampled surface files, mapped fields | ~285 MB |
  | `models/` | trained drag and frontal-area models | 1 MB |

  Tip: to clone without downloading all of it, use `git clone --depth 1 --filter=blob:none <url>`.
- **Member 2's real CFD data** (PP1 delivery) is in `data/member2_sedan_pp1/`; `config.REAL_DATA_DIR` points to it, so `run_workflow1.py real` uses it.
- Third-party data used in `experiments/drivaerml/`: DrivAerML (Ashton et al., 2024), CC BY-SA 4.0. See each `SOURCE.md`.

## Status (October 2026)

Workflow 1, the energy model and the optimiser are built and tested (60 tests).

- **2D CFD stand-in:** physics-labelled augmentation lowered the drag-model error by 50-65%.
- **Member 2's real data, PP1 delivery** (30 shapes x 2 speeds), cross-validated on the 22 testable designs (drag counts, 40 / 100 km/h):
  - a helper study (`experiments/helper_study/`) found where the labeller loses accuracy: patterns beyond the first two cannot be predicted from 30 shapes and pulled extreme shapes toward the average. Choosing the number of patterns by cross-validation (version H3, mostly 2 patterns) won at both speeds among 5 versions fixed in advance, and is now part of Workflow 1;
  - with it the labels are 9.8 / 8.8 off (14.0 / 13.0 before), and **augmentation beats training on the real designs alone for every model**: XGBoost 9.7 / 9.1 vs 18.2 / 17.2, Random Forest 10.3 / 9.6 vs 16.1 / 14.8, ANN 10.1 / 9.3 vs 10.1 / 10.8;
  - **one drag model for all speeds** (inputs: the 5 shape values + speed in km/h; `experiments/speed_study/` showed it is as accurate as one model per speed, and it scales to new CFD speeds). Saved in `models/member2_sedan_pp1/`;
  - the pipeline chooses **XGBoost + physics-labelled augmentation** (9.6 / 8.9 drag counts at 40 / 100 km/h, about 2.5%); augmentation beats real-only training for every model at both speeds;
  - **every synthetic label** carries its own uncertainty (calibrated on held-out real CFD: the real Cd falls inside +/-1 sigma 67-70% of the time, inside +/-2 sigma 85-91%) and must pass physics checks on its own rebuilt pressure field (Cp below the stagnation limit, positive pressure drag, Cd in the CFD range) and lie in the validated region; 98-99% pass. Table: `outputs/workflow1_<dataset>_<speed>kmh_synthetic_labels.csv`;
  - remaining limit: about 6 counts are lost because 30 shapes cannot describe every new shape's pressure; more shapes are needed for that part;
  - PP1 figures: `python scripts/make_pp1_figures.py` writes `outputs/figures/pp1_fig*.png`; run logs are in `outputs/logs/`.

Still to do: the FastAPI service for Component 4.
