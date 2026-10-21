# EV Energy Optimization Workspace

This repository brings together the EV-driving-cycle data pipeline and the
energy-model optimization stack used for aerodynamic and route-aware EV
performance studies.

## Repository layout

- `component01_adaptive_driving_cycle/` — real-world driving-data processing for
  raw vehicle logs, including inventory checks, cleaning, and GPS/OBD
  synchronization.
- `component3_ml_backend/` — ML and optimization backend for surrogate models,
  pressure prediction, and route-energy analysis.
- `figures/` — aggregated output plots and benchmark summaries across the
  project workflow.

## Component 01: Adaptive driving cycle pipeline

The adaptive driving-cycle workflow is implemented as a set of notebooks that
operate in sequence on raw Car Scanner CSV exports stored under
`component01_adaptive_driving_cycle/data/raw/`.

### Notebooks

1. `01_Raw_Data_Inventory.ipynb` — enumerates all raw trips, parses filenames
   into trip IDs and timestamps, and flags invalid or suspicious logs.
2. `02_Data_Cleaning.ipynb` — cleans raw GPS and vehicle data, removes bad rows,
   fills logical gaps, and writes cleaned intermediate files.
3. `03_GPS_OBD_Synchronization.ipynb` — aligns GPS and OBD streams on a 1 Hz
   timeline, estimates clock offsets, and creates synchronized trip data.

### Data folders

- `component01_adaptive_driving_cycle/data/raw/` — original CSV exports
- `component01_adaptive_driving_cycle/data/intermediate/` — cleaned and synced
  trip outputs
- `component01_adaptive_driving_cycle/data/outputs/` — stage reports and plots

Typical workflow:

```bash
cd component01_adaptive_driving_cycle
# Open the notebooks in order and run each cell sequentially
```

## Component 3: ML backend

The ML backend contains the downstream modeling stack for pressure prediction,
feature generation, optimization, and validation. This part of the project is
organized under `component3_ml_backend/` and includes:

- data loading and validation helpers
- aerodynamic surrogate models
- training and optimization workflows
- tests for energy-model behavior and regression checks

## Notes

This repository previously contained a synthetic prototype description that no
longer reflects the working project structure. The current project is centered
on real driving logs and the associated EV energy optimization pipeline.
