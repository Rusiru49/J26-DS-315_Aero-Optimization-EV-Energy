# J26-DS-315 – Aerodynamic Shape Optimization and Energy Efficiency of Electric Vehicles in Local Conditions: A Sri Lanka Study

Final-year research project (2026–2027). We study how small changes to an electric car's body shape change its aerodynamic drag, and what that means for energy use, driving range and running cost **on Sri Lankan roads**, not only on standard international test cycles.

> **`main` holds released versions only.** All active work is on the **`dev`** branch. To see or run the code, switch to `dev`.

## The idea in one picture

```
 Component 1                Component 2                 Component 3                    Component 4
 Driving cycles     ──►     CAD-to-CFD runs     ──►     Drag model + energy     ──►    Dashboard
 (Sri Lankan roads)         (OpenFOAM)                  model + optimiser              (web app)
        │                                                     ▲
        └─────────────── driving cycles (time_s, speed_kmh) ──┘
```

1. **Component 1** builds representative Sri Lankan driving cycles (urban, suburban, rural, intercity, expressway) from real and synthetic GPS + OBD-II trip logs.
2. **Component 2** changes the car shape (5 shape values), meshes it and runs CFD in OpenFOAM to get the drag coefficient and the surface pressure of each design.
3. **Component 3** learns drag from the CFD runs (with physics-labelled data augmentation), turns drag into energy (Wh/km), range and cost over a driving cycle, and searches for a better shape.
4. **Component 4** shows all of this in an interactive dashboard: 3D car shapes, driving conditions, comparisons and statistics.

The 5 shape values used across the project: **windshield angle, hood angle, rear spoiler angle, boat-tail angle and ground clearance.**

## Team

| Component | Topic | Member | Branch |
|---|---|---|---|
| 1 | Sri Lankan driving cycles and design space | T.R.T. Fernando (IT23278080) | `IT23278080_Component01` |
| 2 | CAD-to-CFD automation (OpenFOAM) | S.I.C. Senevirathne (IT23213708) | `IT23213708_Component02` |
| 3 | ML and backend: physics-grounded drag prediction and cycle-aware shape optimisation | L.L. Chandanayaka (IT23320314) | `IT23320314_Component03` |
| 4 | Dashboard: EV aerodynamic impact analysis | R.B.T.S. Wijerathne (IT23200142) | `IT23200142_Component04` |

Supervisor: Dr. Kapila Dissanayake · Co-supervisor: Mr. Dinal Senadheera

## Repository layout (on `dev`)

```
J26-DS-315_Aero-Optimization-EV-Energy/
├── main.py, pipeline.py,          Component 1: driving-cycle pipeline (10 stages)
│   synthetic_data.py,
│   cycle_benchmark.py, SLDC.ipynb
├── raw_trips_csv/                 Component 1: logged trips
├── outputs/, figures/             Component 1: constructed cycles and plots
├── component2_cad_cfd/            Component 2: OpenFOAM case scripts and docs (on its own branch for now)
├── component3_ml_backend/         Component 3: drag model, energy model, optimiser (own README)
└── component4/                    Component 4: dashboard
    ├── backend/                   FastAPI backend
    ├── frontend/                  React + TypeScript + Vite + three.js
    └── mock_c3/                   stand-in for the Component 3 service (dummy data)
```

## Getting started

Clone the repo and switch to `dev`:

```bash
git clone https://github.com/Rusiru49/J26-DS-315_Aero-Optimization-EV-Energy.git
cd J26-DS-315_Aero-Optimization-EV-Energy
git switch dev
```

The repo includes the CFD datasets (about 1.6 GB). To download less, use `git clone --depth 1 --filter=blob:none <url>`.

| Component | How to run (from the repo root on `dev`) |
|---|---|
| 1 – Driving cycles | `python main.py` (writes cycles to `outputs/` and plots to `figures/`) |
| 2 – CAD-to-CFD | needs OpenFOAM; see `component2_cad_cfd/docs/` on the Component 2 branch |
| 3 – ML and backend | `cd component3_ml_backend`, `pip install -r requirements.txt`, then `python scripts/run_workflow1.py real`. Details in `component3_ml_backend/README.md` |
| 4 – Dashboard backend | `cd component4/backend`, `pip install -r requirements.txt`, then `uvicorn app.main:app --port 8000 --reload` |
| 4 – Dashboard frontend | `cd component4/frontend`, `npm install`, then `npm run dev` |

Python 3.11 or newer and Node.js 18 or newer are recommended.

## Branches and how to contribute

| Branch | Purpose |
|---|---|
| `main` | released versions only (e.g. PP2, final) |
| `dev` | integration branch: all component work is merged here |
| `IT2xxxxxxx_ComponentNN` | each member's own working branch |

1. Work and commit on **your own component branch**.
2. Open a pull request: **your branch → `dev`**.
3. `dev` → `main` only for agreed releases.
4. **Never merge `main` into `dev` or into your own branch.** `main` is kept small, so merging it would delete files on the other branch.

## Status (October 2026)

- **Progress Presentation 1 (PP1)** done on 23 October 2026.
- Component 1: driving-cycle pipeline working, real Sri Lankan trip logs being added.
- Component 2: first 30 sedan shapes × 2 speeds (40 and 100 km/h) of real CFD delivered.
- Component 3: drag model built and cross-validated on the real CFD data (about 9 drag counts error, XGBoost with physics-labelled augmentation); energy model and optimiser built; API for the dashboard is next.
- Component 4: dashboard backend and frontend running on dummy Component 3 data.
