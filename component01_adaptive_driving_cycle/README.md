# Adaptive Driving Cycle Pipeline

This folder contains the first project component for building an adaptive
EV driving cycle from real vehicle logs.

## Purpose

The workflow converts raw Car Scanner exports into structured, cleaned, and
synchronized trip data suitable for downstream cycle construction and energy
analysis.

## Workflow stages

### Stage 1: Raw data inventory

`01_Raw_Data_Inventory.ipynb`

- finds all raw CSV files in `data/raw/`
- extracts trip IDs and timestamps from file names
- checks for invalid geographic or sensor values
- creates a trip inventory and initial quality summary

### Stage 2: Data cleaning

`02_Data_Cleaning.ipynb`

- removes impossible or corrupted rows
- identifies GPS and OBD gaps or freeze events
- standardizes units and column naming
- saves cleaned intermediate trip files to `data/intermediate/stage2_cleaned/`
- writes cleaning summaries to `data/outputs/stage2_cleaning/`

### Stage 3: GPS/OBD synchronization

`03_GPS_OBD_Synchronization.ipynb`

- aligns GPS and OBD streams on a common time grid
- estimates clock offsets and interpolation windows
- creates a 1 Hz synchronized dataset for downstream analysis
- stores outputs under `data/intermediate/stage3_synced/`
- writes sync diagnostic plots and reports under `data/outputs/stage3_sync/`

## Folder structure

```text
component01_adaptive_driving_cycle/
├── 01_Raw_Data_Inventory.ipynb
├── 02_Data_Cleaning.ipynb
├── 03_GPS_OBD_Synchronization.ipynb
├── README.md
├── data/
│   ├── raw/
│   ├── intermediate/
│   └── outputs/
└── .gitignore
```

## Recommended execution order

Open and run the notebooks in order:

1. `01_Raw_Data_Inventory.ipynb`
2. `02_Data_Cleaning.ipynb`
3. `03_GPS_OBD_Synchronization.ipynb`

Each stage is designed to build on the outputs of the previous stage.
