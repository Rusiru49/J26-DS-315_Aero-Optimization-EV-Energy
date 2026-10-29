# Adaptive Driving Cycle Pipeline

This component turns recorded EV trips into representative, environment-specific
driving cycles and evaluates how well those cycles represent real driving. The
workflow is implemented as 11 Jupyter notebooks and is intended to be run in
order.

## Pipeline stages

| Stage | Notebook | Purpose | Main artifacts |
|---|---|---|---|
| 1. Raw data inventory | `01_Raw_Data_Inventory.ipynb` | Inventory raw Car Scanner CSVs, identify trips, and summarize available data and quality flags. | `outputs/stage1_inventory/` |
| 2. Data cleaning | `02_Data_Cleaning.ipynb` | Remove invalid or corrupted data, identify gaps and freeze events, and standardize columns and units. | `intermediate/stage2_cleaned/`, `outputs/stage2_cleaning/` |
| 3. GPS/OBD synchronization | `03_GPS_OBD_Synchronization.ipynb` | Align GPS and OBD streams, estimate timing offsets, and produce synchronized 1 Hz trip data with diagnostics. | `intermediate/stage3_synced/`, `outputs/stage3_sync/` |
| 4. Map matching | `04_Map_Matching.ipynb` | Match synchronized GPS traces to roads and produce route-matching diagnostics and road-class summaries. | `intermediate/stage4_map_matched/`, `outputs/stage4_map_matching/` |
| 5. Environment labelling | `05_Environment_Labelling.ipynb` | Assign an environment label to the mapped driving data for downstream analysis. | `intermediate/stage5_labelled/` |
| 6. Adaptive segmentation | `06_Adaptive_Segmentation.ipynb` | Divide trips into driving segments, evaluate candidate segmentation methods, and label segments for use in later stages. | `intermediate/stage6_segmented/`, `outputs/stage6_segmentation/` |
| 7. Feature extraction | `07_Feature_Extraction.ipynb` | Calculate driving-behavior features for usable segments. | `intermediate/stage7_features/`, `outputs/stage7_features/` |
| 8. K-means clustering | `08_KMeans_Clustering.ipynb` | Group similar segments within each environment using their extracted features. | `intermediate/stage8_clusters/`, `outputs/stage8_clustering/` |
| 9. Representative selection | `09_Representative_Selection.ipynb` | Select real recorded segments to represent the clusters and set aside trips for validation. | `intermediate/stage9_selected/`, `outputs/stage9_selection/` |
| 10. Driving-cycle construction | `10_Driving_Cycle_Construction.ipynb` | Join selected real segments into a 1 Hz speed-time cycle for each environment. | `outputs/stage10_driving_cycles/` |
| 11. Statistical validation | `11_Statistical_Validation.ipynb` | Compare the constructed cycles with held-out trips using driving statistics and validation diagnostics. | `outputs/stage11_validation/` |

Intermediate datasets and generated outputs are ignored by Git; rerun the
notebooks to create them.

## Data and folder layout

Place the raw trip CSV files in `data/raw/`. The notebooks write their
intermediate datasets and reports beneath `data/intermediate/` and
`data/outputs/`, respectively.

```text
component01_adaptive_driving_cycle/
├── 01_Raw_Data_Inventory.ipynb
├── 02_Data_Cleaning.ipynb
├── 03_GPS_OBD_Synchronization.ipynb
├── 04_Map_Matching.ipynb
├── 05_Environment_Labelling.ipynb
├── 06_Adaptive_Segmentation.ipynb
├── 07_Feature_Extraction.ipynb
├── 08_KMeans_Clustering.ipynb
├── 09_Representative_Selection.ipynb
├── 10_Driving_Cycle_Construction.ipynb
├── 11_Statistical_Validation.ipynb
├── data/
│   ├── raw/
│   ├── intermediate/
│   └── outputs/
├── docs/
│   └── Driving_Cycle_Pipeline_Documentation.pdf
└── README.md
```

## Running the pipeline

Run the notebooks in numerical order, from `01_Raw_Data_Inventory.ipynb`
through `11_Statistical_Validation.ipynb`. Each stage consumes artifacts
created by earlier stages.

The notebooks currently use the Google Colab Drive location
`/content/drive/MyDrive/Colab Notebooks/Component01` as their base directory.
In Colab, mount Drive and make the component's `data/` directory available at
that location. If you run the notebooks elsewhere, update the base/input paths
in their setup cells to point to this component's directories.

Stage 9 reserves trips for the independent validation in Stage 11. The final
cycles are built from selected real segments, not from cluster-center feature
averages.
