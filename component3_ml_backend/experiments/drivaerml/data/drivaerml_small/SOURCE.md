# DrivAerML small files (third-party data)

- Source: DrivAerML, N. Ashton et al., "DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics", arXiv:2408.11969 (2024). Hosted at https://huggingface.co/datasets/neashton/drivaerml
- License: CC BY-SA 4.0 (credit the authors; anything shared that is derived from this data must keep the same license).
- Downloaded 2026-10-13: for runs 1-500, `force_mom_<i>.csv`, `force_mom_constref_<i>.csv`, `geo_parameters_<i>.csv`, `geo_ref_<i>.csv` from `https://huggingface.co/datasets/neashton/drivaerml/resolve/main/run_<i>/<file>`. 16 runs do not exist in the dataset (167, 211, 218, 221, 248, 282, 291, 295, 316, 325, 329, 364, 370, 376, 403, 473); run 56 has no `force_mom_constref`. 484 usable runs, about 2.4 MB.
- `drivaerml_scalar_table.csv` is built from these by `scripts/run_drivaerml_scalar.py` (Cd uses each design's own frontal area aRef).
