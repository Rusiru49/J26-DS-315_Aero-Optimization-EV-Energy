# EV Aero-Optimizer - Component 2 CFD dataset (complete package: all 30 shapes at both speeds)

**highway = 100 km/h (27.78 m/s)** and **urban = 40 km/h (11.11 m/s)**, 60 runs in total (30 shapes x 2 speeds). Column `scenario` tells which one a row is.

## What is in this folder
- run_table.csv : one row per run (60 rows), all columns, including Cd_label, Cd_mean_se, Cd_se_neff, scenario
- npz/sedan_<D>_100kmh.npz and npz/sedan_<D>_40kmh.npz : per-face data on the same 44,279 baseline faces (centre, area, normal, p, tau, Cd, Aref, U, rho, q_ref, p_ref)
- npz/baseline_faces.npz : face order reference; npz/*_morph.npz : pure morph fields for the low-confidence designs (where a mapped re-mesh field is used in the main file)
- case.json, this README

## Which Cd to use as the label
Use the column **Cd_label** (Cd_remesh when morph_confidence is low, otherwise Cd_mean = last-100 mean). The `Cd` stored inside an NPZ is the last iteration and can differ from the label by about 1 %.

## Standard error
Cd_mean_se = std of the last 100 iterations / sqrt(n_eff), n_eff = 100 / (1 + 2 * sum of autocorrelations) (Cd_se_neff). It covers iteration-to-iteration fluctuation only, not mesh, discretisation or turbulence-model error. Typical value 0.001-0.005 (0.4-1.4 %), so differences of about 1 % or less are within noise.

## Cd_label per design and speed
| Design | morph_confidence | Cd_label 100 km/h (highway) | Cd_label 40 km/h (urban) | difference | Cd_mean_se (100 / 40) |
|---|---|---|---|---|---|
| D000 | high | 0.3668 | 0.3709 | +1.1 % | 0.00356 / 0.00425 |
| D002 | high | 0.3947 | 0.3908 | -1.0 % | 0.00328 / 0.00282 |
| D003 | high | 0.3626 | 0.3536 | -2.5 % | 0.00300 / 0.00307 |
| D004 | high | 0.3442 | 0.3420 | -0.6 % | 0.00129 / 0.00124 |
| D005 | high | 0.3332 | 0.3377 | +1.4 % | 0.00300 / 0.00338 |
| D008 | high | 0.3784 | 0.3727 | -1.5 % | 0.00167 / 0.00094 |
| D012 | high | 0.3873 | 0.3916 | +1.1 % | 0.00333 / 0.00258 |
| D015 | high | 0.3456 | 0.3461 | +0.1 % | 0.00209 / 0.00263 |
| D019 | low | 0.3444 | 0.3511 | +1.9 % | 0.00204 / 0.00181 |
| D021 | high | 0.4173 | 0.4167 | -0.1 % | 0.00330 / 0.00314 |
| D023 | high | 0.3601 | 0.3618 | +0.5 % | 0.00324 / 0.00319 |
| D026 | high | 0.3384 | 0.3361 | -0.7 % | 0.00277 / 0.00237 |
| D031 | high | 0.3658 | 0.3564 | -2.6 % | 0.00276 / 0.00271 |
| D033 | high | 0.4168 | 0.4179 | +0.3 % | 0.00499 / 0.00507 |
| D034 | high | 0.3707 | 0.3673 | -0.9 % | 0.00170 / 0.00345 |
| D036 | low | 0.3734 | 0.3770 | +1.0 % | 0.00348 / 0.00353 |
| D037 | high | 0.3637 | 0.3777 | +3.8 % | 0.00358 / 0.00318 |
| D038 | high | 0.3775 | 0.3750 | -0.7 % | 0.00235 / 0.00219 |
| D039 | high | 0.3879 | 0.3917 | +1.0 % | 0.00447 / 0.00463 |
| D042 | high | 0.3823 | 0.3785 | -1.0 % | 0.00146 / 0.00246 |
| D043 | high | 0.3457 | 0.3444 | -0.4 % | 0.00357 / 0.00284 |
| D045 | high | 0.3184 | 0.3234 | +1.6 % | 0.00185 / 0.00316 |
| D047 | high | 0.4120 | 0.4113 | -0.2 % | 0.00518 / 0.00502 |
| D049 | high | 0.3864 | 0.3864 | +0.0 % | 0.00279 / 0.00267 |
| D058 | high | 0.3266 | 0.3292 | +0.8 % | 0.00143 / 0.00135 |
| D059 | high | 0.3759 | 0.3788 | +0.8 % | 0.00336 / 0.00379 |
| D061 | low | 0.3197 | 0.3204 | +0.2 % | 0.00234 / 0.00235 |
| D064 | high | 0.3369 | 0.3390 | +0.6 % | 0.00374 / 0.00319 |
| D065 | high | 0.3153 | 0.3219 | +2.1 % | 0.00256 / 0.00416 |
| D068 | high | 0.4095 | 0.4157 | +1.5 % | 0.00251 / 0.00232 |

## Honest notes
- OpenFOAM (not ANSYS), coarse mesh (about 0.57 M cells), 30 of the 70 designs so far.
- Morph-versus-re-mesh validation was done at 100 km/h only; at 40 km/h the low-confidence designs (D019, D036) were run as full re-meshes and mapped.
- D037 (+3.8 %) and D003 (-2.5 %) change more between the two speeds than the rest; with a standard error of about 1 % they may be partly noise.

---
# Original package README (highway data)
# Component 2 -> Component 3: first delivery (sedan, 100 km/h)

Author: S.I.C. Senevirathne (IT23213708)

## Contents
- npz/sedan_D###_100kmh.npz : per-face data (centre, area, normal, p, tau), float32. Designs: see run_table.csv.
- npz/baseline_faces.npz    : face connectivity (points, polys, poly_sizes) for plots.
- run_table.csv             : one row per design and speed. Cd, Cd_mean, Cl, A_frontal, flags, residuals.
- design_table.csv          : the 70 designs, 5 corners flagged (is_corner), run_order.
- parameter_ranges.csv      : baseline (D000) and design bounds of the 5 parameters (baseline is the midpoint of each).
- case.json                 : air and solver settings.
- qc_summary.csv            : quality notes per run.
- export_case.py, qc_report.py, make_c3_package.py : export and check scripts.

## Conventions
- Air flows toward -Z. Face normals point outward from the car into the fluid.
- p: gauge pressure in Pa (not divided by density), p_ref = 0 at the outlet. Air density 1.165 kg/m3.
- tau: wall shear stress on the car surface (force of the air on the surface, per area), in Pa.
- centre in metres (CAD frame), area in m2, A_frontal in m2 (each design's own area, the one used for its Cd).
- Number of faces: 44279, same order in every design (one baseline mesh, morphed per design).
- Cd = final iteration. Cd_mean = mean of the last 100 iterations (same window as the exported fields; use this one). Cd_mean500 = mean of the last 500.
- Cl: negative = downforce. Cl is more mesh-sensitive than Cd.
- Full car (no symmetry plane). k-omega SST, simpleFoam, OpenFOAM v2606 (not ANSYS Fluent).
- Convergence rule: residuals < 1e-3 and Cd drift (last-500 mean vs last-300 mean) < 1 %.
- Mesh: coarse (about 0.57 M cells). Mesh independence on D000: 0.57 / 1.02 / 1.86 M cells (not 2/5/8 M, because of an 8 GB laptop); Cd 0.3685 / 0.3787 / 0.3820; coarse is about 3.5 % below the fine mesh, an offset assumed the same for all designs.

## Parameter definitions
windshield angle = glass angle from horizontal; hood angle = hood slope from horizontal; spoiler angle = plate pitch from horizontal (0 = flush);
boat-tail angle = rear body taper from horizontal; ground clearance = vertical gap from the ground to the lowest underbody point (mm).
Values in the tables are the requested values; the morphed surface matches the target CAD within 5 mm (99th percentile).

## Corner designs (pilot)
D002, D019, D042, D047, D049: the 5 most extreme designs of the set (a Latin hypercube has no true corners). All morphed, converged. D047 shows a Cd oscillation (see notes).

## Real car
The sedan shell is a simplified model based on the BYD Seal (no wheels, no mirrors; not the manufacturer's CAD). Its Cd is not expected to match the published BYD Seal value; the baseline D000 is 0.367 on the coarse mesh. Only changes between designs are meaningful.

## Still pending
- 40 km/h runs (D000 and corners next) and the other designs (about 60).

## Validation summary (morph vs full re-mesh, same coarse settings)
remesh_diff_pct = 100 * (Cd_mean of the morph - Cd_remesh) / Cd_remesh, with both Cd values being the mean of the last 100 iterations.
Validated designs: D002 -2.3 %, D003 0.2 %, D012 -0.3 %, D019 -11.8 %, D036 -6.7 %, D038 1.5 %, D049 3.0 %.
The map_* columns describe the mapping of the re-mesh field onto the 44,279 baseline faces: relative Cd error, area-weighted RMS error of p and |tau| divided by q_ref (evaluated on the re-mesh faces), and the mean and p99 mapping distance.


## Which Cd to use as the label
- In run_table.csv, `Cd_mean` is the mean of the last 100 iterations of the run on the MORPHED mesh.
- Rows with `morph_confidence = low` (D019, D036): use `Cd_remesh` (full re-mesh, same settings, last-100 mean) as the label instead. The per-face file npz/sedan_<design>_100kmh.npz of these designs already carries this value (D019 0.3444, D036 0.3734) together with the field mapped from the re-mesh run. The field from the morphed mesh is kept as npz/sedan_<design>_100kmh_morph.npz.
- Rows with `morph_confidence = high`: use `Cd_mean`. Only the designs with a `Cd_remesh` value were checked against a full re-mesh (D002, D003, D012, D038, D049, plus D019 and D036). The other `high` designs passed the mesh-quality check only and were not re-meshed.
- D038: the force file ends at iteration 1348 although the solver reached 1500 (see `notes`); the label is the mean of its last 100 recorded rows.

## Cd_label and standard error (columns added in every run table)
- **Cd_label** = the value to train on. Rule: Cd_label = Cd_remesh when morph_confidence is low, otherwise Cd_mean (last-100 mean).
  Validated high-confidence designs (D002, D003, D012, D038, D049) keep Cd_mean because their NPZ holds the morph field.
- **Cd_mean_se** = standard error of Cd_mean: std of the last 100 iterations divided by sqrt(n_eff), where n_eff = 100 / (1 + 2 * sum of the autocorrelations) (**Cd_se_neff**). It measures only the iteration-to-iteration fluctuation of the converged solution; it does not include mesh, discretisation or turbulence-model error. It can be larger for the oscillating designs (D039, D047).
- **scenario** = highway (100 km/h) or urban (40 km/h).

