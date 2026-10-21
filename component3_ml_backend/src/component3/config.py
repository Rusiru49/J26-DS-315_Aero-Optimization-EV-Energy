"""
Component 3 configuration — all tunable constants in one place.

Nothing in this file is deployment logic; it is only named numbers so that
every other module imports its constants from here instead of burying magic
numbers in function bodies. Anything marked TODO is a placeholder value that
must be confirmed with a teammate or a cited source before the final report
numbers are treated as real.
"""

import os

# ------------------------------------------------------------------ #
# PATHS
# ------------------------------------------------------------------ #
# config.py lives in <component>/src/component3/, so the component root is three levels up
COMPONENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(COMPONENT_ROOT)

# Member 1's real driving-cycle output lives at the repo root, not inside
# this component's folder.
DRIVING_CYCLE_DIR = os.path.join(REPO_ROOT, "outputs")
DRIVING_CYCLE_FILES = {
    "urban": os.path.join(DRIVING_CYCLE_DIR, "driving_cycle_urban.csv"),
    "suburban": os.path.join(DRIVING_CYCLE_DIR, "driving_cycle_suburban.csv"),
    "rural": os.path.join(DRIVING_CYCLE_DIR, "driving_cycle_rural.csv"),
    "intercity": os.path.join(DRIVING_CYCLE_DIR, "driving_cycle_intercity.csv"),
    "expressway": os.path.join(DRIVING_CYCLE_DIR, "driving_cycle_expressway.csv"),
}

MODELS_DIR = os.path.join(COMPONENT_ROOT, "models")
OUTPUTS_DIR = os.path.join(COMPONENT_ROOT, "outputs")

RANDOM_STATE = 42

# ------------------------------------------------------------------ #
# VEHICLE CONSTANTS — placeholder, representative Sri Lanka-market
# compact/crossover EV. Replace with the team's confirmed vehicle spec
# once Member 1's design-space survey is finalised.
# ------------------------------------------------------------------ #
VEHICLE_MASS_KG = 1650.0
BATTERY_CAPACITY_KWH = 50.0
# Baseline sedan D000 from Member 2's CFD (100 km/h, coarse mesh, delivery
# of 2026-10-20): Cd_label 0.3668, A_frontal 2.1812 m^2. Only changes between
# shapes are meaningful (the coarse mesh reads about 3.5% low).
BASELINE_CD = 0.3668
FRONTAL_AREA_M2 = 2.1812
ROLLING_RESISTANCE_COEFF = 0.011  # C_rr, typical passenger-EV tyre on asphalt

DRIVETRAIN_EFFICIENCY = 0.90
REGEN_EFFICIENCY = 0.65
AUX_LOAD_W = 650.0  # constant accessory load: AC, electronics, lights

GRAVITY_M_S2 = 9.81

# Tropical ambient air density (~31 degC, sea level) rather than the
# standard 1.225 kg/m^3 ISA value, since this project is calibrated to
# Sri Lankan conditions.
AIR_DENSITY_KG_M3 = 1.165

# TODO: confirm with team — cite the actual CEB (Ceylon Electricity Board)
# domestic/EV tariff band in effect for this study rather than guessing.
ELECTRICITY_TARIFF_LKR_PER_KWH = 45.0

# ------------------------------------------------------------------ #
# SHAPE PARAMETERS — the 5 locked design variables and their limits.
# Member 2's parameter_ranges.csv (delivery of 2026-10-20), which Member 2
# copied from Member 1's design space; the baseline is the midpoint of each
# range. TODO: Member 1 to confirm the ranges and their evidence basis
# (Component3-Request-to-Member1.md, B1-B3).
#
# A data folder can carry its own parameter_ranges.csv; the loader then
# calls set_design_space() so the whole pipeline uses that folder's box.
# The older test datasets (placeholder, 2D stand-in, scale test) were made
# with the earlier guessed box and carry it in their own parameter_ranges.csv.
# ------------------------------------------------------------------ #
SHAPE_PARAM_NAMES = [
    "windshield_angle_deg",
    "spoiler_angle_deg",
    "boat_tail_angle_deg",
    "hood_angle_deg",
    "ground_clearance_mm",
]

SHAPE_PARAM_BOUNDS = {
    "windshield_angle_deg": (22.0, 34.0),
    "spoiler_angle_deg": (0.0, 20.0),
    "boat_tail_angle_deg": (5.0, 25.0),
    "hood_angle_deg": (5.0, 15.0),
    "ground_clearance_mm": (110.0, 190.0),
}

# Baseline (undeformed) sedan shape = design D000.
BASELINE_SHAPE = {
    "windshield_angle_deg": 28.0,
    "spoiler_angle_deg": 10.0,
    "boat_tail_angle_deg": 15.0,
    "hood_angle_deg": 10.0,
    "ground_clearance_mm": 150.0,
}


def set_design_space(bounds, baseline):
    """Switch the design box and baseline used by every module. The dicts are
    updated in place, so modules that already hold a reference see the change."""
    SHAPE_PARAM_BOUNDS.update({n: (float(bounds[n][0]), float(bounds[n][1])) for n in SHAPE_PARAM_NAMES})
    BASELINE_SHAPE.update({n: float(baseline[n]) for n in SHAPE_PARAM_NAMES})


# ------------------------------------------------------------------ #
# DATA CONTRACT WITH MEMBER 2 (see D:\Research\Component3-Data-Contract.md)
# ------------------------------------------------------------------ #
CATEGORY = "sedan"
SPEEDS_MS = {100: 27.78, 40: 11.11}   # key = km/h, value = m/s
DEFAULT_SPEED_KMH = 100
# CAD axes: nose toward +Z, so the air flows toward -Z (confirmed in Member 2's
# case.json).
FLOW_DIRECTION = (0.0, 0.0, -1.0)
LIFT_DIRECTION = (0.0, 1.0, 0.0)      # Y is up
DYNAMIC_VISCOSITY_PA_S = 1.86e-5
N_DESIGNS = 70                        # LHS designs per category (baseline D000 is extra)
CD_SUM_CHECK_TOLERANCE = 0.01         # Member 2's integrity-check tolerance (1%)

# Placeholder data uses the same folder layout Member 2 delivers. Member 2's
# PP1 delivery (30 sedan shapes x 2 speeds, 2026-10-21) is in the repo; a later
# delivery goes in its own folder and REAL_DATA_DIR points to it.
PLACEHOLDER_DATA_DIR = os.path.join(COMPONENT_ROOT, "data", "placeholder_sedan")
REAL_DATA_DIR = os.path.join(COMPONENT_ROOT, "data", "member2_sedan_pp1")

# ------------------------------------------------------------------ #
# POD / AUGMENTATION / SURROGATE
# ------------------------------------------------------------------ #
HOLDOUT_FRACTION = 0.18             # single hold-out split (loader); Workflow 1 itself uses cross-validation
N_POD_MODES = 15
N_SYNTHETIC_SHAPES = 1200           # generator/LHS-proposed shapes to label

# ------------------------------------------------------------------ #
# OPTIMISER
# ------------------------------------------------------------------ #
OPTUNA_N_TRIALS = 100                # within the ~50-150 evaluation budget
BASELINE_OPTIMAL_TOLERANCE = 0.02    # 2% — within this, report "already near-optimal"

# Old fixed label-check limit (Cd units; 0.001 = 1 drag count). Workflow 1 no
# longer stops on it: the synthetic labels are used only if the augmented model
# beats the real-only model in cross-validation (workflow1/evaluation.py). Kept
# for the DrivAerML experiment script.
LABEL_CHECK_MEAN_TOLERANCE = 0.006
# How synthetic shapes get their drag label (workflow1/pod.py): "exact" = rebuild
# the pressure field and integrate it over the predicted face geometry (the
# proposal's design); "ridge" = statistical map; "auto" = chosen per dataset by
# cross-validation on the training designs (see HELPER_KIND).
LABEL_METHOD = "auto"
# The helper model (shape -> pattern amounts): "per_mode" = one GP per pattern,
# "shared" = one multi-output GP. "auto" = chosen together with LABEL_METHOD by
# cross-validation on the training designs (helper_gp_model.select_labeller), so
# the data, not the developer, decides.
HELPER_KIND = "auto"
# Workflow 1 tests itself by cross-validation by design (workflow1/evaluation.py):
# the testable designs are cut into N_CV_FOLDS groups, each group is the exam
# once, and the whole thing is repeated N_CV_REPEATS times with new groups.
N_CV_FOLDS = 4
N_CV_REPEATS = 3
N_JOBS = 2                             # parallel worker processes for the folds (more ran an 8 GB laptop out of memory)
