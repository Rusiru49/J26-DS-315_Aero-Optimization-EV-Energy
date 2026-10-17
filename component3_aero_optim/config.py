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
COMPONENT_ROOT = os.path.dirname(os.path.abspath(__file__))
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

MODELS_DIR = os.path.join(COMPONENT_ROOT, "data", "models")
OUTPUTS_DIR = os.path.join(COMPONENT_ROOT, "outputs")

RANDOM_STATE = 42

# ------------------------------------------------------------------ #
# VEHICLE CONSTANTS — placeholder, representative Sri Lanka-market
# compact/crossover EV. Replace with the team's confirmed vehicle spec
# once Member 1's design-space survey is finalised.
# ------------------------------------------------------------------ #
VEHICLE_MASS_KG = 1650.0
BATTERY_CAPACITY_KWH = 50.0
BASELINE_CD = 0.30
FRONTAL_AREA_M2 = 2.4
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
# SHAPE PARAMETERS — the 5 locked design variables and their styling
# limits (realistic min/max a stylist could vary within). Placeholder
# ranges informed by the literature in "Component 3 Research Papers/";
# TODO: replace with Member 1's confirmed design-space bounds (regulatory
# limits + comparable-EV survey + the CFD sampling box) once available.
# ------------------------------------------------------------------ #
SHAPE_PARAM_NAMES = [
    "windshield_angle_deg",
    "spoiler_angle_deg",
    "boat_tail_angle_deg",
    "hood_angle_deg",
    "ground_clearance_mm",
]

# Member 2's ranges (data contract, 2026-10-10): spoiler 0-25 deg, boat-tail
# 0-20 deg, ground clearance 130-190 mm. Windshield and hood are NOT final:
# Member 2 proposes "baseline +/- 5 deg" with baselines still TBD, so those
# two are placeholders around an assumed baseline.
SHAPE_PARAM_BOUNDS = {
    "windshield_angle_deg": (25.0, 35.0),   # TBD: placeholder, baseline 30 +/- 5
    "spoiler_angle_deg": (0.0, 25.0),
    "boat_tail_angle_deg": (0.0, 20.0),
    "hood_angle_deg": (1.0, 11.0),          # TBD: placeholder, baseline 6 +/- 5
    "ground_clearance_mm": (130.0, 190.0),
}

# Baseline (undeformed) sedan shape = design D000. TBD from Member 2; the
# values below are placeholders inside the ranges above.
BASELINE_SHAPE = {
    "windshield_angle_deg": 30.0,
    "spoiler_angle_deg": 5.0,
    "boat_tail_angle_deg": 8.0,
    "hood_angle_deg": 6.0,
    "ground_clearance_mm": 160.0,
}

# ------------------------------------------------------------------ #
# DATA CONTRACT WITH MEMBER 2 (see D:\Research\Component3-Data-Contract.md)
# ------------------------------------------------------------------ #
CATEGORY = "sedan"
SPEEDS_MS = {100: 27.78, 40: 11.11}   # key = km/h, value = m/s
DEFAULT_SPEED_KMH = 100
# CAD axes: nose toward +Z, so the air flows toward -Z. To be confirmed from
# Member 2's check script.
FLOW_DIRECTION = (0.0, 0.0, -1.0)
LIFT_DIRECTION = (0.0, 1.0, 0.0)      # Y is up
DYNAMIC_VISCOSITY_PA_S = 1.86e-5
N_DESIGNS = 70                        # LHS designs per category (baseline D000 is extra)
CD_SUM_CHECK_TOLERANCE = 0.01         # Member 2's integrity-check tolerance [to confirm]

# Placeholder data uses the same folder layout Member 2 will deliver. Set
# REAL_DATA_DIR to the shared folder once it exists (location TBD).
PLACEHOLDER_DATA_DIR = os.path.join(COMPONENT_ROOT, "data", "placeholder_sedan")
REAL_DATA_DIR = None

# ------------------------------------------------------------------ #
# POD / AUGMENTATION / SURROGATE
# ------------------------------------------------------------------ #
HOLDOUT_FRACTION = 0.18             # 15-20% of designs held out; split is by design
N_POD_MODES = 15
N_SYNTHETIC_SHAPES = 1200           # generator/LHS-proposed shapes to label

# ------------------------------------------------------------------ #
# OPTIMISER
# ------------------------------------------------------------------ #
OPTUNA_N_TRIALS = 100                # within the ~50-150 evaluation budget
BASELINE_OPTIMAL_TOLERANCE = 0.02    # 2% — within this, report "already near-optimal"

# Label check gate: the physics labels may only be used if the mean error on the
# hold-out designs is below this (Cd units; 0.001 = 1 drag count). Placeholder
# value, to be re-set once real data shows the noise level.
LABEL_CHECK_MEAN_TOLERANCE = 0.006
# How synthetic shapes get their drag label (pod_extraction.py): "exact" = rebuild
# the pressure field and integrate it over the predicted face geometry (the
# proposal's design); "ridge" = statistical map; "auto" = chosen per dataset by
# cross-validation on the training designs (see HELPER_KIND).
LABEL_METHOD = "auto"
# The helper model (shape -> pattern amounts): "per_mode" = one GP per pattern,
# "shared" = one multi-output GP. "auto" = chosen together with LABEL_METHOD by
# cross-validation on the training designs (helper_gp_model.select_labeller), so
# the data, not the developer, decides.
HELPER_KIND = "auto"
# Workflow 1 repeats the whole build on this many different hold-out draws and
# reports mean +/- spread (one split of ~12 test designs is too noisy to trust).
N_REPEAT_SPLITS = 10
