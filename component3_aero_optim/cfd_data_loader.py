"""
The ONE place that reads CFD data. Everything downstream (POD, helper model,
augmentation, surrogate training) goes through load_cfd_dataset().

The dummy dataset and Member 2's real delivery use the same folder layout
(see D:\\Research\\Component3-Data-Contract.md), so "placeholder" and "real"
differ only by the folder they point at:

    <data_dir>/run_table.csv, design_table.csv, case.json (optional)
    <data_dir>/faces/<category>_<design_id>_<speed>kmh.npz

Contract facts used here:
  * one run-table row per design AND speed; the surface fields are p [Pa] and
    tau [Pa] per face, so the field handed to POD is Cp = p / q_ref;
  * the model target is Cd_mean (the converged, averaged value);
  * the split is BY DESIGN (both speeds of a design stay on the same side),
    and the hold-out is drawn from interior designs only (the extreme corner
    designs and the baseline always stay in training).

Array names inside the NPZ files are an ASSUMPTION until the first real file
arrives, so read_run_npz() accepts a few common aliases.
"""

import json
import os

import numpy as np
import pandas as pd

import config

COLUMN_ALIASES = {
    "y+_mean": "y_plus_mean", "yplus_mean": "y_plus_mean", "y+_max": "y_plus_max", "yplus_max": "y_plus_max",
    "A_front": "A_frontal", "frontal_area": "A_frontal", "checkMesh": "checkMesh_passed",
    "morph": "morph_passed", "speed": "speed_kmh",
}
ARRAY_ALIASES = {
    "centre": ["centre", "center", "face_centre", "face_center", "centres", "centers"],
    "area": ["area", "areas", "face_area", "face_areas"],
    "normal": ["normal", "normals", "face_normal", "face_normals"],
    "p": ["p", "pressure"],
    "tau": ["tau", "wall_shear", "wss", "wallShearStress", "shear"],
}
COMPONENT_KEYS = {"centre": ("cx", "cy", "cz"), "normal": ("nx", "ny", "nz"), "tau": ("tx", "ty", "tz")}


def normalize_run_table(df):
    return df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns and v not in df.columns})


def read_run_npz(path):
    """Return a dict with centre (n,3), area (n,), normal (n,3), p (n,), tau (n,3) as float64."""
    raw = np.load(path)
    out = {}
    for name, aliases in ARRAY_ALIASES.items():
        key = next((a for a in aliases if a in raw.files), None)
        if key is not None:
            out[name] = np.asarray(raw[key], dtype=np.float64)
        elif name in COMPONENT_KEYS and all(c in raw.files for c in COMPONENT_KEYS[name]):
            out[name] = np.stack([raw[c] for c in COMPONENT_KEYS[name]], axis=1).astype(np.float64)
        else:
            raise KeyError(f"{os.path.basename(path)}: no array for '{name}' (tried {aliases}); has {raw.files}")
    return out


def faces_path(data_dir, design_id, speed_kmh, category=config.CATEGORY):
    return os.path.join(data_dir, "faces", f"{category}_{design_id}_{speed_kmh}kmh.npz")


def _complete_design_table(designs):
    """Member 2's design_table.csv is promised to hold only design_id and the 5
    shape values. The hold-out rule also needs to know the baseline, the
    extreme corner designs and the run order, so use the columns if they are
    there and otherwise work them out: the baseline is D000, and the corners
    are the 5 designs nearest to the corners of the shape box."""
    designs = designs.copy()
    if "is_baseline" not in designs:
        designs["is_baseline"] = designs["design_id"] == "D000"
    if "is_corner" not in designs:
        lo = np.array([config.SHAPE_PARAM_BOUNDS[n][0] for n in config.SHAPE_PARAM_NAMES])
        hi = np.array([config.SHAPE_PARAM_BOUNDS[n][1] for n in config.SHAPE_PARAM_NAMES])
        unit = (designs[config.SHAPE_PARAM_NAMES].to_numpy(dtype=float) - lo) / (hi - lo)
        corners = np.array(np.meshgrid(*[[0.0, 1.0]] * len(lo))).reshape(len(lo), -1).T
        gap = np.min(np.linalg.norm(unit[:, None, :] - corners[None, :, :], axis=2), axis=1)
        gap[designs["is_baseline"].to_numpy()] = np.inf
        designs["is_corner"] = False
        designs.loc[designs.index[np.argsort(gap)[:5]], "is_corner"] = True
    if "run_order" not in designs:
        designs["run_order"] = np.arange(len(designs))
    return designs


class CFDDataset:
    """
    runs: one row per usable design at ONE speed. Columns: run_id (the
          design_id), is_holdout, the 5 shape values, Cd (= Cd_mean),
          Cd_pressure, Cd_friction, Cl, frontal_area, q_ref, is_corner,
          is_baseline, plus the other run-table columns.
    get_field(run_id): Cp = p / q_ref for that design (one value per face).
    get_faces(run_id): the raw per-face arrays (centre, area, normal, p, tau).
    mesh_face_areas: face areas of the baseline D000 (reference only).
    """

    def __init__(self, data_dir, speed_kmh=config.DEFAULT_SPEED_KMH, category=config.CATEGORY):
        self.data_dir, self.speed_kmh, self.category = data_dir, speed_kmh, category
        runs = normalize_run_table(pd.read_csv(os.path.join(data_dir, "run_table.csv")))
        designs = pd.read_csv(os.path.join(data_dir, "design_table.csv"))
        # case.json is optional (not part of the agreed contract): without it the
        # constants in config.py are used.
        case_path = os.path.join(data_dir, "case.json")
        self.case = {}
        if os.path.exists(case_path):
            with open(case_path) as fh:
                self.case = json.load(fh)

        runs = runs[runs["speed_kmh"] == speed_kmh].copy()
        usable = runs["morph_passed"].astype(bool) & runs["converged"].astype(bool)
        usable &= runs["design_id"].map(lambda d: os.path.exists(faces_path(data_dir, d, speed_kmh, category)))
        self.n_failed = int((~usable).sum())
        designs = _complete_design_table(designs)
        runs = runs[usable].merge(designs[["design_id", "is_corner", "is_baseline", "run_order"]], on="design_id", how="left")
        # The run table's "Cd" is the final-iteration value; the model target is Cd_mean.
        runs = runs.rename(columns={"Cd": "Cd_final_iteration"})
        runs = runs.rename(columns={"design_id": "run_id", "Cd_mean": "Cd", "Cd_p": "Cd_pressure",
                                    "Cd_f": "Cd_friction", "A_frontal": "frontal_area"})
        self.runs = runs.sort_values("run_order").reset_index(drop=True)
        self.assign_holdout(config.RANDOM_STATE + 7)

        base = faces_path(data_dir, "D000", speed_kmh, category)
        self.mesh_face_areas = read_run_npz(base)["area"] if os.path.exists(base) else None

    def assign_holdout(self, seed):
        """Hold out ~HOLDOUT_FRACTION of the designs, drawn from the interior
        designs only (corners and the baseline always train)."""
        runs = self.runs
        eligible = runs[~runs["is_corner"].astype(bool) & ~runs["is_baseline"].astype(bool)]["run_id"].sort_values()
        n_hold = int(round(config.HOLDOUT_FRACTION * len(runs)))
        rng = np.random.default_rng(seed)
        holdout_ids = set(rng.choice(eligible.to_numpy(), size=min(n_hold, len(eligible)), replace=False).tolist())
        runs["is_holdout"] = runs["run_id"].isin(holdout_ids)

    def with_split(self, seed):
        """A copy of this dataset with a different hold-out draw (same rules)."""
        import copy
        other = copy.copy(self)
        other.runs = self.runs.copy()
        other.assign_holdout(seed)
        return other

    def get_faces(self, run_id):
        return read_run_npz(faces_path(self.data_dir, run_id, self.speed_kmh, self.category))

    def get_field(self, run_id):
        faces = self.get_faces(run_id)
        q_ref = float(self.runs.loc[self.runs["run_id"] == run_id, "q_ref"].iloc[0])
        return faces["p"] / q_ref

    def real_runs(self):
        return self.runs[~self.runs["is_holdout"]].reset_index(drop=True)

    def holdout_runs(self):
        return self.runs[self.runs["is_holdout"]].reset_index(drop=True)


def load_cfd_dataset(source="placeholder", speed_kmh=config.DEFAULT_SPEED_KMH):
    """The single entry point. source: 'placeholder' (dummy data), 'real', or a folder path."""
    if source == "placeholder":
        data_dir = config.PLACEHOLDER_DATA_DIR
        if not os.path.exists(os.path.join(data_dir, "run_table.csv")):
            raise FileNotFoundError(f"No placeholder dataset in {data_dir}. Run placeholder_cfd_generator.py first.")
    elif source == "real":
        if not config.REAL_DATA_DIR:
            raise NotImplementedError("Set config.REAL_DATA_DIR to Member 2's shared folder once it exists.")
        data_dir = config.REAL_DATA_DIR
    elif os.path.isdir(source):
        data_dir = source  # any folder in the contract layout (e.g. the 2D CFD stand-in)
    else:
        raise ValueError(f"Unknown source '{source}', expected 'placeholder', 'real' or a data folder")
    return CFDDataset(data_dir, speed_kmh)
