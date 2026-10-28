"""
Placeholder (dummy) sedan dataset in Member 2's delivery format.

NEVER a source of real aerodynamic truth: the body is a simple lofted shape
and the pressure field is a hand-made function of the surface normal and the
shape values. What it DOES reproduce is the exact file layout, columns, units
and axes of the data contract (D:\\Research\\Component3-Data-Contract.md), so
the loader, checker and the whole model builder can be built and tested now
and pointed at the real folder later.

Layout written to the output folder:
    run_table.csv          one row per design and speed
    design_table.csv       design_id + the 5 shape values (+ run order, flags)
    case.json              case constants and conventions
    faces/sedan_connectivity.npz         triangle indices, shared by all runs
    faces/sedan_D017_100kmh.npz ...      per-face arrays, one file per run

Geometry (CAD axes): Y up, car length along Z, nose toward +Z, width along X;
the air therefore flows toward -Z. The drag coefficient stored in the run
table IS the sum of p*n*A and tau*A over the faces (plus a little run noise),
so the integrity check passes on this data by construction.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

from component3 import config
LENGTH_M = 4.7
NZ, NTHETA = 60, 72
ROOF_HEIGHT_M = 1.38
SUPERELLIPSE_N = 3.5
RHO = config.AIR_DENSITY_KG_M3
PARAM_COLS = config.SHAPE_PARAM_NAMES


# ------------------------------------------------------------------ #
# GEOMETRY
# ------------------------------------------------------------------ #
def _top_profile(z, ws_deg, sp_deg, bt_deg, hd_deg, clearance_m):
    z_rear, z_nose = -LENGTH_M / 2, LENGTH_M / 2
    y_nose = clearance_m + 0.50
    z_cowl = z_nose - 1.05
    y_cowl = y_nose + 1.05 * np.tan(np.radians(hd_deg))
    rise = ROOF_HEIGHT_M - y_cowl
    z_roof_front = z_cowl - rise / np.tan(np.radians(ws_deg))
    z_roof_rear = -0.60
    drop = (z_roof_rear - z_rear) * np.tan(np.radians(bt_deg))
    zs = [z_rear, z_roof_rear, z_roof_front, z_cowl, z_nose]
    ys = [ROOF_HEIGHT_M - drop, ROOF_HEIGHT_M, ROOF_HEIGHT_M, y_cowl, y_nose]
    top = np.interp(z, zs, ys)
    spoiler = 0.35 * np.sin(np.radians(sp_deg)) * np.exp(-(((z - z_rear) / 0.18) ** 2))
    return top + spoiler


def build_vertices(params):
    """Return the (n_vertices, 3) vertices of the deformed body. Topology is
    fixed, so every design has the same vertices in the same order."""
    ws, sp = params["windshield_angle_deg"], params["spoiler_angle_deg"]
    bt, hd = params["boat_tail_angle_deg"], params["hood_angle_deg"]
    clearance = params["ground_clearance_mm"] / 1000.0

    z = np.linspace(-LENGTH_M / 2, LENGTH_M / 2, NZ)
    top = _top_profile(z, ws, sp, bt, hd, clearance)
    bottom = np.full_like(z, clearance)
    yc, b = (top + bottom) / 2, (top - bottom) / 2
    a = 0.90 + 0.02 * np.tanh(z)

    z_n0 = 2.0
    s = np.sqrt(np.clip(1 - ((z - z_n0) / (LENGTH_M / 2 - z_n0)) ** 2, 0.0, 1.0))
    s = np.where(z > z_n0, np.maximum(s, 0.06), 1.0)

    theta = np.linspace(0, 2 * np.pi, NTHETA, endpoint=False)
    cth, sth = np.cos(theta), np.sin(theta)
    ex = np.sign(cth) * np.abs(cth) ** (2 / SUPERELLIPSE_N)
    ey = np.sign(sth) * np.abs(sth) ** (2 / SUPERELLIPSE_N)

    x = (a * s)[:, None] * ex[None, :]
    y = yc[:, None] + (b * s)[:, None] * ey[None, :]
    zz = np.repeat(z[:, None], NTHETA, axis=1)
    rings = np.stack([x, y, zz], axis=-1).reshape(-1, 3)

    rear_centre = np.array([[0.0, yc[0], z[0]]])
    nose_centre = np.array([[0.0, yc[-1], z[-1]]])
    return np.vstack([rings, rear_centre, nose_centre])


def build_triangles():
    """Fixed triangle connectivity (n_faces, 3), independent of the design."""
    tris = []
    for j in range(NZ - 1):
        for k in range(NTHETA):
            k1 = (k + 1) % NTHETA
            a, b = j * NTHETA + k, j * NTHETA + k1
            c, d = (j + 1) * NTHETA + k1, (j + 1) * NTHETA + k
            tris += [[a, b, c], [a, c, d]]
    rear_c, nose_c = NZ * NTHETA, NZ * NTHETA + 1
    for k in range(NTHETA):
        k1 = (k + 1) % NTHETA
        tris.append([rear_c, k1, k])
        tris.append([nose_c, (NZ - 1) * NTHETA + k, (NZ - 1) * NTHETA + k1])
    return np.array(tris, dtype=np.int32)


def face_geometry(vertices, triangles):
    v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
    cross = np.cross(v1 - v0, v2 - v0)
    twice_area = np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(twice_area, 1e-12)[:, None]
    return (v0 + v1 + v2) / 3.0, 0.5 * twice_area, normal


def _orient_triangles(triangles, vertices):
    """Flip triangle winding where needed so every normal points out of the
    body (decided once, on the baseline, because topology never changes)."""
    centre, _, normal = face_geometry(vertices, triangles)
    outward = np.einsum("ij,ij->i", normal, centre - vertices.mean(axis=0)) > 0
    fixed = triangles.copy()
    fixed[~outward] = fixed[~outward][:, [0, 2, 1]]
    return fixed


# ------------------------------------------------------------------ #
# FLOW FIELD (hand-made, NOT physics) AND FORCES
# ------------------------------------------------------------------ #
def surface_fields(params, centre, area, normal, speed_ms, rng):
    d = np.array(config.FLOW_DIRECTION)
    q = 0.5 * RHO * speed_ms ** 2
    c = -(normal @ d)            # +1 on the nose, -1 on the rear
    ny, nx = normal[:, 1], normal[:, 0]
    z = centre[:, 2]
    bt = params["boat_tail_angle_deg"]
    sp = params["spoiler_angle_deg"]
    clearance = params["ground_clearance_mm"] / 1000.0

    k_base = 0.30 - 0.014 * bt + 0.00045 * bt ** 2 - 0.0015 * sp + 0.25 * (clearance - 0.16)
    k_base *= 0.55 * (1 + 0.05 * ((27.78 - speed_ms) / 16.67) * (0.5 + bt / 20))

    front = 0.62 * np.clip(c, 0, 1) ** 2
    top = -0.70 * np.clip(ny, 0, 1) ** 8 * np.exp(-(((z - 0.4) / 1.6) ** 2))
    rear = -k_base * np.clip(-c, 0, 1)
    side = -0.30 * np.clip(np.abs(nx) - 0.6, 0, 1)
    under = -0.30 * np.clip(-ny, 0, 1) ** 4 * (0.16 / clearance) ** 0.6
    cp = front + top + rear + side + under + rng.normal(0, 0.003, len(c))

    cf0 = 0.0030 * (27.78 / speed_ms) ** 0.2
    tangent = d[None, :] - (normal @ d)[:, None] * normal
    norm_t = np.linalg.norm(tangent, axis=1)
    tangent = tangent / np.maximum(norm_t, 1e-9)[:, None]
    weight = (1 - 0.7 * np.clip(-c, 0, 1)) * (norm_t > 1e-3)
    tau = (cf0 * q * weight)[:, None] * tangent
    return cp * q, tau.astype(np.float64)


def integrate_forces(centre, area, normal, p, tau, speed_ms):
    d, lift = np.array(config.FLOW_DIRECTION), np.array(config.LIFT_DIRECTION)
    q = 0.5 * RHO * speed_ms ** 2
    a_front = 0.5 * np.sum(np.abs(normal @ d) * area)
    cd_p = np.sum(-p * (normal @ d) * area) / (q * a_front)
    cd_f = np.sum((tau @ d) * area) / (q * a_front)
    cl_p = np.sum(-p * (normal @ lift) * area) / (q * a_front)
    cl_f = np.sum((tau @ lift) * area) / (q * a_front)
    return dict(Cd_p=cd_p, Cd_f=cd_f, Cd=cd_p + cd_f, Cl_p=cl_p, Cl_f=cl_f, A_frontal=a_front, q_ref=q)


# ------------------------------------------------------------------ #
# DESIGN TABLE
# ------------------------------------------------------------------ #
def _make_design_table(n_designs, rng_seed):
    names = PARAM_COLS
    lo = np.array([config.SHAPE_PARAM_BOUNDS[n][0] for n in names])
    hi = np.array([config.SHAPE_PARAM_BOUNDS[n][1] for n in names])
    unit = qmc.LatinHypercube(d=len(names), seed=rng_seed).random(n_designs)
    values = qmc.scale(unit, lo, hi)

    corner_targets = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 1]], float)
    corner_idx = []
    for target in corner_targets:
        dist = np.linalg.norm(unit - target, axis=1)
        dist[corner_idx] = np.inf
        corner_idx.append(int(np.argmin(dist)))

    rng = np.random.default_rng(rng_seed + 1)
    others = [i for i in range(n_designs) if i not in corner_idx]
    order = corner_idx + list(rng.permutation(others))
    run_order = np.empty(n_designs, dtype=int)
    run_order[order] = np.arange(1, n_designs + 1)

    interior = [i for i in others if i not in corner_idx]
    failed = set(rng.choice(interior, size=2, replace=False).tolist())

    table = pd.DataFrame(values, columns=names)
    table.insert(0, "design_id", [f"D{i + 1:03d}" for i in range(n_designs)])
    table["run_order"] = run_order
    table["is_corner"] = [i in corner_idx for i in range(n_designs)]
    table["is_baseline"] = False
    table["morph_passed"] = [i not in failed for i in range(n_designs)]
    baseline = {"design_id": "D000", "run_order": 0, "is_corner": False, "is_baseline": True, "morph_passed": True}
    baseline.update(config.BASELINE_SHAPE)
    return pd.concat([pd.DataFrame([baseline]), table], ignore_index=True)[
        ["design_id"] + names + ["run_order", "is_corner", "is_baseline", "morph_passed"]
    ]


# ------------------------------------------------------------------ #
# WRITE THE DATASET
# ------------------------------------------------------------------ #
def generate_dataset(out_dir=config.PLACEHOLDER_DATA_DIR, n_designs=config.N_DESIGNS, seed=config.RANDOM_STATE):
    faces_dir = os.path.join(out_dir, "faces")
    os.makedirs(faces_dir, exist_ok=True)
    rng = np.random.default_rng(seed + 100)

    design_table = _make_design_table(n_designs, seed)
    design_table.to_csv(os.path.join(out_dir, "design_table.csv"), index=False)

    base_vertices = build_vertices(config.BASELINE_SHAPE)
    triangles = _orient_triangles(build_triangles(), base_vertices)
    np.savez_compressed(os.path.join(faces_dir, f"{config.CATEGORY}_connectivity.npz"), triangles=triangles)

    rows = []
    for _, d in design_table.iterrows():
        params = {n: float(d[n]) for n in PARAM_COLS}
        for kmh, ms in config.SPEEDS_MS.items():
            row = {"design_id": d["design_id"], "speed_kmh": kmh, "speed_ms": ms, "baseline_flag": bool(d["is_baseline"])}
            row.update(params)
            ok = bool(d["morph_passed"])
            row.update(checkMesh_passed=ok, morph_passed=ok, converged=ok)
            if not ok:
                rows.append(row)
                continue
            centre, area, normal = face_geometry(build_vertices(params), triangles)
            p, tau = surface_fields(params, centre, area, normal, ms, rng)
            f = integrate_forces(centre, area, normal, p, tau, ms)
            cd_mean = f["Cd"] + rng.normal(0, 0.0004)
            row.update(
                Cd=cd_mean + rng.normal(0, 0.0007), Cd_mean=cd_mean, Cd_std=abs(rng.normal(0.0006, 0.0001)),
                Cd_p=f["Cd_p"], Cd_f=f["Cd_f"], Cl=f["Cl_p"] + f["Cl_f"], Cl_p=f["Cl_p"], Cl_f=f["Cl_f"],
                A_frontal=f["A_frontal"], q_ref=f["q_ref"], iterations=int(rng.integers(1200, 2000)),
                y_plus_mean=float(rng.uniform(30, 60)), y_plus_max=float(rng.uniform(90, 140)),
                cell_count=4_800_000, final_res_p=float(10 ** rng.uniform(-5.5, -4.2)),
                final_res_U=float(10 ** rng.uniform(-6, -4.5)),
            )
            np.savez_compressed(
                os.path.join(faces_dir, f"{config.CATEGORY}_{d['design_id']}_{kmh}kmh.npz"),
                centre=centre.astype(np.float32), area=area.astype(np.float32), normal=normal.astype(np.float32),
                p=p.astype(np.float32), tau=tau.astype(np.float32),
            )
            rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "run_table.csv"), index=False)
    case = {
        "category": config.CATEGORY, "source": "PLACEHOLDER (synthetic), not real CFD", "full_car": True,
        "rho_kg_m3": RHO, "mu_pa_s": config.DYNAMIC_VISCOSITY_PA_S, "speeds_ms": config.SPEEDS_MS,
        "axes": "Y up, Z length (nose toward +Z), X width", "flow_direction": list(config.FLOW_DIRECTION),
        "normals": "outward from the car into the fluid", "p_ref_pa": 0.0,
        "tau_convention": "force per area exerted by the fluid on the car (assumed; confirm with Member 2)",
        "averaging": "last 100 iterations (fieldAverage)", "n_faces": int(len(triangles)),
    }
    with open(os.path.join(out_dir, "case.json"), "w") as fh:
        json.dump(case, fh, indent=2)
    # The design box this data was made with, in Member 2's parameter_ranges.csv columns.
    pd.DataFrame([
        {"parameter": n, "unit": "mm" if n.endswith("_mm") else "deg", "baseline_D000": config.BASELINE_SHAPE[n],
         "design_min": config.SHAPE_PARAM_BOUNDS[n][0], "design_max": config.SHAPE_PARAM_BOUNDS[n][1]}
        for n in config.SHAPE_PARAM_NAMES
    ]).to_csv(os.path.join(out_dir, "parameter_ranges.csv"), index=False)
    return design_table, pd.DataFrame(rows)


if __name__ == "__main__":
    dt, rt = generate_dataset()
    ok = rt[rt["morph_passed"]]
    print(f"Wrote {len(dt)} designs ({int((~dt['morph_passed']).sum())} failed) -> {len(rt)} run-table rows, "
          f"{len(ok)} with face files, to {config.PLACEHOLDER_DATA_DIR}")
    base = rt[(rt["design_id"] == "D000")][["speed_kmh", "Cd_mean", "Cd_p", "Cd_f", "Cl", "A_frontal"]]
    print("Baseline D000:"); print(base.to_string(index=False))
    print("Cd_mean range over designs (100 km/h):",
          round(ok[ok.speed_kmh == 100].Cd_mean.min(), 4), "to", round(ok[ok.speed_kmh == 100].Cd_mean.max(), 4))
