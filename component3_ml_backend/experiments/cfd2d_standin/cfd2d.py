"""
2D CFD stand-in for Member 2's sedan data (runs in Google Colab with OpenFOAM).

What it makes: a 2D side profile (centreline section) of a sedan controlled
by the same 5 shape parameters as Member 2's car, meshed with ONE fixed block
structure (same faces in the same order for every design, like Member 2's
Method A), solved with OpenFOAM simpleFoam (steady RANS, k-omega SST) at the
two agreed speeds, and written in Member 2's exact delivery format:

    <out>/design_table.csv, run_table.csv, case.json
    <out>/faces/sedan_D017_100kmh.npz   (centre, area, normal, p [Pa], tau [Pa], float32)

It is REAL flow physics (separation, wake), but 2D: no flow around the sides,
so the Cd values are not those of a 3D car. Built to test the Component 3
pipeline on real CFD in Member 2's structure before his data arrives.

Case coordinates: x = streamwise (air flows +x, nose at x = 0), y = up,
z = span (empty direction, SPAN metres deep). Saved files use the contract
axes: X = width (= z), Y = up, Z = length with the nose toward +Z (= -x),
so the air flows toward -Z.
"""

import glob
import json
import os
import re
import shutil
import subprocess

import numpy as np

# ------------------------------------------------------------------ #
# CONSTANTS (match Component 3 config.py and the data contract)
# ------------------------------------------------------------------ #
RHO = 1.165
MU = 1.86e-5
NU = MU / RHO
SPEEDS_MS = {100: 27.78, 40: 11.11}
SPAN = 1.6                      # m, depth of the 2D slice (only scales areas)
CATEGORY = "sedan"
PARAMS = ["windshield_angle_deg", "spoiler_angle_deg", "boat_tail_angle_deg", "hood_angle_deg", "ground_clearance_mm"]
BOUNDS = {"windshield_angle_deg": (25.0, 35.0), "spoiler_angle_deg": (0.0, 25.0), "boat_tail_angle_deg": (0.0, 20.0),
          "hood_angle_deg": (1.0, 11.0), "ground_clearance_mm": (130.0, 190.0)}
BASELINE = {"windshield_angle_deg": 30.0, "spoiler_angle_deg": 5.0, "boat_tail_angle_deg": 8.0,
            "hood_angle_deg": 6.0, "ground_clearance_mm": 160.0}
N_DESIGNS = 70
SEED = 42

# Body dimensions (m)
L_BODY = 4.7          # nose to rear face
H_BODY = 1.25         # roof height above the underbody
HOOD_LEN = 1.05
X_ROOF_REAR = 3.05
DECK_END = 4.40       # end of the boat-tail slope, start of the ducktail spoiler
NOSE_TOP = 0.60       # height of the hood start above the underbody

# Domain (m)
X_IN, X_OUT, Y_TOP = -8.0, L_BODY + 20.0, 8.0

# Mesh resolution
N_NOSE, N_HW, N_ROOF, N_BT, N_REAR, N_BOT = 24, 56, 30, 48, 30, 60
N_FAR, N_GAP = 90, 22
FIRST_CELL = 0.004    # m, wall-normal size of the first cell next to the car
POLY_PTS = 40         # interior points per curved car edge

# Solver
N_ITER = 1500
AVG_WINDOW = 500      # fields and Cd averaged over the last AVG_WINDOW iterations
# (trial: Cd settles by about 1000 iterations and then wobbles by about +/-3 drag counts, so 1500 with a
#  500-iteration average gives the same answer as 3000 in half the time)
TURB_INTENSITY = 0.005
VISC_RATIO = 10.0


# ------------------------------------------------------------------ #
# 1. DESIGN TABLE (70 LHS designs + baseline D000)
# ------------------------------------------------------------------ #
def make_design_table(n_designs=N_DESIGNS, seed=SEED):
    import pandas as pd
    from scipy.stats import qmc
    lo = np.array([BOUNDS[p][0] for p in PARAMS])
    hi = np.array([BOUNDS[p][1] for p in PARAMS])
    unit = qmc.LatinHypercube(d=len(PARAMS), seed=seed).random(n_designs)
    values = qmc.scale(unit, lo, hi)
    targets = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 1]], float)
    corners = []
    for t in targets:
        d = np.linalg.norm(unit - t, axis=1)
        d[corners] = np.inf
        corners.append(int(np.argmin(d)))
    rng = np.random.default_rng(seed + 1)
    others = [i for i in range(n_designs) if i not in corners]
    order = corners + list(rng.permutation(others))
    run_order = np.empty(n_designs, dtype=int)
    run_order[order] = np.arange(1, n_designs + 1)
    table = pd.DataFrame(np.round(values, 4), columns=PARAMS)
    table.insert(0, "design_id", [f"D{i + 1:03d}" for i in range(n_designs)])
    table["is_corner"] = [i in corners for i in range(n_designs)]
    table["run_order"] = run_order
    base = {"design_id": "D000", "is_corner": False, "run_order": 0, **BASELINE}
    return pd.concat([pd.DataFrame([base]), table], ignore_index=True)[["design_id"] + PARAMS + ["is_corner", "run_order"]]


# ------------------------------------------------------------------ #
# 2. GEOMETRY: the 2D sedan profile
# ------------------------------------------------------------------ #
def _line(a, b, n):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def _bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def profile(params):
    """Return the key points and the curved car edges (case coordinates)."""
    ws, sp = np.radians(params["windshield_angle_deg"]), np.radians(params["spoiler_angle_deg"])
    bt, hd = np.radians(params["boat_tail_angle_deg"]), np.radians(params["hood_angle_deg"])
    c = params["ground_clearance_mm"] / 1000.0
    y_roof = c + H_BODY

    P0 = np.array([0.35, c])                                   # front of the flat underbody
    F = np.array([0.0, c + 0.30])                              # bumper front
    P1 = np.array([0.08, c + NOSE_TOP])                        # hood start
    cowl = P1 + HOOD_LEN * np.array([1.0, np.tan(hd)])         # hood end / windshield base
    FT = np.array([cowl[0] + (y_roof - cowl[1]) / np.tan(ws), y_roof])   # windshield top = roof front
    RT = np.array([X_ROOF_REAR, y_roof])                       # roof rear
    B = np.array([DECK_END, y_roof - (DECK_END - X_ROOF_REAR) * np.tan(bt)])  # end of the boat-tail slope
    S = np.array([L_BODY, B[1] + (L_BODY - DECK_END) * np.tan(sp)])          # ducktail spoiler tip
    R1 = np.array([L_BODY, c + 0.15])
    P5 = np.array([L_BODY - 0.15, c])                          # rear end of the flat underbody

    lower_nose = _bezier(P0, P0 + 0.15 * np.array([np.cos(np.radians(120)), np.sin(np.radians(120))]),
                         F - np.array([0.0, 0.12]), F, 25)
    hood_dir = np.array([np.cos(hd), np.sin(hd)])
    upper_nose = _bezier(F, F + np.array([0.0, 0.15]), P1 - 0.12 * hood_dir, P1, 20)
    nose = np.vstack([lower_nose, upper_nose[1:]])
    hood_ws = np.vstack([_line(P1, cowl, 25), _line(cowl, FT, 30)[1:]])
    deck = np.vstack([_line(RT, B, 30), _line(B, S, 12)[1:]])
    rear = np.vstack([_line(S, R1, 20), _line(R1, P5, 8)[1:]])

    points = {"P0": P0, "P1": P1, "FT": FT, "RT": RT, "S": S, "P5": P5}
    curves = {("P0", "P1"): nose, ("P1", "FT"): hood_ws, ("RT", "S"): deck, ("S", "P5"): rear}
    return points, curves


def outline(params):
    """Closed car outline (for plots and checks), from P0 over the top and back."""
    pts, curves = profile(params)
    return np.vstack([curves[("P0", "P1")], curves[("P1", "FT")][1:], _line(pts["FT"], pts["RT"], 10)[1:],
                      curves[("RT", "S")][1:], curves[("S", "P5")][1:], _line(pts["P5"], pts["P0"], 10)[1:]])


def resample(curve, n):
    """n points evenly spaced by arc length along a polyline."""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, curve[:, 0]), np.interp(t, s, curve[:, 1])])


# ------------------------------------------------------------------ #
# 3. BLOCK STRUCTURE (identical for every design)
# ------------------------------------------------------------------ #
# Blocks given as (bottom-left, bottom-right, top-right, top-left) in the sense
# "first edge along the car / ground, second edge going outward".
BLOCKS = [
    ("I0", "G0", "P0", "IA"),    # under the front overhang, upstream
    ("IA", "P0", "P1", "IT"),    # nose
    ("P1", "FT", "TF", "IT"),    # hood + windshield
    ("FT", "RT", "TR", "TF"),    # roof
    ("RT", "S", "ST", "TR"),     # boat-tail + spoiler
    ("S", "P5", "OC", "Q"),      # rear face / near wake
    ("S", "Q", "OT", "ST"),      # wake above the spoiler tip
    ("G5", "O0", "OC", "P5"),    # under the wake
    ("G0", "G5", "P5", "P0"),    # underbody gap
]

# Edge cell counts and grading, keyed by a canonical direction (a, b).
#   ("uniform",)                      uniform spacing
#   ("first", h)                      first cell h at a, growing toward b
#   ("both", r)                       clustered at both ends (two-sided, ratio r per half)
#   ("ratio", R)                      last/first = R from a to b
def edge_table():
    t = {
        ("P0", "P1"): (N_NOSE, ("uniform",)), ("P1", "FT"): (N_HW, ("uniform",)), ("FT", "RT"): (N_ROOF, ("uniform",)),
        ("RT", "S"): (N_BT, ("uniform",)), ("S", "P5"): (N_REAR, ("uniform",)), ("P5", "P0"): (N_BOT, ("uniform",)),
        ("G0", "G5"): (N_BOT, ("uniform",)),
        # outward from the car
        ("P1", "IT"): (N_FAR, ("first", FIRST_CELL)), ("FT", "TF"): (N_FAR, ("first", FIRST_CELL)),
        ("RT", "TR"): (N_FAR, ("first", FIRST_CELL)), ("S", "ST"): (N_FAR, ("first", FIRST_CELL)),
        ("S", "Q"): (N_FAR, ("first", FIRST_CELL)),
        ("P0", "IA"): (N_FAR, ("first", FIRST_CELL)), ("P5", "OC"): (N_FAR, ("first", FIRST_CELL)),
        ("G0", "I0"): (N_FAR, ("first", FIRST_CELL)), ("G5", "O0"): (N_FAR, ("first", FIRST_CELL)),
        # far boundaries opposite the car edges
        ("IA", "IT"): (N_NOSE, ("ratio", 10.0)), ("IT", "TF"): (N_HW, ("uniform",)), ("TF", "TR"): (N_ROOF, ("uniform",)),
        ("TR", "ST"): (N_BT, ("uniform",)), ("ST", "OT"): (N_FAR, ("uniform",)),
        ("Q", "OT"): (N_FAR, ("uniform",)), ("OC", "Q"): (N_REAR, ("uniform",)),
        # between ground and underbody height
        ("I0", "IA"): (N_GAP, ("both", 4.0)), ("G0", "P0"): (N_GAP, ("both", 4.0)),
        ("G5", "P5"): (N_GAP, ("both", 4.0)), ("O0", "OC"): (N_GAP, ("both", 4.0)),
    }
    return t


def all_points(params):
    pts, curves = profile(params)
    c = params["ground_clearance_mm"] / 1000.0
    pts = dict(pts)
    pts.update({
        "I0": np.array([X_IN, 0.0]), "IA": np.array([X_IN, c]), "IT": np.array([X_IN, Y_TOP]),
        "G0": np.array([pts["P0"][0], 0.0]), "G5": np.array([pts["P5"][0], 0.0]),
        "TF": np.array([pts["FT"][0], Y_TOP]), "TR": np.array([pts["RT"][0], Y_TOP]),
        "OT": np.array([X_OUT, Y_TOP]), "OC": np.array([X_OUT, c]), "O0": np.array([X_OUT, 0.0]),
        "Q": np.array([X_OUT, pts["S"][1]]),
        "ST": np.array([pts["S"][0] + (Y_TOP - pts["S"][1]) / np.tan(np.radians(55)), Y_TOP]),
    })
    return pts, curves


def _first_cell_ratio(length, n, h0):
    """Expansion ratio (last/first) so that n geometric cells starting at h0 fill length."""
    if length / n <= h0 * 1.0001:
        return 1.0
    lo, hi = 1.0 + 1e-9, 3.0
    for _ in range(200):
        g = 0.5 * (lo + hi)
        s = h0 * (g ** n - 1) / (g - 1)
        lo, hi = (g, hi) if s < length else (lo, g)
    return g ** (n - 1)


def edge_spec(a, b, pts, curves):
    """(n, grading) for the directed edge a -> b, grading in blockMesh syntax."""
    table = edge_table()
    if (a, b) in table:
        n, spec = table[(a, b)]
        rev = False
    else:
        n, spec = table[(b, a)]
        rev = True
    if spec[0] == "uniform":
        return n, "1"
    if spec[0] == "both":
        r = spec[1]
        return n, f"((0.5 0.5 {r:g}) (0.5 0.5 {1 / r:g}))"
    if spec[0] == "ratio":
        r = spec[1]
    else:
        key = (a, b) if not rev else (b, a)
        length = np.linalg.norm(pts[key[1]] - pts[key[0]])
        r = _first_cell_ratio(length, n, spec[1])
    return n, f"{(1 / r if rev else r):.8g}"


def edge_fractions(n, grading):
    """Point positions (0..1) along an edge for a blockMesh grading string."""
    if grading.startswith("(("):
        secs = [tuple(float(v) for v in s.split()) for s in re.findall(r"\(([^()]+)\)", grading)]
        out, start_l, cells_left = [0.0], 0.0, n
        counts = [int(round(s[1] * n)) for s in secs]
        counts[-1] = n - sum(counts[:-1])
        for (fl, _fc, r), nc in zip(secs, counts):
            f = _geom(nc, r)
            out += list(start_l + fl * f[1:])
            start_l += fl
        return np.array(out)
    return _geom(n, float(grading))


def _geom(n, r):
    if abs(r - 1) < 1e-12 or n == 1:
        return np.linspace(0, 1, n + 1)
    g = r ** (1.0 / (n - 1))
    sizes = g ** np.arange(n)
    return np.concatenate([[0], np.cumsum(sizes) / sizes.sum()])


def _ccw(quad):
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def oriented_blocks(pts):
    """Block corner names in counter-clockwise order (needed by blockMesh)."""
    out = []
    for blk in BLOCKS:
        quad = np.array([pts[n] for n in blk])
        out.append(blk if _ccw(quad) else (blk[0], blk[3], blk[2], blk[1]))
    return out


def edge_curve(a, b, pts, curves):
    """The points of edge a -> b (straight unless it is a curved car edge)."""
    if (a, b) in curves:
        return curves[(a, b)]
    if (b, a) in curves:
        return curves[(b, a)][::-1]
    return _line(pts[a], pts[b], 2)


# ------------------------------------------------------------------ #
# 4. PREVIEW MESH IN PYTHON (transfinite interpolation, for checks/plots)
# ------------------------------------------------------------------ #
def preview_mesh(params):
    """Approximate the blockMesh mesh with transfinite interpolation.
    Returns a list of (block name tuple, X array (n1+1, n2+1, 2))."""
    pts, curves = all_points(params)
    blocks = []
    for blk in oriented_blocks(pts):
        v0, v1, v2, v3 = blk
        n1, g01 = edge_spec(v0, v1, pts, curves)
        _, g32 = edge_spec(v3, v2, pts, curves)
        n2, g03 = edge_spec(v0, v3, pts, curves)
        _, g12 = edge_spec(v1, v2, pts, curves)

        def along(a, b, g, n):
            cur = resample(edge_curve(a, b, pts, curves), 400)
            seg = np.linalg.norm(np.diff(cur, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
            f = edge_fractions(n, g)
            return np.column_stack([np.interp(f, s, cur[:, 0]), np.interp(f, s, cur[:, 1])])

        bottom, top = along(v0, v1, g01, n1), along(v3, v2, g32, n1)
        left, right = along(v0, v3, g03, n2), along(v1, v2, g12, n2)
        u = np.linspace(0, 1, n1 + 1)[:, None, None]
        w = np.linspace(0, 1, n2 + 1)[None, :, None]
        # parameters from the edge fractions so grading is respected
        fu = (edge_fractions(n1, g01) + edge_fractions(n1, g32)) / 2
        fw = (edge_fractions(n2, g03) + edge_fractions(n2, g12)) / 2
        U, W = fu[:, None, None], fw[None, :, None]
        X = ((1 - W) * bottom[:, None, :] + W * top[:, None, :] + (1 - U) * left[None, :, :] + U * right[None, :, :]
             - ((1 - U) * (1 - W) * pts[v0] + U * (1 - W) * pts[v1] + U * W * pts[v2] + (1 - U) * W * pts[v3]))
        blocks.append((blk, X))
    return blocks


def mesh_quality(params):
    """Cell count, smallest cell area, share of inverted cells, worst corner angle (deg)."""
    n_cells, min_area, inverted, worst = 0, np.inf, 0, 180.0
    for _, X in preview_mesh(params):
        a, b, c, d = X[:-1, :-1], X[1:, :-1], X[1:, 1:], X[:-1, 1:]
        area = 0.5 * ((c[..., 0] - a[..., 0]) * (d[..., 1] - b[..., 1]) - (d[..., 0] - b[..., 0]) * (c[..., 1] - a[..., 1]))
        n_cells += area.size
        inverted += int((area <= 0).sum())
        min_area = min(min_area, float(area.min()))
        for p, q, r in ((d, a, b), (a, b, c), (b, c, d), (c, d, a)):
            v1, v2 = p - q, r - q
            cosang = np.sum(v1 * v2, -1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-30)
            worst = min(worst, float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).min()))
    return {"cells": n_cells, "min_area": min_area, "inverted": inverted, "min_corner_angle_deg": worst}


# ------------------------------------------------------------------ #
# 5. OPENFOAM CASE FILES
# ------------------------------------------------------------------ #
HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  Component 3 2D CFD stand-in (generated)
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def _w(path, cls, obj, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(HEADER.format(cls=cls, obj=obj) + body)


def block_mesh_dict(params):
    pts, curves = all_points(params)
    names = list(pts)
    idx = {n: i for i, n in enumerate(names)}
    npt = len(names)
    verts = [f"    ({pts[n][0]:.6f} {pts[n][1]:.6f} 0)  // {i} {n}" for i, n in enumerate(names)]
    verts += [f"    ({pts[n][0]:.6f} {pts[n][1]:.6f} {SPAN})  // {i + npt} {n}'" for i, n in enumerate(names)]

    blocks = []
    for blk in oriented_blocks(pts):
        v = [idx[n] for n in blk]
        v8 = v + [i + npt for i in v]
        n1, g01 = edge_spec(blk[0], blk[1], pts, curves)
        _, g32 = edge_spec(blk[3], blk[2], pts, curves)
        n2, g03 = edge_spec(blk[0], blk[3], pts, curves)
        _, g12 = edge_spec(blk[1], blk[2], pts, curves)
        grading = f"edgeGrading ({g01} {g32} {g32} {g01}  {g03} {g12} {g12} {g03}  1 1 1 1)"
        blocks.append(f"    hex ({' '.join(map(str, v8))}) ({n1} {n2} 1) {grading}  // {'-'.join(blk)}")

    edges = []
    for (a, b), cur in curves.items():
        inner = resample(cur, POLY_PTS + 2)[1:-1]
        for off, z in ((0, 0.0), (npt, SPAN)):
            plist = " ".join(f"({x:.6f} {y:.6f} {z})" for x, y in inner)
            edges.append(f"    polyLine {idx[a] + off} {idx[b] + off} ({plist})")

    def face(a, b):
        # outward face for the CCW block edge a -> b
        return f"({idx[a]} {idx[b]} {idx[b] + npt} {idx[a] + npt})"

    patches = {"inlet": [], "outlet": [], "top": [], "ground": [], "car": []}
    car_edges = {frozenset(k) for k in [("P0", "P1"), ("P1", "FT"), ("FT", "RT"), ("RT", "S"), ("S", "P5"), ("P5", "P0")]}
    front_back = []
    for blk in oriented_blocks(pts):
        ring = list(blk) + [blk[0]]
        for a, b in zip(ring[:-1], ring[1:]):
            pa, pb = pts[a], pts[b]
            if frozenset((a, b)) in car_edges:
                patches["car"].append(face(a, b))
            elif abs(pa[0] - X_IN) < 1e-9 and abs(pb[0] - X_IN) < 1e-9:
                patches["inlet"].append(face(a, b))
            elif abs(pa[0] - X_OUT) < 1e-9 and abs(pb[0] - X_OUT) < 1e-9:
                patches["outlet"].append(face(a, b))
            elif abs(pa[1] - Y_TOP) < 1e-9 and abs(pb[1] - Y_TOP) < 1e-9:
                patches["top"].append(face(a, b))
            elif abs(pa[1]) < 1e-9 and abs(pb[1]) < 1e-9:
                patches["ground"].append(face(a, b))
        v = [idx[n] for n in blk]
        front_back.append(f"({v[0]} {v[3]} {v[2]} {v[1]})")
        front_back.append(f"({v[0] + npt} {v[1] + npt} {v[2] + npt} {v[3] + npt})")

    types = {"inlet": "patch", "outlet": "patch", "top": "patch", "ground": "wall", "car": "wall"}
    bnd = []
    for name, faces in patches.items():
        bnd.append(f"    {name}\n    {{\n        type {types[name]};\n        faces\n        (\n            "
                   + "\n            ".join(faces) + "\n        );\n    }")
    bnd.append("    frontAndBack\n    {\n        type empty;\n        faces\n        (\n            "
               + "\n            ".join(front_back) + "\n        );\n    }")
    body = ("scale 1;\n\nvertices\n(\n" + "\n".join(verts) + "\n);\n\nblocks\n(\n" + "\n".join(blocks)
            + "\n);\n\nedges\n(\n" + "\n".join(edges) + "\n);\n\nboundary\n(\n" + "\n".join(bnd)
            + "\n);\n\nmergePatchPairs\n(\n);\n")
    return body


def frontal_area(params):
    o = outline(params)
    return float((o[:, 1].max() - o[:, 1].min()) * SPAN)


def write_case(case_dir, params, speed_ms):
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    k_in = 1.5 * (TURB_INTENSITY * speed_ms) ** 2
    om_in = k_in / (NU * VISC_RATIO)
    a_ref = frontal_area(params)
    _w(f"{case_dir}/system/blockMeshDict", "dictionary", "blockMeshDict", block_mesh_dict(params))
    _w(f"{case_dir}/system/controlDict", "dictionary", "controlDict", f"""
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {N_ITER};
deltaT          1;
writeControl    timeStep;
writeInterval   {N_ITER};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;

functions
{{
    wallShearStress1
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        patches         (car);
        executeControl  timeStep;
        writeControl    writeTime;
    }}
    yPlus1
    {{
        type            yPlus;
        libs            (fieldFunctionObjects);
        executeControl  writeTime;
        writeControl    writeTime;
    }}
    fieldAverage1
    {{
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        timeStart       {N_ITER - AVG_WINDOW};
        writeControl    writeTime;
        fields
        (
            p {{ mean on; prime2Mean off; base time; }}
            wallShearStress {{ mean on; prime2Mean off; base time; }}
        );
    }}
    forceCoeffs1
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (car);
        rho             rhoInf;
        rhoInf          {RHO};
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        CofR            ({L_BODY / 2} 0 0);
        pitchAxis       (0 0 1);
        magUInf         {speed_ms};
        lRef            {L_BODY};
        Aref            {a_ref:.6f};
    }}
}}
""")
    _w(f"{case_dir}/system/fvSchemes", "dictionary", "fvSchemes", """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.5; }
wallDist        { method meshWave; }
""")
    _w(f"{case_dir}/system/fvSolution", "dictionary", "fvSolution", """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.1;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      no;
}
relaxationFactors
{
    fields          { p 0.3; }
    equations       { U 0.7; k 0.7; omega 0.7; }
}
""")
    _w(f"{case_dir}/constant/transportProperties", "dictionary", "transportProperties",
       f"\ntransportModel  Newtonian;\nnu              {NU:.8g};\n")
    _w(f"{case_dir}/constant/turbulenceProperties", "dictionary", "turbulenceProperties",
       "\nsimulationType  RAS;\nRAS\n{\n    RASModel        kOmegaSST;\n    turbulence      on;\n    printCoeffs     on;\n}\n")

    U = f"({speed_ms} 0 0)"
    _w(f"{case_dir}/0/U", "volVectorField", "U", f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform {U};
boundaryField
{{
    inlet           {{ type fixedValue; value uniform {U}; }}
    outlet          {{ type inletOutlet; inletValue uniform (0 0 0); value uniform {U}; }}
    top             {{ type slip; }}
    ground          {{ type fixedValue; value uniform {U}; }}
    car             {{ type noSlip; }}
    frontAndBack    {{ type empty; }}
}}
""")
    _w(f"{case_dir}/0/p", "volScalarField", "p", """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet           { type zeroGradient; }
    outlet          { type fixedValue; value uniform 0; }
    top             { type slip; }
    ground          { type zeroGradient; }
    car             { type zeroGradient; }
    frontAndBack    { type empty; }
}
""")
    for name, dims, val, wall in (("k", "[0 2 -2 0 0 0 0]", k_in, "kqRWallFunction"),
                                  ("omega", "[0 0 -1 0 0 0 0]", om_in, "omegaWallFunction")):
        _w(f"{case_dir}/0/{name}", "volScalarField", name, f"""
dimensions      {dims};
internalField   uniform {val:.8g};
boundaryField
{{
    inlet           {{ type fixedValue; value uniform {val:.8g}; }}
    outlet          {{ type inletOutlet; inletValue uniform {val:.8g}; value uniform {val:.8g}; }}
    top             {{ type slip; }}
    ground          {{ type {wall}; value uniform {val:.8g}; }}
    car             {{ type {wall}; value uniform {val:.8g}; }}
    frontAndBack    {{ type empty; }}
}}
""")
    _w(f"{case_dir}/0/nut", "volScalarField", "nut", """
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet           { type calculated; value uniform 0; }
    outlet          { type calculated; value uniform 0; }
    top             { type calculated; value uniform 0; }
    ground          { type nutkWallFunction; value uniform 0; }
    car             { type nutkWallFunction; value uniform 0; }
    frontAndBack    { type empty; }
}
""")
    return a_ref


# ------------------------------------------------------------------ #
# 6. READING OPENFOAM ASCII OUTPUT
# ------------------------------------------------------------------ #
def _strip_header(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"FoamFile\s*\{.*?\}", "", text, count=1, flags=re.S)


def _read_list(text, pos):
    """Read an OpenFOAM list 'N ( ... )' starting at text[pos:]; returns (items, end)."""
    m = re.compile(r"\s*(\d+)\s*\(").match(text, pos)
    n = int(m.group(1))
    i, depth, start = m.end(), 1, m.end()
    while depth:
        ch = text[i]
        depth += (ch == "(") - (ch == ")")
        i += 1
    return n, text[start:i - 1], i


def read_points(path):
    text = _strip_header(open(path).read())
    n, body, _ = _read_list(text, 0)
    vals = np.array(re.findall(r"[-+0-9.eE]+", body), dtype=float).reshape(-1, 3)
    assert len(vals) == n, "point count mismatch"
    return vals


def read_faces(path):
    raw = open(path).read()
    text = _strip_header(raw)
    if "faceCompactList" in raw:
        n1, body1, end = _read_list(text, 0)
        n2, body2, _ = _read_list(text, end)
        offsets = np.array(body1.split(), dtype=int)
        flat = np.array(body2.split(), dtype=int)
        return [flat[offsets[i]:offsets[i + 1]] for i in range(n1 - 1)]
    n, body, _ = _read_list(text, 0)
    faces = [np.array(f.split(), dtype=int) for f in re.findall(r"\d+\(([^()]*)\)", body)]
    assert len(faces) == n, "face count mismatch"
    return faces


def read_boundary(path):
    text = _strip_header(open(path).read())
    out = {}
    for m in re.finditer(r"(\w+)\s*\{([^{}]*)\}", text):
        body = m.group(2)
        nf = re.search(r"nFaces\s+(\d+)", body)
        sf = re.search(r"startFace\s+(\d+)", body)
        if nf and sf:
            out[m.group(1)] = (int(sf.group(1)), int(nf.group(1)))
    return out


def read_patch_field(path, patch="car", n_expected=None):
    """Values of a field on one patch (scalar -> (n,), vector -> (n,3))."""
    text = _strip_header(open(path).read())
    b = text.index("boundaryField")
    m = re.compile(r"\b" + patch + r"\b\s*\{").search(text, b)
    seg_start = m.end()
    v = re.compile(r"value\s+").search(text, seg_start)
    rest = text[v.end():]
    if rest.startswith("uniform"):
        val = re.match(r"uniform\s+(\([^)]*\)|[-+0-9.eE]+)", rest).group(1)
        arr = np.array(re.findall(r"[-+0-9.eE]+", val), dtype=float)
        return np.tile(arr, (n_expected, 1)).squeeze()
    lm = re.match(r"nonuniform\s+List<(\w+)>", rest)
    kind = lm.group(1)
    n, body, _ = _read_list(rest, lm.end())
    vals = np.array(re.findall(r"[-+0-9.eE]+", body), dtype=float)
    return vals.reshape(n, 3) if kind == "vector" else vals.reshape(n)


def car_geometry(case_dir):
    """Face centres, area vectors (OpenFOAM: pointing out of the fluid, i.e. into the car)."""
    mesh = f"{case_dir}/constant/polyMesh"
    pts = read_points(f"{mesh}/points")
    faces = read_faces(f"{mesh}/faces")
    start, n = read_boundary(f"{mesh}/boundary")["car"]
    centres, sf = [], []
    for f in faces[start:start + n]:
        p = pts[f]
        centres.append(p.mean(axis=0))
        sf.append(0.5 * np.cross(p[2] - p[0], p[3] - p[1]) if len(f) == 4 else
                  0.5 * sum(np.cross(p[i], p[(i + 1) % len(f)]) for i in range(len(f))))
    return np.array(centres), np.array(sf), [faces[i] for i in range(start, start + n)], pts


def read_residuals(log_path):
    """Initial residuals of p and Ux in the LAST iteration. With non-orthogonal
    correctors p is solved twice per iteration; the first solve is the one to
    monitor (the corrector's residual is relative and always looks large)."""
    text = open(log_path).read()
    blocks = [b for b in re.split(r"\nTime = ", text) if "Solving for p" in b]
    if not blocks:
        return float("nan"), float("nan")
    last = blocks[-1]
    def first(field):
        m = re.search(r"Solving for " + field + r", Initial residual = ([-+0-9.eE]+)", last)
        return float(m.group(1)) if m else float("nan")
    return first("p"), first("Ux")


def read_force_history(case_dir):
    files = glob.glob(f"{case_dir}/postProcessing/forceCoeffs1/*/*.dat")
    for path in files:
        header, rows = None, []
        for line in open(path):
            if line.startswith("#"):
                cols = line.lstrip("#").split()
                if "Cd" in cols:
                    header = cols
            elif header and line.strip():
                vals = line.split()
                if len(vals) == len(header):
                    rows.append([float(v) for v in vals])
        if header and rows:
            arr = np.array(rows)
            return arr[:, header.index("Cd")], arr[:, header.index("Cl")] if "Cl" in header else None
    return None, None


# ------------------------------------------------------------------ #
# 7. ONE RUN: mesh, solve, extract, save
# ------------------------------------------------------------------ #
def openfoam_bashrc():
    found = sorted(glob.glob("/usr/lib/openfoam/openfoam*/etc/bashrc"))
    if not found:
        raise RuntimeError("OpenFOAM not found: run the install cell first")
    return found[-1]


def run_of(cmd, case_dir, log_name, bashrc):
    full = f"source {bashrc} >/dev/null 2>&1; cd '{case_dir}' && {cmd} > {log_name} 2>&1"
    return subprocess.run(["bash", "-c", full]).returncode


def to_contract(vec):
    """Case (x, y, z) -> contract (X, Y, Z) = (z, y, -x)."""
    v = np.atleast_2d(vec)
    return np.column_stack([v[:, 2], v[:, 1], -v[:, 0]])


def extract(case_dir, params, speed_ms):
    centres, sf, _, _ = car_geometry(case_dir)
    area = np.linalg.norm(sf, axis=1)
    normal = -sf / area[:, None]                      # outward from the car into the fluid
    tdir = os.path.join(case_dir, str(N_ITER))
    p = read_patch_field(f"{tdir}/pMean", "car", len(area)) * RHO          # kinematic -> Pa
    tau = read_patch_field(f"{tdir}/wallShearStressMean", "car", len(area)) * RHO
    # OpenFOAM's wallShearStress is (-Sf/|Sf|) & Reff: the stress the WALL puts on
    # the fluid. The force per area the fluid puts ON the car is its negative
    # (checked against forceCoeffs' viscous drag in the trial runs).
    tau = -tau
    q = 0.5 * RHO * speed_ms ** 2
    a_ref = frontal_area(params)
    drag, lift = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    f_p = (-p[:, None] * normal * area[:, None]).sum(axis=0)
    f_f = (tau * area[:, None]).sum(axis=0)
    out = {"Cd_p": f_p @ drag / (q * a_ref), "Cd_f": f_f @ drag / (q * a_ref),
           "Cl_p": f_p @ lift / (q * a_ref), "Cl_f": f_f @ lift / (q * a_ref), "A_frontal": a_ref, "q_ref": q}
    out["Cd_mean"] = out["Cd_p"] + out["Cd_f"]
    out["Cl"] = out["Cl_p"] + out["Cl_f"]
    arrays = {"centre": to_contract(centres).astype(np.float32), "area": area.astype(np.float32),
              "normal": to_contract(normal).astype(np.float32), "p": p.astype(np.float32),
              "tau": to_contract(tau).astype(np.float32)}
    try:
        yp = read_patch_field(f"{tdir}/yPlus", "car", len(area))
        out["y_plus_mean"], out["y_plus_max"] = float(np.mean(yp)), float(np.max(yp))
    except Exception:
        out["y_plus_mean"] = out["y_plus_max"] = float("nan")
    return out, arrays


def run_design(design, speed_kmh, out_dir, work_dir, bashrc, keep_case=False):
    """Mesh, solve and save one design at one speed. Returns the run-table row."""
    did = design["design_id"]
    params = {p: float(design[p]) for p in PARAMS}
    speed_ms = SPEEDS_MS[speed_kmh]
    case = os.path.join(work_dir, f"{did}_{speed_kmh}kmh")
    row = {"design_id": did, "speed_kmh": speed_kmh, "speed_ms": speed_ms, **params,
           "morph_passed": False, "checkMesh_passed": False, "converged": False}
    try:
        write_case(case, params, speed_ms)
        if run_of("blockMesh", case, "log.blockMesh", bashrc) != 0:
            row["error"] = "blockMesh failed: " + open(f"{case}/log.blockMesh").read()[-1500:]
            return row
        row["morph_passed"] = True
        run_of("checkMesh", case, "log.checkMesh", bashrc)
        cm = open(f"{case}/log.checkMesh").read()
        row["checkMesh_passed"] = "Mesh OK" in cm
        m = re.search(r"cells:\s+(\d+)", cm)
        row["cell_count"] = int(m.group(1)) if m else None
        if run_of("simpleFoam", case, "log.simpleFoam", bashrc) != 0:
            row["error"] = "simpleFoam failed: " + open(f"{case}/log.simpleFoam").read()[-1500:]
            return row
        res, arrays = extract(case, params, speed_ms)
        row.update(res)
        cd_hist, _ = read_force_history(case)
        if cd_hist is not None and len(cd_hist) >= AVG_WINDOW:
            window = cd_hist[-AVG_WINDOW:]
            row["Cd"] = float(cd_hist[-1])
            row["Cd_std"] = float(np.std(window))
            row["Cd_mean_forceCoeffs"] = float(np.mean(window))
        else:
            row["Cd"], row["Cd_std"], row["Cd_mean_forceCoeffs"] = row["Cd_mean"], float("nan"), float("nan")
        row["final_res_p"], row["final_res_U"] = read_residuals(f"{case}/log.simpleFoam")
        row["iterations"] = N_ITER
        # Steady RANS around a bluff 2D body: the pressure residual stalls (about 0.3 in the
        # trial) while the forces settle, so convergence is judged on the drag instead:
        # small wobble over the averaging window and no drift between its two halves.
        if cd_hist is not None and len(cd_hist) >= AVG_WINDOW:
            w = cd_hist[-AVG_WINDOW:]
            drift = abs(np.mean(w[AVG_WINDOW // 2:]) - np.mean(w[:AVG_WINDOW // 2])) / abs(np.mean(w))
            row["Cd_drift"] = float(drift)
            row["converged"] = bool(np.std(w) / abs(np.mean(w)) < 0.01 and drift < 0.005)
        else:
            row["converged"] = False
        os.makedirs(os.path.join(out_dir, "faces"), exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "faces", f"{CATEGORY}_{did}_{speed_kmh}kmh.npz"), **arrays)
    except Exception as exc:  # keep going; the row records what failed
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if not keep_case and os.path.exists(case):
            shutil.rmtree(case, ignore_errors=True)
    return row


def write_case_json(out_dir):
    n_faces = N_NOSE + N_HW + N_ROOF + N_BT + N_REAR + N_BOT
    info = {"category": CATEGORY, "source": "2D OpenFOAM stand-in (Component 3), NOT Member 2's data",
            "solver": "simpleFoam, steady RANS, kOmegaSST, wall functions",
            "convergence": "drag-based: std over the averaging window < 1% of Cd and drift between its halves < 0.5% (pressure residual stalls near 0.3)", "full_car": False, "dimension": "2D",
            "span_m": SPAN, "rho_kg_m3": RHO, "mu_pa_s": MU, "speeds_ms": {str(k): v for k, v in SPEEDS_MS.items()},
            "axes": "Y up, Z length (nose toward +Z), X width", "flow_direction": [0, 0, -1],
            "normals": "outward from the car into the fluid", "p_ref_pa": 0.0,
            "tau_convention": "force per area exerted by the fluid on the car",
            "averaging": f"fieldAverage over the last {AVG_WINDOW} of {N_ITER} iterations", "n_faces": n_faces,
            "ground": "moving at the free-stream speed", "wheels": False}
    with open(os.path.join(out_dir, "case.json"), "w") as fh:
        json.dump(info, fh, indent=2)


# ------------------------------------------------------------------ #
# 8. BATCH RUNNER (resumable: every finished run is saved at once)
# ------------------------------------------------------------------ #
def _row_path(out_dir, did, kmh):
    return os.path.join(out_dir, "rows", f"{did}_{kmh}kmh.json")


def run_many(design_table, design_ids, out_dir, work_dir, workers=2, keep_case=False):
    """Run every (design, speed) not already saved. Prints one line per finished run."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    bashrc = openfoam_bashrc()
    os.makedirs(os.path.join(out_dir, "rows"), exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    todo = []
    for did in design_ids:
        d = design_table[design_table["design_id"] == did].iloc[0].to_dict()
        for kmh in SPEEDS_MS:
            if not os.path.exists(_row_path(out_dir, did, kmh)):
                todo.append((d, kmh))
    print(f"{len(todo)} runs to do ({len(design_ids) * len(SPEEDS_MS) - len(todo)} already saved)")
    t0 = time.time()

    def job(item):
        d, kmh = item
        row = run_design(d, kmh, out_dir, work_dir, bashrc, keep_case=keep_case)
        with open(_row_path(out_dir, d["design_id"], kmh), "w") as fh:
            json.dump(row, fh, indent=1, default=float)
        return row

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(job, it) for it in todo]):
            row = fut.result()
            done += 1
            status = "OK " if row.get("converged") else ("NOT CONVERGED" if "Cd_mean" in row else "FAILED")
            cd = f"Cd {row['Cd_mean']:.4f} (std {row.get('Cd_std', float('nan')):.4f})" if "Cd_mean" in row else row.get("error", "")[:300]
            print(f"[{done}/{len(todo)}  {(time.time() - t0) / 60:5.1f} min] {row['design_id']} {row['speed_kmh']} km/h  {status}  {cd}", flush=True)


def assemble_run_table(out_dir):
    """Collect every saved run into run_table.csv (Member 2's format)."""
    import pandas as pd
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, "rows", "*.json")))]
    df = pd.DataFrame(rows)
    first = ["design_id", "speed_kmh", "speed_ms"] + PARAMS + ["Cd", "Cd_mean", "Cd_std", "Cd_p", "Cd_f", "Cl", "Cl_p", "Cl_f",
             "A_frontal", "q_ref", "iterations", "converged", "y_plus_mean", "y_plus_max", "cell_count",
             "checkMesh_passed", "morph_passed", "final_res_p", "final_res_U"]
    cols = [c for c in first if c in df.columns] + [c for c in df.columns if c not in first]
    df = df[cols].sort_values(["design_id", "speed_kmh"])
    df.to_csv(os.path.join(out_dir, "run_table.csv"), index=False)
    return df
