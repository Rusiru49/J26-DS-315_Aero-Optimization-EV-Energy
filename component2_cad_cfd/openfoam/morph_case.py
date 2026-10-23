#!/usr/bin/env python3
"""Mesh morphing for Component 2: every design uses the SAME mesh topology (the D000 base mesh),
only the point positions move so that the car surface takes the new design's shape.

  1. read the base polyMesh (points / faces / boundary) of the D000 case
  2. surface displacement of the car-patch points  (from the CAD STL files of base and target design)
        mode "correspondence": both STLs have the same triangles in the same order  -> exact displacement
        mode "projection"    : otherwise, project every base surface point onto the target STL
  3. spread that displacement into the volume (inverse-distance weighting, smooth taper to 0 at --radius,
     far boundaries and the ground are kept fixed)
  4. write constant/polyMesh into the new case folder (made before by make_case.py) and replace its Allrun
     by a version that skips snappyHexMesh and stops if the morphed mesh fails checkMesh.

needs: python3, numpy, scipy
usage:
  python3 morph_case.py --base-case cases_prod/D000_100kmh_coarse --base-stl sedan_D000.stl \
        --target-stl sedan_D019.stl --case cases_morph/D019_100kmh_coarse --np 4
  python3 morph_case.py --base-case cases_prod/D000_100kmh_coarse --inspect          # only look at the base mesh
  add --check-only to compute and report without writing any mesh
Frames: STL files are in the CAD frame (mm; X lateral, Y up, Z along the car);
        the mesh is in the OpenFOAM frame (m; x = -Z, y = -X, z = Y)  - same conversion as make_case.py.
"""
import argparse, json, os, re, shutil, struct, sys
import numpy as np
try:
    from scipy.spatial import cKDTree
except ImportError:
    sys.exit("scipy is missing:  pip3 install scipy numpy   (run it in a normal Terminal, not needed inside OpenFOAM)")


# ----------------------------------------------------------------------------------------- STL
def read_stl_np(path):
    """-> (n,3,3) float64 triangle vertices, CAD frame as stored in the file."""
    data = open(path, "rb").read()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        vs = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
        return np.array(vs).astype(float).reshape(-1, 3, 3)
    n = struct.unpack("<I", data[80:84])[0]
    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(float)


def cad_to_foam(T):
    return np.stack([-T[..., 2], -T[..., 0], T[..., 1]], axis=-1) / 1000.0


# ----------------------------------------------------------------------------------------- geometry
def closest_on_tri(P, A, B, C):
    """Closest point of triangles (A,B,C) to points P (all arrays (n,3)). -> Q, bary (n,3) with Q = u*A+v*B+w*C"""
    es = lambda x, y: np.einsum("ij,ij->i", x, y)
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = es(ab, ap), es(ac, ap)
    bp = P - B; d3, d4 = es(ab, bp), es(ac, bp)
    cp = P - C; d5, d6 = es(ab, cp), es(ac, cp)
    vc = d1 * d4 - d3 * d2; vb = d5 * d2 - d1 * d6; va = d3 * d6 - d5 * d4
    n = len(P); bary = np.zeros((n, 3)); done = np.zeros(n, bool)
    with np.errstate(all="ignore"):
        def put(mask, u, v, w):
            m = mask & ~done
            bary[m, 0] = np.broadcast_to(u, n)[m]; bary[m, 1] = np.broadcast_to(v, n)[m]; bary[m, 2] = np.broadcast_to(w, n)[m]
            done[m] = True
        put((d1 <= 0) & (d2 <= 0), 1.0, 0.0, 0.0)
        put((d3 >= 0) & (d4 <= d3), 0.0, 1.0, 0.0)
        t = d1 / (d1 - d3); put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - t, t, 0.0)
        put((d6 >= 0) & (d5 <= d6), 0.0, 0.0, 1.0)
        t = d2 / (d2 - d6); put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t, 0.0, t)
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), 0.0, 1 - t, t)
        den = 1.0 / (va + vb + vc); v = vb * den; w = vc * den; put(np.ones(n, bool), 1 - v - w, v, w)
    Q = bary[:, :1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
    return Q, bary


def refine(T, h, max_tris=3_000_000):
    """split triangles with an edge longer than h (4-way midpoint split); returns refined tris and parent index."""
    cur, parent = T, np.arange(len(T))
    for _ in range(10):
        e = np.stack([np.linalg.norm(cur[:, 1] - cur[:, 0], axis=1), np.linalg.norm(cur[:, 2] - cur[:, 1], axis=1),
                      np.linalg.norm(cur[:, 0] - cur[:, 2], axis=1)], 1).max(1)
        big = e > h
        if not big.any() or len(cur) > max_tris: break
        b, bp = cur[big], parent[big]
        a, bb, c = b[:, 0], b[:, 1], b[:, 2]; ab, bc, ca = (a + bb) / 2, (bb + c) / 2, (c + a) / 2
        new = np.concatenate([np.stack(x, 1) for x in ([a, ab, ca], [ab, bb, bc], [ca, bc, c], [ab, bc, ca])])
        cur = np.concatenate([cur[~big], new]); parent = np.concatenate([parent[~big], np.tile(bp, 4)])
    return cur, parent


class TriSurface:
    """closest-point queries on a triangle soup"""
    def __init__(self, T, h=0.02):
        self.T = T
        nn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); self.tn = nn / np.maximum(np.linalg.norm(nn, axis=1), 1e-30)[:, None]
        R, parent = refine(T, h)
        samp = np.concatenate([R.mean(1), R[:, 0], R[:, 1], R[:, 2]]); self.sp = np.tile(parent, 4)
        self.tree = cKDTree(samp)

    def closest(self, P, k=8, normals=None, min_dot=0.3):
        """closest point on the surface. If `normals` (n,3, outward) are given, triangles facing the other way
        (dot < min_dot) are strongly penalised: stops a point jumping to the back side of a thin part (spoiler) or to a panel facing elsewhere."""
        k = min(k if normals is None else max(k, 24), len(self.sp))
        _, idx = self.tree.query(P, k=k)
        idx = idx.reshape(len(P), -1); cand = self.sp[idx]
        best = np.full(len(P), np.inf); score = np.full(len(P), np.inf)
        bQ = np.zeros_like(P); btri = np.zeros(len(P), int); bb = np.zeros((len(P), 3))
        for j in range(cand.shape[1]):
            tid = cand[:, j]
            Q, bary = closest_on_tri(P, self.T[tid, 0], self.T[tid, 1], self.T[tid, 2])
            d = np.linalg.norm(P - Q, axis=1); sc = d.copy()
            if normals is not None: sc = sc + 10.0 * (np.einsum("ij,ij->i", self.tn[tid], normals) < min_dot)
            u = sc < score
            score[u] = sc[u]; best[u] = d[u]; bQ[u] = Q[u]; btri[u] = tid[u]; bb[u] = bary[u]
        return bQ, btri, bb, best


def coverage_stats(T, pts, n=40000, seed=0):
    """how well do `pts` (surface points) cover the surface T?  distance from random samples of T to the nearest point (mm)"""
    R, _ = refine(T, 0.03); smp = R.mean(1); rng = np.random.default_rng(seed)
    if len(smp) > n: smp = smp[rng.choice(len(smp), n, replace=False)]
    d, _ = cKDTree(pts).query(smp)
    return float(np.percentile(d, 99) * 1000), float(d.max() * 1000)


def jump_stats(P, d, k=8):
    """roughness of the displacement field: difference to the mean of the k nearest neighbours (mm)"""
    _, nb = cKDTree(P).query(P, k=k + 1); j = np.linalg.norm(d - d[nb[:, 1:]].mean(1), axis=1)
    return float(np.percentile(j, 99) * 1000), float(j.max() * 1000)


def register(P, S0, S1, T1, iters=20, smooth=0.5, k=10, nsamp=150000):
    """Chamfer-style registration of the base surface points P onto the target surface:
    forward pull (each point -> nearest target surface point, normal-aware start) + reverse pull (target samples -> nearest morphed point,
    so no part of the target is left uncovered) + neighbour smoothing (keeps the displacement field regular). Returns displacement d (n,3)."""
    _, tri0, _, _ = S0.closest(P); Q, _, _, _ = S1.closest(P, normals=S0.tn[tri0]); d = Q - P
    R1, _ = refine(T1, 0.03); Y = R1.mean(1)
    if len(Y) > nsamp: Y = Y[np.random.default_rng(0).choice(len(Y), nsamp, replace=False)]
    _, nbP = cKDTree(P).query(P, k=k + 1)
    for _ in range(iters):
        X = P + d
        Qf, _, _, _ = S1.closest(X); rf = Qf - X
        _, j = cKDTree(X).query(Y); cnt = np.bincount(j, minlength=len(P)); rr = np.zeros_like(P)
        for a in range(3): rr[:, a] = np.bincount(j, weights=Y[:, a] - X[j, a], minlength=len(P))
        rr /= np.maximum(cnt, 1)[:, None]
        d = d + 0.5 * rf + 0.5 * rr
        d = (1 - smooth) * d + smooth * d[nbP[:, 1:]].mean(1)
    X = P + d; Qf, _, _, _ = S1.closest(X)
    return d + (Qf - X)


def _face_normals(P, faces_by_n):
    """area-weighted normals (fan triangulation) for faces given as dict n_vertices -> (m,n) local point indices; returns list of (ids, normals)"""
    out = []
    for n, F in faces_by_n.items():
        N = np.zeros((len(F), 3))
        for i in range(1, n - 1): N += np.cross(P[F[:, i]] - P[F[:, 0]], P[F[:, i + 1]] - P[F[:, 0]])
        out.append(N)
    return out


def regularize(P, d, faces_by_n, gmax=0.5, k=8, iters=400, min_dot=0.5, verbose=True):
    """Make the surface displacement fold-free: (1) limit the gradient of d between neighbouring points (adaptive Laplacian smoothing only where
    the gradient exceeds gmax, so the fit stays untouched elsewhere); (2) smooth locally around every surface face whose normal turned by more than
    ~60 degrees (flipped / folded faces, exactly what checkMesh complains about). Returns d, number of folded faces before and after."""
    _, nb = cKDTree(P).query(P, k=k + 1); nb = nb[:, 1:]
    dx = np.maximum(np.linalg.norm(P[nb] - P[:, None], axis=2), 1e-6)
    grad = lambda dd: (np.linalg.norm(dd[nb] - dd[:, None], axis=2) / dx).max(1)
    def folded(dd):
        N0 = _face_normals(P, faces_by_n); N1 = _face_normals(P + dd, faces_by_n); res = []
        for F, a, b in zip(faces_by_n.values(), N0, N1):
            c = np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)
            res.append(F[c < min_dot])
        return res
    def nfold(dd): return int(sum(len(r) for r in folded(dd)))
    n_before = nfold(d); d = d.copy()
    for _ in range(iters):                                    # (1) gradient limiter
        bad = grad(d) > gmax
        if not bad.any(): break
        m = bad.copy(); m[nb[bad].ravel()] = True
        d[m] = 0.5 * d[m] + 0.5 * d[nb[m]].mean(1)
    for it in range(iters):                                   # (2) fold repair (gets stronger if folds survive)
        res = folded(d); cnt = sum(len(r) for r in res)
        if cnt == 0: break
        m = np.zeros(len(P), bool)
        for r in res: m[r.ravel()] = True
        for _r in range(2 + it // 50): m[nb[m].ravel()] = True
        al = min(0.5 + 0.1 * (it // 50), 0.9)
        d[m] = (1 - al) * d[m] + al * d[nb[m]].mean(1)
    n_after = nfold(d)
    if verbose: print("surface folds: %d folded faces before, %d after regularisation" % (n_before, n_after))
    return d, n_before, n_after


def check_correspondence(T0, T1, tol=5e-4):
    """are T0 and T1 the same triangles (same order) with moved vertices?"""
    if T0.shape != T1.shape:
        return False, "different triangle count (%d vs %d)" % (len(T0), len(T1)), {}
    D = (T1 - T0).reshape(-1, 3); V = T0.reshape(-1, 3)
    key = np.round(V / 1e-4).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel(); cnt = np.bincount(inv)
    mean = np.stack([np.bincount(inv, weights=D[:, c]) / cnt for c in range(3)], 1)
    spread = np.linalg.norm(D - mean[inv], axis=1); dmax = np.linalg.norm(D, axis=1).max()
    info = {"n_tris": int(len(T0)), "max_vertex_disp_mm": float(dmax * 1000), "max_vertex_spread_mm": float(spread.max() * 1000)}
    ok = spread.max() < tol and dmax < 0.5
    return ok, ("ok" if ok else "same count but vertices do not correspond (spread %.2f mm, max disp %.0f mm)" % (spread.max() * 1000, dmax * 1000)), info


# ----------------------------------------------------------------------------------------- OpenFOAM polyMesh I/O
def _skip(data, pos):
    n = len(data)
    while pos < n:
        c = data[pos:pos + 1]
        if c in (b" ", b"\t", b"\r", b"\n"): pos += 1
        elif data[pos:pos + 2] == b"//":
            e = data.find(b"\n", pos); pos = n if e < 0 else e
        elif data[pos:pos + 2] == b"/*":
            e = data.find(b"*/", pos); pos = n if e < 0 else e + 2
        else: break
    return pos


def foam_header(data):
    i = data.find(b"FoamFile"); j = data.find(b"{", i); k = data.find(b"}", j)
    if i < 0 or j < 0 or k < 0: raise ValueError("no FoamFile header")
    h = data[j:k].decode("latin1")
    fmt = re.search(r"format\s+(\w+)", h).group(1); cls = re.search(r"class\s+(\w+)", h).group(1)
    label, scalar = 32, 64
    m = re.search(r'arch\s+"([^"]*)"', h)
    if m:
        a = re.search(r"label=(\d+)", m.group(1)); b = re.search(r"scalar=(\d+)", m.group(1))
        label = int(a.group(1)) if a else 32; scalar = int(b.group(1)) if b else 64
    return fmt, cls, label, scalar, k + 1


def _list_start(data, pos):
    pos = _skip(data, pos); m = re.compile(rb"(\d+)").match(data, pos)
    if not m: raise ValueError("list count not found at byte %d" % pos)
    n = int(m.group(1)); p2 = _skip(data, m.end())
    if data[p2:p2 + 1] != b"(": raise ValueError("'(' expected after count at byte %d" % p2)
    return n, pos, p2 + 1          # count, position where the count starts, first payload byte


def read_points(path):
    data = open(path, "rb").read(); fmt, cls, lab, sca, hend = foam_header(data)
    n, cpos, ps = _list_start(data, hend)
    if fmt == "binary":
        dt = "<f8" if sca == 64 else "<f4"
        pts = np.frombuffer(data, dtype=dt, count=3 * n, offset=ps).astype(float).reshape(n, 3)
    else:
        end = data.rfind(b")")
        pts = np.array(data[ps:end].replace(b"(", b" ").replace(b")", b" ").split(), dtype=float).reshape(-1, 3)
        if len(pts) != n: raise ValueError("points: expected %d got %d" % (n, len(pts)))
    return pts, dict(fmt=fmt, lab=lab, sca=sca, head=data[:cpos])


def write_points(path, pts, meta):
    foot = b")\n\n\n// ************************************************************************* //\n"
    if meta["fmt"] == "binary":
        payload = pts.astype("<f8" if meta["sca"] == 64 else "<f4").tobytes()
    else:
        payload = ("\n".join("(%.12g %.12g %.12g)" % tuple(p) for p in pts) + "\n").encode()
    open(path, "wb").write(meta["head"] + b"%d\n(" % len(pts) + (b"\n" if meta["fmt"] != "binary" else b"") + payload + foot)


def read_faces(path):
    """-> offsets (nFaces+1), labels (flat)"""
    data = open(path, "rb").read(); fmt, cls, lab, sca, hend = foam_header(data)
    ldt = "<i4" if lab == 32 else "<i8"
    if cls == "faceCompactList":
        out = []; pos = hend
        for _ in range(2):
            n, cpos, ps = _list_start(data, pos)
            if fmt == "binary":
                arr = np.frombuffer(data, dtype=ldt, count=n, offset=ps).astype(np.int64); pos = ps + n * (4 if lab == 32 else 8) + 1
            else:
                e = data.find(b")", ps); arr = np.array(data[ps:e].split(), dtype=np.int64); pos = e + 1
                if len(arr) != n: raise ValueError("faces: list size mismatch")
            out.append(arr)
        return out[0], out[1]
    if cls == "faceList" and fmt == "ascii":
        n, cpos, ps = _list_start(data, hend); end = data.rfind(b")")
        tok = np.array(data[ps:end].replace(b"(", b" ").replace(b")", b" ").split(), dtype=np.int64)
        offs = [0]; labels = []; i = 0
        while i < len(tok):
            s = int(tok[i]); labels.append(tok[i + 1:i + 1 + s]); i += 1 + s; offs.append(offs[-1] + s)
        if len(offs) - 1 != n: raise ValueError("faces: expected %d got %d" % (n, len(offs) - 1))
        return np.array(offs, dtype=np.int64), np.concatenate(labels)
    raise ValueError("unsupported faces file (class %s, format %s). Convert the mesh to ascii:  foamFormatConvert" % (cls, fmt))


def read_boundary(path):
    txt = open(path, errors="ignore").read(); txt = txt[txt.find("}") + 1:]      # skip FoamFile block
    out = []
    for m in re.finditer(r"(\S+)\s*\{([^{}]*)\}", txt):
        body = m.group(2)
        t = re.search(r"\btype\s+(\w+)", body); nf = re.search(r"\bnFaces\s+(\d+)", body); sf = re.search(r"\bstartFace\s+(\d+)", body)
        if t and nf and sf: out.append(dict(name=m.group(1), type=t.group(1), nFaces=int(nf.group(1)), startFace=int(sf.group(1))))
    return out


# ----------------------------------------------------------------------------------------- morphing
def propagate(pts, car_idx, d_car, fixed_idx, symm_idx, radius, z0, k=16, power=3.0, chunk=100000):
    """displacement of every mesh point from the displacement of the car-surface points"""
    tree = cKDTree(pts[car_idx]); n = len(pts); d = np.zeros((n, 3)); k = min(k, len(car_idx)); soft = 2e-3
    for s in range(0, n, chunk):
        P = pts[s:s + chunk]
        try: dist, nb = tree.query(P, k=k, workers=-1)
        except TypeError: dist, nb = tree.query(P, k=k)
        dist = dist.reshape(len(P), -1); nb = nb.reshape(len(P), -1)
        w = 1.0 / (dist ** 2 + soft ** 2) ** (power / 2.0)
        dd = (w[:, :, None] * d_car[nb]).sum(1) / w.sum(1)[:, None]
        t = np.clip(dist[:, 0] / radius, 0, 1); dd *= (0.5 * (1 + np.cos(np.pi * t)))[:, None]
        if z0 > 0:
            q = np.clip(P[:, 2] / z0, 0, 1); dd *= (q * q * (3 - 2 * q))[:, None]
        d[s:s + chunk] = dd
    d[car_idx] = d_car
    d[fixed_idx] = 0.0
    d[symm_idx, 1] = 0.0
    return d


ALLRUN = """#!/bin/bash
# MORPHED-mesh case (made by morph_case.py).  run inside an OpenFOAM terminal.   usage: ./Allrun [NP]
cd "$(dirname "$0")"
NP=${1:-@NP@}
set -e
for t in checkMesh simpleFoam foamToVTK; do
    command -v $t >/dev/null 2>&1 || { echo "MISSING TOOL: $t  (tell Claude)"; exit 1; }
done
echo "step 1/6 features skipped (morphed mesh)"
echo "step 2/6 blockMesh skipped (morphed mesh)"
echo "step 3/6 morph done (mesh from base case)"
checkMesh -allGeometry -allTopology > log.checkMesh 2>&1 || true
BAD=$(grep '\\*\\*\\*' log.checkMesh | grep -v -i -e oncave -e determinant -e 'face pyramids' -e 'face tets' -e 'interpolation weight' | wc -l | tr -d ' ')
LW=$(grep 'small interpolation weight' log.checkMesh | head -1 | awk '{print $NF}'); LW=${LW:-0}
if [ "$LW" -gt 50 ]; then BAD=$((BAD+1)); fi
if [ "$LW" -gt 0 ]; then echo "NOTE: tolerated $LW faces with low interpolation weight (limit 50)"; fi
PYR=$(grep 'face pyramids' log.checkMesh | head -1 | awk '{for(i=1;i<=NF;i++) if($i=="pyramids:") print $(i+1)}'); PYR=${PYR:-0}
TET=$(grep 'face tets' log.checkMesh | head -1 | awk '{for(i=1;i<=NF;i++) if($i=="tets:") print $(i+1)}'); TET=${TET:-0}
# a handful of bad faces among 1.8 M is tolerated (flagged in run_table: checkMesh_passed=False); more than that stops the run
if [ "$PYR" -gt 20 ] || [ "$TET" -gt 500 ]; then BAD=$((BAD+1)); fi
if [ "$PYR" -gt 0 ] || [ "$TET" -gt 0 ]; then echo "NOTE: tolerated mesh defects: $PYR badly oriented face pyramids, $TET low-quality face tets (limits 20 / 500)"; fi
if [ "$BAD" -gt 0 ]; then
    echo "MORPHED MESH FAILED checkMesh ($BAD hard failures) - see log.checkMesh"; grep '\\*\\*\\*' log.checkMesh; exit 3
fi
echo "step 4/6 checkMesh done"
if [ "$NP" -gt 1 ] && command -v mpirun >/dev/null 2>&1; then
    decomposePar -force            > log.decomposePar 2>&1
    mpirun -np $NP simpleFoam -parallel > log.simpleFoam 2>&1 || echo "NOTE: mpirun returned non-zero; checking whether the run finished"
    ENDT=$(grep -m1 -E "^endTime" system/controlDict | awk '{print $2}' | tr -d ";")
    if [ ! -d "processor0/$ENDT" ]; then echo "SOLVER FAILED: processor0/$ENDT missing - see log.simpleFoam"; exit 1; fi
    reconstructPar -latestTime     > log.reconstructPar 2>&1
    rm -rf processor*
else
    simpleFoam                     > log.simpleFoam 2>&1
fi
echo "step 5/6 solver done"
PATCH=$(awk '/^ *car[A-Za-z0-9_]* *$/ {print $1; exit}' constant/polyMesh/boundary)
echo "$PATCH" > car_patch_name.txt
foamToVTK -ascii -latestTime -patches "($PATCH)" > log.foamToVTK 2>&1
echo "step 6/6 export done"
echo DONE
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-case", required=True, help="finished D000 case folder that still has constant/polyMesh")
    ap.add_argument("--base-stl"); ap.add_argument("--target-stl")
    ap.add_argument("--case", help="new case folder made by make_case.py for the target design")
    ap.add_argument("--np", type=int, default=4); ap.add_argument("--patch", default=None, help="car patch name (default: first patch starting with 'car')")
    ap.add_argument("--radius", type=float, default=1.2, help="distance (m) over which the surface motion fades to zero")
    ap.add_argument("--ground-ramp", type=float, default=0.09, help="height (m) over which the motion fades to zero towards the ground (0 = off)")
    ap.add_argument("--ffd-h", type=float, default=0.15, help="FFD box control-point spacing along x and y (m)")
    ap.add_argument("--ffd-hz", type=float, default=0.10, help="FFD box control-point spacing along z (m)")
    ap.add_argument("--ffd-zlo", type=float, default=0.0, help="lower face of the FFD box (m); 0 = the ground plane (kept fixed)")
    ap.add_argument("--ffd-ztop", type=float, default=0.45, help="box margin above the car roof (m)")
    ap.add_argument("--ffd-margin", type=float, default=0.45, help="box margin around the car in x and y (m)")
    ap.add_argument("--ffd-h2", type=float, default=0.08, help="stage-2 (fine) box spacing in x and y (m)")
    ap.add_argument("--ffd-hz2", type=float, default=0.06, help="stage-2 (fine) box spacing in z (m)")
    ap.add_argument("--ffd-lambda2", type=float, default=0.5, help="stage-2 smoothness weight")
    ap.add_argument("--ffd-refine-tol", type=float, default=3.0, help="stage 3: if after stage 2 some surface points are still more than this many mm from the target, add a local fine box over that x-range (0 = off)")
    ap.add_argument("--ffd-h3", type=float, default=0.04, help="stage-3 (local, finest) box spacing (m)"); ap.add_argument("--ffd-lambda3", type=float, default=0.3)
    ap.add_argument("--ffd-iters", type=int, default=14); ap.add_argument("--ffd-lambda", type=float, default=3.0, help="stage-1 smoothness weight (large = very smooth, fold-free)")
    ap.add_argument("--mode", default="auto", choices=["auto", "correspondence", "ffd", "projection"])
    ap.add_argument("--inspect", action="store_true"); ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    pm = os.path.join(a.base_case, "constant", "polyMesh")
    for f in ("points", "faces", "boundary", "owner", "neighbour"):
        if not os.path.exists(os.path.join(pm, f)):
            sys.exit("MISSING %s/%s  (the base case was cleaned? re-run D000 without --clean)" % (pm, f))
    pts, meta = read_points(os.path.join(pm, "points"))
    offs, labels = read_faces(os.path.join(pm, "faces")); bnd = read_boundary(os.path.join(pm, "boundary"))
    print("base mesh: %d points, %d faces, points format %s (label %d bit, scalar %d bit)" % (len(pts), len(offs) - 1, meta["fmt"], meta["lab"], meta["sca"]))
    for p in bnd: print("   patch %-10s type %-14s nFaces %-8d startFace %d" % (p["name"], p["type"], p["nFaces"], p["startFace"]))
    name = a.patch or next((p["name"] for p in bnd if p["name"].startswith("car")), None)
    if name is None: sys.exit("no car patch found in boundary file; give --patch NAME")
    def patch_points(p): return np.unique(labels[offs[p["startFace"]]:offs[p["startFace"] + p["nFaces"]]])
    carp = next(p for p in bnd if p["name"] == name); car_idx = patch_points(carp)
    fixed = [patch_points(p) for p in bnd if p["name"] != name and p["type"] not in ("symmetryPlane", "empty")]
    fixed_idx = np.unique(np.concatenate(fixed)) if fixed else np.array([], int)
    symm = [patch_points(p) for p in bnd if p["type"] == "symmetryPlane"]
    symm_idx = np.unique(np.concatenate(symm)) if symm else np.array([], int)
    P = pts[car_idx]
    print("car patch '%s': %d surface points; bbox x[%.3f, %.3f] y[%.3f, %.3f] z[%.3f, %.3f]" % ((name, len(car_idx)) + tuple(np.ravel([P.min(0), P.max(0)], order="F"))))
    if a.inspect and not (a.base_stl and a.target_stl): return
    if not (a.base_stl and a.target_stl and (a.case or a.check_only)): sys.exit("need --base-stl --target-stl --case (or --inspect, or --check-only)")

    T0 = cad_to_foam(read_stl_np(a.base_stl)); T1 = cad_to_foam(read_stl_np(a.target_stl))
    S0 = TriSurface(T0); S1 = TriSurface(T1)
    _, _, _, dist0 = S0.closest(P)
    print("base mesh surface vs base STL: mean %.2f mm, max %.2f mm  (should be ~1 mm)" % (dist0.mean() * 1000, dist0.max() * 1000))
    rep = {"base_fit_mean_mm": float(dist0.mean() * 1000), "base_fit_max_mm": float(dist0.max() * 1000), "radius_m": a.radius, "ground_ramp_m": a.ground_ramp}
    ok, why, info = check_correspondence(T0, T1)
    mode = a.mode if a.mode != "auto" else ("correspondence" if ok else "ffd")
    if mode == "correspondence" and not ok: sys.exit("STL files do not correspond: " + why)
    print("STL correspondence: %s  -> mode %s" % (why, mode)); rep.update(info)
    if mode == "correspondence":
        _, tri, bary, _ = S0.closest(P)
        D = T1 - T0; d_car = bary[:, :1] * D[tri, 0] + bary[:, 1:2] * D[tri, 1] + bary[:, 2:3] * D[tri, 2]
    elif mode == "ffd":
        import ffd
        lo = [P[:, 0].min() - a.ffd_margin, -(np.abs(P[:, 1]).max() + a.ffd_margin), a.ffd_zlo]; hi = [P[:, 0].max() + a.ffd_margin, np.abs(P[:, 1]).max() + a.ffd_margin, P[:, 2].max() + a.ffd_ztop]
        lat1 = ffd.Lattice(lo, hi, [a.ffd_h, a.ffd_h, a.ffd_hz]); lat2 = ffd.Lattice(lo, hi, [a.ffd_h2, a.ffd_h2, a.ffd_hz2])
        print("FFD morphing box, stage 1 (coarse, very smooth): %d x %d x %d cells, %d free control points" % (*lat1.n_cells, lat1.nfree))
        print("FFD morphing box, stage 2 (fine, removes the small remaining error): %d x %d x %d cells, %d free control points" % (*lat2.n_cells, lat2.nfree))
        H = ffd.fit_hier(P, S0, S1, T1, refine, lat1, lat2, lam1=a.ffd_lambda, lam2=a.ffd_lambda2, iters1=a.ffd_iters, verbose=False)
        if a.ffd_refine_tol > 0:
            X2 = P + H.displacement(P); _, _, _, e2 = S1.closest(X2); bad = e2 * 1e3 > a.ffd_refine_tol
            print("FFD after stage 2: %d surface points (%.1f %%) still more than %.1f mm from the target" % (bad.sum(), 100 * bad.mean(), a.ffd_refine_tol))
            if bad.sum() >= 20:
                x0 = max(lo[0], X2[bad, 0].min() - 0.25); x1 = min(hi[0], X2[bad, 0].max() + 0.25)
                lat3 = ffd.Lattice([x0, lo[1], lo[2]], [x1, hi[1], hi[2]], [a.ffd_h3] * 3)
                print("FFD morphing box, stage 3 (local fine, x = %.2f .. %.2f m): %d x %d x %d cells, %d free control points" % (x0, x1, *lat3.n_cells, lat3.nfree))
                dP3 = ffd.fit_ffd_close(X2, S1, T1, refine, lat3, iters=8, lam=a.ffd_lambda3)
                H = ffd.HierFFD(H.stages + [(lat3, dP3)]); rep.update(ffd_cells_stage3=[int(c) for c in lat3.n_cells], ffd_stage3_x=[float(x0), float(x1)])
        d_all = H.displacement(pts); d_car = d_all[car_idx]
        inbox = np.all((pts >= lat1.lo) & (pts < lat1.hi), axis=1); jd = H.jacobian_det(pts[inbox][::max(1, int(inbox.sum() // 150000))])
        rep.update(ffd_cells_stage1=[int(c) for c in lat1.n_cells], ffd_cells_stage2=[int(c) for c in lat2.n_cells], ffd_min_jacobian=float(jd.min()), ffd_pts_in_box=int(inbox.sum()))
        print("FFD smoothness: min Jacobian %.2f over %d mesh points in the box (must stay > 0; > 0.2 is comfortable)" % (jd.min(), inbox.sum()))
    else:
        d_car = register(P, S0, S1, T1)
    # surface folds, counted on the real car faces of the base mesh
    lut = np.full(len(pts), -1, np.int64); lut[car_idx] = np.arange(len(car_idx)); fb = {}
    for fi in range(carp["startFace"], carp["startFace"] + carp["nFaces"]): fb.setdefault(int(offs[fi + 1] - offs[fi]), []).append(labels[offs[fi]:offs[fi + 1]])
    fb = {n: lut[np.array(v)] for n, v in fb.items()}
    d_car, nf0, nf1 = regularize(P, d_car, fb, iters=0 if mode == "ffd" else 400)
    rep.update(folded_faces_before=nf0, folded_faces_after=nf1)
    _, _, _, err = S1.closest(P + d_car)
    mag = np.linalg.norm(d_car, axis=1)
    rep.update(morph_mode=mode, n_car_points=int(len(car_idx)), max_disp_mm=float(mag.max() * 1000), mean_disp_mm=float(mag.mean() * 1000),
               err_mean_mm=float(err.mean() * 1000), err_p99_mm=float(np.percentile(err, 99) * 1000), err_max_mm=float(err.max() * 1000))
    ref99, refmax = coverage_stats(T0, P); c99, cmax = coverage_stats(T1, P + d_car); j99, jmax = jump_stats(P, d_car)
    rep.update(cover_ref_p99_mm=ref99, cover_p99_mm=c99, cover_max_mm=cmax, jump_p99_mm=j99, jump_max_mm=jmax)
    print("surface point coverage of the target (99%%): reference %.0f mm, morphed %.0f mm (max %.0f mm); displacement roughness 99%% %.0f mm, max %.0f mm" % (ref99, c99, cmax, j99, jmax))
    rep["morph_surface_ok"] = bool(rep["err_p99_mm"] < 10.0 and c99 < 1.75 * ref99 + 5.0)
    rep["large_displacement"] = bool(rep["max_disp_mm"] > 150.0)
    print("surface displacement: mean %.1f mm, max %.1f mm" % (rep["mean_disp_mm"], rep["max_disp_mm"]))
    print("morphed surface vs target STL: mean %.2f mm, 99%% %.2f mm, max %.2f mm  -> %s" %
          (rep["err_mean_mm"], rep["err_p99_mm"], rep["err_max_mm"], "OK" if rep["morph_surface_ok"] else "WARNING: surface does not match the target well"))
    if rep["large_displacement"]: print("WARNING: surface moves up to %.0f mm - the morphed mesh may be badly distorted (checkMesh will tell)" % rep["max_disp_mm"])
    if mode == "ffd":
        d = d_all; d[fixed_idx] = 0.0; d[symm_idx, 1] = 0.0
    else:
        d = propagate(pts, car_idx, d_car, fixed_idx, symm_idx, a.radius, a.ground_ramp)
    rep["moved_points"] = int((np.linalg.norm(d, axis=1) > 1e-9).sum()); rep["total_points"] = int(len(pts))
    print("volume: %d of %d mesh points moved" % (rep["moved_points"], len(pts)))
    if a.check_only: print(json.dumps(rep, indent=1)); return

    os.makedirs(os.path.join(a.case, "constant", "polyMesh"), exist_ok=True)
    out = os.path.join(a.case, "constant", "polyMesh")
    for f in ("faces", "owner", "neighbour", "boundary", "cellZones", "faceZones", "pointZones"):
        if os.path.exists(os.path.join(pm, f)): shutil.copy2(os.path.join(pm, f), os.path.join(out, f))
    write_points(os.path.join(out, "points"), pts + d, meta)
    open(os.path.join(a.case, "Allrun"), "w").write(ALLRUN.replace("@NP@", str(a.np))); os.chmod(os.path.join(a.case, "Allrun"), 0o755)
    json.dump(rep, open(os.path.join(a.case, "morph_report.json"), "w"), indent=1)
    print("wrote morphed mesh to", out, "and a new Allrun (no snappyHexMesh)")


if __name__ == "__main__":
    main()
