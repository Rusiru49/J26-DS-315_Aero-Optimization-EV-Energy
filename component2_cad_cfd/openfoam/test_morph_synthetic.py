#!/usr/bin/env python3
"""Self-test of morph_case.py on a small synthetic structured mesh (no OpenFOAM needed).
   python3 test_morph_synthetic.py          (writes into ./morph_test_tmp)"""
import os, shutil, struct, subprocess, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import morph_case as mc

NX, NY, NZ, DX = 40, 40, 20, 0.25
X0, Y0 = 0.0, -5.0

def pid(i, j, k): return i + (NX + 1) * (j + (NY + 1) * k)

def build_mesh():
    xs = X0 + DX * np.arange(NX + 1); ys = Y0 + DX * np.arange(NY + 1); zs = DX * np.arange(NZ + 1)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")          # index order k,j,i  -> flat i fastest
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    faces = {"internal": [], "inlet": [], "outlet": [], "ground": [], "top": [], "sides": [], "car": []}
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                if i > 0: faces["internal"].append((pid(i, j, k), pid(i, j + 1, k), pid(i, j + 1, k + 1), pid(i, j, k + 1)))
                if j > 0: faces["internal"].append((pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j, k + 1), pid(i, j, k + 1)))
                if k > 0: faces["internal"].append((pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)))
    for k in range(NZ):
        for j in range(NY): faces["inlet"].append((pid(0, j, k), pid(0, j, k + 1), pid(0, j + 1, k + 1), pid(0, j + 1, k)))
        for j in range(NY): faces["outlet"].append((pid(NX, j, k), pid(NX, j + 1, k), pid(NX, j + 1, k + 1), pid(NX, j, k + 1)))
    for k in range(NZ):
        for i in range(NX):
            faces["sides"].append((pid(i, 0, k), pid(i + 1, 0, k), pid(i + 1, 0, k + 1), pid(i, 0, k + 1)))
            faces["sides"].append((pid(i, NY, k), pid(i, NY, k + 1), pid(i + 1, NY, k + 1), pid(i + 1, NY, k)))
    for j in range(NY):
        for i in range(NX):
            f = (pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0))
            x, y = X0 + DX * (i + .5), Y0 + DX * (j + .5)
            (faces["car"] if (3 < x < 7 and -2 < y < 2) else faces["ground"]).append(f)
            faces["top"].append((pid(i, j, NZ), pid(i + 1, j, NZ), pid(i + 1, j + 1, NZ), pid(i, j + 1, NZ)))
    order = ["internal", "inlet", "outlet", "ground", "top", "sides", "car"]
    allf = []; patches = []
    for n in order:
        if n != "internal": patches.append((n, "wall" if n in ("ground", "car") else "patch", len(faces[n]), len(allf)))
        allf += faces[n]
    return pts, np.array(allf), patches

HEAD = "FoamFile\n{\n    version     2.0;\n    format      %s;\n    arch        \"LSB;label=32;scalar=64\";\n    class       %s;\n    location    \"constant/polyMesh\";\n    object      %s;\n}\n\n// * * * * * * * * //\n\n"

def write_mesh(d, fmt):
    os.makedirs(d, exist_ok=True)
    pts, faces, patches = build_mesh()
    h = lambda cls, obj: (HEAD % (fmt, cls, obj)).encode()
    foot = b")\n\n\n// ******* //\n"
    if fmt == "binary":
        open(d + "/points", "wb").write(h("vectorField", "points") + b"%d\n(" % len(pts) + pts.astype("<f8").tobytes() + foot)
        offs = (4 * np.arange(len(faces) + 1)).astype("<i4"); lab = faces.ravel().astype("<i4")
        open(d + "/faces", "wb").write(h("faceCompactList", "faces") + b"%d\n(" % len(offs) + offs.tobytes() + b")\n\n%d\n(" % len(lab) + lab.tobytes() + foot)
    else:
        open(d + "/points", "wb").write(h("vectorField", "points") + ("%d\n(\n" % len(pts) + "\n".join("(%.10g %.10g %.10g)" % tuple(p) for p in pts) + "\n").encode() + foot)
        open(d + "/faces", "wb").write(h("faceList", "faces") + ("%d\n(\n" % len(faces) + "\n".join("4(%d %d %d %d)" % tuple(f) for f in faces) + "\n").encode() + foot)
    for n in ("owner", "neighbour"): open(d + "/" + n, "wb").write(h("labelList", n) + b"0\n(\n)\n")
    b = "6\n(\n" + "".join("    %s\n    {\n        type            %s;\n        inGroups        List<word> 1(%s);\n        nFaces          %d;\n        startFace       %d;\n    }\n" % (n, t, t, nf, sf) for n, t, nf, sf in patches) + ")\n"
    open(d + "/boundary", "w").write((HEAD % ("ascii", "polyBoundaryMesh", "boundary")) + b)
    return pts

def bump(x, y): return 0.4 * np.maximum(0, 1 - ((x - 5) ** 2 + y ** 2) / 4.0)

def write_stl(path, n, shape):
    """plate x[3,7] y[-2,2] in the OpenFOAM frame with n x n squares; height z = bump if shape=='bump' else 0. Written in CAD frame (mm)."""
    xs = np.linspace(3, 7, n + 1); ys = np.linspace(-2, 2, n + 1)
    def v(i, j):
        x, y = xs[i], ys[j]; z = bump(x, y) if shape == "bump" else 0.0
        return (-y * 1000.0, z * 1000.0, -x * 1000.0)        # CAD X=-y, Y=z, Z=-x
    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)
            tris += [(a, b, c), (a, c, d)]
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for t in tris: f.write(struct.pack("<12fH", 0, 0, 0, *[x for p in t for x in p], 0))

def jacobians_ok(pts):
    P = pts.reshape(NZ + 1, NY + 1, NX + 1, 3)
    e1 = P[:-1, :-1, 1:] - P[:-1, :-1, :-1]; e2 = P[:-1, 1:, :-1] - P[:-1, :-1, :-1]; e3 = P[1:, :-1, :-1] - P[:-1, :-1, :-1]
    return float(np.einsum("...i,...i->...", e1, np.cross(e2, e3)).min())

def run(args):
    r = subprocess.run([sys.executable, os.path.join(HERE, "morph_case.py")] + args, capture_output=True, text=True)
    return r

def main():
    tmp = os.path.join(os.getcwd(), "morph_test_tmp"); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    write_stl(tmp + "/base.stl", 8, "flat"); write_stl(tmp + "/tgt_same.stl", 8, "bump"); write_stl(tmp + "/tgt_other.stl", 11, "bump")
    ok_all = True
    for fmt in ("binary", "ascii"):
        base = tmp + "/base_" + fmt; pts0 = write_mesh(base + "/constant/polyMesh", fmt)
        r = run(["--base-case", base, "--inspect"]); print("--- inspect", fmt); print(r.stdout[-700:], r.stderr[-500:])
        assert r.returncode == 0 and "car patch 'car': 289 surface points" in r.stdout, "inspect failed"
        for tgt, expect in (("tgt_same.stl", "correspondence"), ("tgt_other.stl", "ffd")):
            case = tmp + "/new_%s_%s" % (fmt, expect); os.makedirs(case)
            r = run(["--base-case", base, "--base-stl", tmp + "/base.stl", "--target-stl", tmp + "/" + tgt, "--case", case,
                     "--ground-ramp", "0", "--radius", "2.0", "--np", "4", "--ffd-zlo", "-1.5", "--ffd-ztop", "2.0", "--ffd-margin", "1.2", "--ffd-h", "0.25", "--ffd-hz", "0.25", "--ffd-h2", "0.125", "--ffd-hz2", "0.125", "--ffd-lambda", "0.05", "--ffd-refine-tol", "0"])
            print("--- morph", fmt, expect); print(r.stdout, r.stderr[-800:])
            assert r.returncode == 0, "morph failed"
            rep = json.load(open(case + "/morph_report.json")); assert rep["morph_mode"] == expect, rep["morph_mode"]
            pts1, meta = mc.read_points(case + "/constant/polyMesh/points"); assert meta["fmt"] == fmt and len(pts1) == len(pts0)
            d = pts1 - pts0; mag = np.linalg.norm(d, axis=1)
            bnd = mc.read_boundary(case + "/constant/polyMesh/boundary"); offs, lab = mc.read_faces(case + "/constant/polyMesh/faces")
            fixed = np.unique(np.concatenate([lab[offs[p["startFace"]]:offs[p["startFace"] + p["nFaces"]]] for p in bnd if p["name"] != "car"]))
            checks = {"fixed patches unchanged": float(mag[fixed].max()) < 1e-9,
                      "max displacement <= 0.45 m": float(mag.max()) <= 0.45,
                      "centre moved ~0.4 m": abs(d[np.argmin(np.linalg.norm(pts0 - [5, 0, 0], axis=1)), 2] - 0.4) < (0.01 if expect == "correspondence" else 0.05),
                      "min cell Jacobian > 0": jacobians_ok(pts1) > 0,
                      "surface error p99 < 15 mm": rep["err_p99_mm"] < 15.0,
                      "Allrun written": os.path.exists(case + "/Allrun")}
            for k, v in checks.items(): print("   ", "PASS" if v else "FAIL", k)
            ok_all &= all(checks.values())
    print("\nALL TESTS PASSED" if ok_all else "\nSOME TESTS FAILED"); sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
