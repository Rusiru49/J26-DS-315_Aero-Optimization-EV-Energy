import csv, os, shutil, sys
import numpy as np
HOME = os.path.expanduser("~/Documents/EV_Aero")
OUT = HOME + "/for_component3_urban1"
PKG = HOME + "/for_component3"
DES = ["D000", "D019", "D002", "D042", "D047", "D049"]
SRC = {"D000": "npz/sedan_D000_40kmh.npz", "D019": PKG + "/npz/sedan_D019_40kmh.npz"}
for d in DES[2:]: SRC[d] = "npz_morph/sedan_%s_40kmh.npz" % d
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT + "/npz")
base = np.load("npz_morph/baseline_faces.npz")["centre"].astype(float)
ok = True
for d in DES:
    f = SRC[d]
    if not os.path.exists(f): print("MISSING", f); ok = False; continue
    z = np.load(f); n = len(z["area"]); msgs = []
    if n != 44279: msgs.append("faces %d" % n)
    for k in ("centre", "area", "normal", "p", "tau"):
        if not np.isfinite(z[k].astype(float)).all(): msgs.append("non-finite " + k)
    if (z["area"] <= 0).any(): msgs.append("area<=0")
    if np.abs(np.linalg.norm(z["normal"].astype(float), axis=1) - 1).max() > 1e-3: msgs.append("normals not unit")
    q = float(z["q_ref"]); A = float(z["Aref"])
    F = (-(z["p"].astype(float) * z["area"].astype(float))[:, None] * z["normal"].astype(float)).sum(0) + (z["tau"].astype(float) * z["area"].astype(float)[:, None]).sum(0)
    cd_sum = -F[2] / (q * A); cd = float(z["Cd"])
    dd = 100 * (cd_sum - cd) / cd
    extra = ""
    if d == "D000":
        extra = " centre vs baseline max %.3f mm" % (np.abs(z["centre"].astype(float) - base).max() * 1000)
        if np.abs(z["centre"].astype(float) - base).max() > 1e-3: msgs.append("D000 face order differs from baseline")
    if d != "D019" and abs(dd) > 1.0: msgs.append("Cd_sum differs %.2f%%" % dd)
    print("%s faces %d Cd(file) %.4f Cd(sum) %.4f (%.2f%%)%s  %s" % (d, n, cd, cd_sum, dd, extra, "FAIL: " + "; ".join(msgs) if msgs else "OK"))
    if msgs: ok = False
    shutil.copy(f, OUT + "/npz/sedan_%s_40kmh.npz" % d)
shutil.copy("npz_morph/baseline_faces.npz", OUT + "/npz/baseline_faces.npz")
rows = list(csv.DictReader(open("run_table.csv")))
sel = [r for r in rows if r.get("design_id") in DES and str(r.get("speed_kmh")) in ("40", "40.0")]
sel.sort(key=lambda r: DES.index(r["design_id"]))
with open(OUT + "/run_table_urban1.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(sel)
hw = {}
for r in rows:
    if str(r.get("speed_kmh")) in ("100", "100.0") and r.get("Cd_mean"): hw[r["design_id"]] = float(r["Cd_mean"])
try:
    for r in csv.DictReader(open(PKG + "/run_table.csv")):
        if str(r.get("speed_kmh")) in ("100", "100.0"):
            if r.get("Cd_remesh"): hw[r["design_id"]] = float(r["Cd_remesh"])
except Exception as e: print("note: no 100 km/h comparison", e)
tab = "| Design | Cd_mean 40 km/h (label) | Cd 100 km/h (label) | difference |\n|---|---|---|---|\n"
for r in sel:
    c = float(r["Cd_mean"]); h = hw.get(r["design_id"])
    tab += "| %s | %.4f | %s | %s |\n" % (r["design_id"], c, "%.4f" % h if h else "-", "%+.1f %%" % (100 * (c - h) / h) if h else "-")
txt = """# Urban batch 1 (40 km/h) - EV Aero-Optimizer, Component 2

Scenario: **urban = 40 km/h (11.11 m/s)**. The main package holds the **highway = 100 km/h (27.78 m/s)** data.
6 designs: D000 (baseline), D019, D002, D042, D047, D049.

## Setup (same as the 100 km/h data, only the inlet/ground speed changes)
OpenFOAM ESI v2606, simpleFoam, k-omega SST, full car, moving ground, rho 1.165 kg/m3, mu 1.86e-5, 1500 iterations,
coarse mesh (about 0.57 M cells), p in Pa in the NPZ files (OpenFOAM computes kinematic p; it is multiplied by rho).

## Files
- npz/sedan_<D>_40kmh.npz : per-face data on the same 44,279 baseline faces (centre, area, normal, p, tau, Cd, Aref, U, rho, q_ref, p_ref)
- npz/baseline_faces.npz : baseline face centres (face order reference)
- run_table_urban1.csv : one row per design with Cd_mean (mean of the last 100 iterations = the label), drift, convergence, notes

## Which Cd to use as the label
Use **Cd_mean** (last-100 mean). D000 and D019 are full re-mesh runs. D019: the NPZ holds the re-mesh field mapped onto the morph faces
(mapping Cd error about -2.8 %), and its Cd is the re-mesh Cd_mean. D002, D042, D047, D049 are FFD-morph runs from the D000 base mesh.

## Results
""" + tab + """
## Honest notes
- At 40 km/h the Cd is within about 1 % of the 100 km/h value for these designs (turbulent flow, high Reynolds number), so the urban set mainly confirms the trend and adds a speed input.
- The morph-versus-re-mesh validation (D003, D012, D038, D002, D049, D036, D019) was done at 100 km/h only. D002/D049 passed it (-2.3 %, +3.0 %); D042 and D047 were only mesh-quality checked, not re-meshed.
- D047 shows a large Cd oscillation (flagged at 100 and 40 km/h); Cd_mean is the last-100 mean.
- This is OpenFOAM (not ANSYS) on a coarse mesh; 40 km/h exists for these 6 designs only.
"""
open(OUT + "/README.md", "w").write(txt)
if not ok: print("\nSOME CHECKS FAILED - zip not made"); sys.exit(1)
os.chdir(HOME)
if os.path.exists("for_component3_urban1.zip"): os.remove("for_component3_urban1.zip")
shutil.make_archive("for_component3_urban1", "zip", HOME, "for_component3_urban1")
print("\nZIP:", HOME + "/for_component3_urban1.zip", os.path.getsize("for_component3_urban1.zip") // 1024, "KB")
