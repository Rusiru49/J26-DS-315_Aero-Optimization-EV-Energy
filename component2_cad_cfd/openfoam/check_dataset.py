"""Checks that what Component 2 hands to Component 3 is complete and consistent.   python3 check_dataset.py
Looks at run_table.csv, npz_morph/*.npz and design_table_sedan_v5.csv and prints a checklist."""
import csv, glob, os, sys
import numpy as np
npzdir = sys.argv[1] if len(sys.argv) > 1 else "npz_morph"
rt = list(csv.DictReader(open("run_table.csv"))) if os.path.exists("run_table.csv") else []
design = [r["design_id"] for r in csv.DictReader(open("design_table_sedan_v5.csv"))]
ok = lambda b: "OK  " if b else "MISSING"
print("=== 1. run_table.csv (one row per design and speed)")
cols = ["design_id","speed_kmh","windshield_angle_deg","hood_angle_deg","spoiler_angle_deg","boat_tail_angle_deg","ground_clearance_mm",
        "Cd","Cl","A_frontal","converged","Cd_std","Cd_drift_pct","y_plus_mean","cell_count","checkMesh_passed","morph_passed"]
have = set(rt[0].keys()) if rt else set()
for c in cols: print("  %-22s %s" % (c, ok(c in have)))
print("  rows: %d   designs with a 100 km/h row: %d / %d   with a 40 km/h row: %d" % (len(rt), len({r["design_id"] for r in rt if r["speed_kmh"] == "100"}), len(design), len({r["design_id"] for r in rt if r["speed_kmh"] == "40"})))
print("\n=== 2. per-face files  %s/sedan_Dxxx_100kmh.npz" % npzdir)
files = sorted(f for f in glob.glob(npzdir + "/sedan_D*_100kmh.npz"))
nface = None; bad = []
for f in files:
    d = np.load(f); n = len(d["area"]); nface = nface or n
    probs = []
    for k in ("centre", "area", "normal", "p", "tau"):
        if k not in d.files: probs.append("no " + k)
        elif not np.isfinite(d[k]).all(): probs.append("NaN in " + k)
    if n != nface: probs.append("faces %d != %d" % (n, nface))
    if "area" in d.files and (d["area"] <= 0).any(): probs.append("area<=0")
    if probs: bad.append((os.path.basename(f), probs))
print("  files: %d   faces per file: %s   baseline_faces.npz: %s" % (len(files), nface, ok(os.path.exists(npzdir + "/baseline_faces.npz"))))
print("  problems: %s" % ("none" if not bad else ""))
for b in bad: print("   ", b[0], "; ".join(b[1]))
print("\n=== 3. promised in the proposal (Component 2 'enriched dataset')")
print("  Cd highway (100 km/h)         %s  (%d/%d designs)" % (ok(len(files) > 0), len(files), len(design)))
print("  Cd city (40 km/h)             %s" % ok(any(r["speed_kmh"] == "40" for r in rt)))
print("  Cl                            %s" % ok("Cl" in have))
print("  frontal area                  %s" % ok("A_frontal" in have))
print("  convergence flag              %s" % ok("converged" in have))
print("  per-run uncertainty (Cd_std)  %s" % ok("Cd_std" in have))
print("  finite-difference gradients   MISSING (needs extra perturbed designs, see plan)")
miss = [d for d in design if not os.path.exists("%s/sedan_%s_100kmh.npz" % (npzdir, d))]
print("\nDesigns still without a 100 km/h NPZ (%d): %s" % (len(miss), " ".join(miss[:40]) + (" ..." if len(miss) > 40 else "")))
