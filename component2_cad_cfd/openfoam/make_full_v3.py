import csv, os, shutil, subprocess, sys
import numpy as np
HOME = os.path.expanduser("~/Documents/EV_Aero")
PKG, U1, U2, OUT = HOME + "/for_component3", HOME + "/for_component3_urban1", HOME + "/for_component3_urban2", HOME + "/for_component3_full"
import glob
BT = sorted(glob.glob(HOME + "/for_component3_batch[3-9]"))
def rd(p): return list(csv.DictReader(open(p)))
for need in (PKG + "/run_table.csv", U1 + "/run_table_urban1.csv", U2 + "/run_table_urban2.csv"):
    if not os.path.exists(need): sys.exit("MISSING " + need)
P, A, B = rd(PKG + "/run_table.csv"), rd(U1 + "/run_table_urban1.csv"), rd(U2 + "/run_table_urban2.csv")
for b in BT:
    print("including", b); B = B + rd(b + "/run_table_" + os.path.basename(b).split("_")[-1] + ".csv")
cols = list(P[0].keys())
for r in A + B:
    for k in r:
        if k not in cols: cols.append(k)
rows = P + A + B
for r in rows:
    r["speed_kmh"] = str(int(float(r["speed_kmh"])))
    for k in ("converged", "checkMesh_passed", "morph_passed"):
        if k in r and r[k] != "": r[k] = "true" if str(r[k]).strip().lower() in ("true", "1", "1.0", "yes") else "false"
    if not r.get("scenario"): r["scenario"] = "highway" if r["speed_kmh"] == "100" else "urban"
rows.sort(key=lambda r: (r["design_id"], -int(r["speed_kmh"])))
keys = [(r["design_id"], r["speed_kmh"]) for r in rows]
dup = [k for k in set(keys) if keys.count(k) > 1]
designs = sorted(set(d for d, s in keys))
miss = [(d, s) for d in designs for s in ("100", "40") if (d, s) not in keys]
print("designs:", len(designs), " runs:", len(rows), " both speeds:", sum(1 for d in designs if (d, "100") in keys and (d, "40") in keys))
if dup: sys.exit("DUPLICATE rows: %s" % dup)
if miss: sys.exit("MISSING rows: %s" % miss)
if os.path.exists(OUT): shutil.rmtree(OUT)
shutil.copytree(PKG, OUT, ignore=shutil.ignore_patterns(".DS_Store"))
for src in [U1, U2] + BT:
    if not os.path.isdir(src + "/npz"): continue
    for f in os.listdir(src + "/npz"):
        if f.endswith("_40kmh.npz") or f.endswith("_40kmh_morph.npz") or (src in BT and f.endswith("_100kmh.npz")): shutil.copy(src + "/npz/" + f, OUT + "/npz/" + f)
bad = []
for d in designs:
    for s in ("100", "40"):
        f = OUT + "/npz/sedan_%s_%skmh.npz" % (d, s)
        if not os.path.exists(f): bad.append("missing " + f); continue
        z = np.load(f)
        if len(z["area"]) != 44279 or not np.isfinite(z["p"].astype(float)).all(): bad.append("bad " + f)
print("NPZ files checked:", len(designs)*2, " problems:", len(bad))
if bad: sys.exit("\n".join(bad))
with open(OUT + "/run_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", restval=""); w.writeheader(); w.writerows(rows)
lab = {k: r for k, r in zip(keys, rows)}
tab = "| Design | morph_confidence | Cd_label 100 km/h (highway) | Cd_label 40 km/h (urban) | difference | Cd_mean_se (100 / 40) |\n|---|---|---|---|---|---|\n"
for d in designs:
    h, u = lab[(d, "100")], lab[(d, "40")]; a, b = float(h["Cd_label"]), float(u["Cd_label"])
    tab += "| %s | %s | %.4f | %.4f | %+.1f %% | %s / %s |\n" % (d, h.get("morph_confidence", ""), a, b, 100*(b-a)/a, h.get("Cd_mean_se", ""), u.get("Cd_mean_se", ""))
orig = open(PKG + "/README.md").read() if os.path.exists(PKG + "/README.md") else ""
txt = """# EV Aero-Optimizer - Component 2 CFD dataset (complete package: all %d shapes at both speeds)

**highway = 100 km/h (27.78 m/s)** and **urban = 40 km/h (11.11 m/s)**, %d runs in total (%d shapes x 2 speeds). Column `scenario` tells which one a row is.

## What is in this folder
- run_table.csv : one row per run (%d rows), all columns, including Cd_label, Cd_mean_se, Cd_se_neff, scenario
- npz/sedan_<D>_100kmh.npz and npz/sedan_<D>_40kmh.npz : per-face data on the same 44,279 baseline faces (centre, area, normal, p, tau, Cd, Aref, U, rho, q_ref, p_ref)
- npz/baseline_faces.npz : face order reference; npz/*_morph.npz : pure morph fields for the low-confidence designs (where a mapped re-mesh field is used in the main file)
- case.json, this README

## Which Cd to use as the label
Use the column **Cd_label** (Cd_remesh when morph_confidence is low, otherwise Cd_mean = last-100 mean). The `Cd` stored inside an NPZ is the last iteration and can differ from the label by about 1 %%.

## Standard error
Cd_mean_se = std of the last 100 iterations / sqrt(n_eff), n_eff = 100 / (1 + 2 * sum of autocorrelations) (Cd_se_neff). It covers iteration-to-iteration fluctuation only, not mesh, discretisation or turbulence-model error. Typical value 0.001-0.005 (0.4-1.4 %%), so differences of about 1 %% or less are within noise.

## Cd_label per design and speed
%s
## Honest notes
- OpenFOAM (not ANSYS), coarse mesh (about 0.57 M cells), %d of the 70 designs so far.
- Morph-versus-re-mesh validation was done at 100 km/h only; at 40 km/h the low-confidence designs (D019, D036) were run as full re-meshes and mapped.
- D037 (+3.8 %%) and D003 (-2.5 %%) change more between the two speeds than the rest; with a standard error of about 1 %% they may be partly noise.

---
# Original package README (highway data)
%s
""" % (len(designs), len(rows), len(designs), len(rows), tab, len(designs), orig)
open(OUT + "/README.md", "w").write(txt)
os.chdir(HOME)
if os.path.exists("for_component3_full.zip"): os.remove("for_component3_full.zip")
subprocess.run(["zip", "-r", "-q", "for_component3_full.zip", "for_component3_full", "-x", "*.DS_Store"], check=True)
print("ZIP:", HOME + "/for_component3_full.zip", os.path.getsize("for_component3_full.zip")//1024, "KB")
