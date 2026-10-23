import csv, glob, json, math, os, shutil, subprocess, sys
import numpy as np
if len(sys.argv) < 3: sys.exit("usage: python3 make_urban.py <tag> <D003,D004,...>")
TAG = sys.argv[1]; DES = sys.argv[2].split(",")
HOME = os.path.expanduser("~/Documents/EV_Aero"); PKG = HOME + "/for_component3"; OUT = HOME + "/for_component3_" + TAG
MAPK = ["map_cd_err_pct", "map_p_err_cp", "map_tau_err_cp", "map_dist_mean_mm", "map_dist_p99_mm"]
NEW = ["Cd_label", "Cd_mean_se", "Cd_se_neff", "scenario"]
def rd(p): return list(csv.DictReader(open(p)))
def series(d, sp, remesh):
    order = ["cases_prod", "cases_morph"] if (remesh or d == "D000") else ["cases_morph", "cases_prod"]
    for b in order:
        c = "%s/%s_%dkmh_coarse" % (b, d, sp)
        for name in ("coefficient.dat", "forceCoeffs.dat"):
            fs = glob.glob(c + "/postProcessing/**/" + name, recursive=True)
            if not fs: continue
            idx = 1; vals = []
            for l in open(fs[0]):
                if l.startswith("#"):
                    t = l.lstrip("#").split()
                    if "Cd" in t and "Time" in t: idx = t.index("Cd")
                    continue
                s = l.split()
                try: vals.append(float(s[idx]))
                except Exception: pass
            if len(vals) >= 100: return vals
    return None
def se_of(x):
    x = x[-100:]; n = len(x); m = sum(x)/n
    v = sum((a-m)**2 for a in x)/(n-1); sd = math.sqrt(v)
    if v == 0: return sd, float(n)
    tau = 1.0
    for k in range(1, n//2):
        r = sum((x[i]-m)*(x[i+k]-m) for i in range(n-k))/((n-k)*v)
        if r <= 0: break
        tau += 2*r
    ne = max(1.0, n/tau)
    return sd/math.sqrt(ne), ne
P = rd(PKG + "/run_table.csv"); cols = list(P[0].keys()); cols += [c for c in NEW if c not in cols]
p100 = {r["design_id"]: r for r in P if int(float(r["speed_kmh"])) == 100}
O = {(r["design_id"], int(float(r["speed_kmh"]))): r for r in rd("run_table.csv")}
os.makedirs(OUT + "/npz", exist_ok=True)
base = np.load("npz_morph/baseline_faces.npz")["centre"].astype(float)
low = {d: (p100.get(d, {}).get("morph_confidence") == "low") for d in DES}
ok = True; rows = []
for d in DES:
    f = "npz/sedan_D000_40kmh.npz" if d == "D000" else (OUT + "/npz/sedan_%s_40kmh.npz" % d if low[d] else "npz_morph/sedan_%s_40kmh.npz" % d)
    if not os.path.exists(f):
        print("MISSING", f, ("  -> run: python3 map_remesh.py %s --speed=40 --write --pkg=%s/npz" % (d, OUT)) if low[d] else ""); ok = False; continue
    z = np.load(f); n = len(z["area"]); msgs = []
    if n != 44279: msgs.append("faces %d" % n)
    for k in ("centre", "area", "normal", "p", "tau"):
        if not np.isfinite(z[k].astype(float)).all(): msgs.append("non-finite " + k)
    if (z["area"] <= 0).any(): msgs.append("area<=0")
    if np.abs(np.linalg.norm(z["normal"].astype(float), axis=1) - 1).max() > 1e-3: msgs.append("normals not unit")
    q = float(z["q_ref"]); A = float(z["Aref"]); a = z["area"].astype(float)
    F = (-(z["p"].astype(float)*a)[:, None]*z["normal"].astype(float)).sum(0) + (z["tau"].astype(float)*a[:, None]).sum(0)
    cd_sum = -F[2]/(q*A); cd = float(z["Cd"]); dd = 100*(cd_sum-cd)/cd
    if d == "D000" and np.abs(z["centre"].astype(float)-base).max() > 1e-3: msgs.append("face order differs from baseline")
    ref = float(O[(d, 40)]["Cd_mean"]); dm = 100*(cd_sum-ref)/ref
    if not low[d] and abs(dm) > 1.0: msgs.append("Cd_sum differs from Cd_mean by %.2f%%" % dm)
    print("%s faces %d Cd(file) %.4f Cd_mean %.4f Cd(sum) %.4f (vs Cd_mean %.2f%%)  %s" % (d, n, cd, ref, cd_sum, dm, "FAIL: " + "; ".join(msgs) if msgs else "OK"))
    if msgs: ok = False
    if not os.path.exists(OUT + "/npz/sedan_%s_40kmh.npz" % d): shutil.copy(f, OUT + "/npz/sedan_%s_40kmh.npz" % d)
    r = dict(O[(d, 40)]); b = p100.get(d, {})
    for k in cols: r.setdefault(k, "")
    r["morph_confidence"] = "low" if low[d] else (b.get("morph_confidence") or "high")
    r["Cd_remesh"] = ""; r["remesh_diff_pct"] = ""
    for k in MAPK: r[k] = ""
    note = ""
    if low[d]:
        r["Cd_remesh"] = "%.4f" % float(r["Cd_mean"])
        if "morph_passed" in b: r["morph_passed"] = b["morph_passed"]
        jp = "npz/sedan_%s_40kmh_maperr.json" % d
        if os.path.exists(jp):
            j = json.load(open(jp))
            for k in MAPK: r[k] = j.get(k, "")
        note = "urban re-mesh run (full snappyHexMesh) at 40 km/h; label = this re-mesh Cd_mean; NPZ = re-mesh field mapped onto the morph faces (mapping Cd error %s %%); morph_confidence low as at 100 km/h; no 40 km/h morph run exists so remesh_diff_pct is blank" % r["map_cd_err_pct"]
    elif b.get("remesh_diff_pct"):
        note = "morph_confidence high from the 100 km/h validation (morph vs re-mesh %s %%); not re-meshed at 40 km/h" % b["remesh_diff_pct"]
    elif d != "D000":
        note = "morph run, mesh-quality gate passed; not re-meshed (no validation)"
    if "notes" in r and note: r["notes"] = (r["notes"] + "; " if r["notes"] else "") + note
    sp = 40; useRM = low[d] and r["Cd_remesh"] != ""
    r["Cd_label"] = r["Cd_remesh"] if useRM else "%.4f" % float(r["Cd_mean"]); r["scenario"] = "urban"
    x = series(d, sp, useRM)
    if x: s, ne = se_of(x); r["Cd_mean_se"] = "%.5f" % s; r["Cd_se_neff"] = "%.1f" % ne
    else: r["Cd_mean_se"] = ""; r["Cd_se_neff"] = ""; print("WARNING: no Cd series for", d)
    rows.append(r)
if not ok: sys.exit("\nSOME CHECKS FAILED / FILES MISSING - nothing zipped")
shutil.copy("npz_morph/baseline_faces.npz", OUT + "/npz/baseline_faces.npz")
with open(OUT + "/run_table_%s.csv" % TAG, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
tab = "| Design | morph_confidence | Cd_label 40 km/h | Cd_label 100 km/h | difference | Cd_mean_se (40) |\n|---|---|---|---|---|---|\n"; diffs = []
for r in rows:
    d = r["design_id"]; a = float(r["Cd_label"]); h = float(p100[d]["Cd_label"]) if d in p100 and p100[d].get("Cd_label") else None
    if h: diffs.append(100*(a-h)/h)
    tab += "| %s | %s | %.4f | %s | %s | %s |\n" % (d, r["morph_confidence"], a, "%.4f" % h if h else "-", "%+.1f %%" % (100*(a-h)/h) if h else "-", r["Cd_mean_se"])
val = ", ".join("%s %s %%" % (d, p100[d]["remesh_diff_pct"]) for d in DES if d in p100 and p100[d].get("remesh_diff_pct"))
rng = "from %+.1f %% to %+.1f %%" % (min(diffs), max(diffs)) if diffs else "n/a"
txt = """# Urban batch %s (40 km/h) - EV Aero-Optimizer, Component 2

Scenario: **urban = 40 km/h (11.11 m/s)**. The main package holds the **highway = 100 km/h (27.78 m/s)** data.
%d designs: %s.

## Setup (same as the 100 km/h data, only the inlet/ground speed changes)
OpenFOAM ESI v2606, simpleFoam, k-omega SST, full car, moving ground, rho 1.165 kg/m3, mu 1.86e-5, 1500 iterations,
coarse mesh (about 0.57 M cells), p in Pa in the NPZ files (OpenFOAM computes kinematic p; it is multiplied by rho).

## Files
- npz/sedan_<D>_40kmh.npz : per-face data on the same 44,279 baseline faces (centre, area, normal, p, tau, Cd, Aref, U, rho, q_ref, p_ref)
- npz/baseline_faces.npz : baseline face centres (face order reference)
- run_table_%s.csv : one row per design, same columns as the 100 km/h run table (morph_confidence, Cd_remesh, remesh_diff_pct, map_*, Cd_label, Cd_mean_se, Cd_se_neff, scenario)

## Which Cd to use as the label
Use the column **Cd_label**: Cd_remesh when morph_confidence is low, otherwise Cd_mean (last-100 mean). Low-confidence designs are full re-mesh runs whose field is mapped onto the morph faces; the others are FFD-morph runs from the D000 base mesh.

## Results
%s
## Honest notes
- The 40 km/h label differs from the 100 km/h label %s. The standard error of a single Cd_mean is about 0.001-0.005 (0.4-1.4 %%), so differences of about 1 %% or less are within noise.
- The morph-versus-re-mesh validation was done at 100 km/h only (%s). Designs without an entry were only mesh-quality checked at 100 km/h and were not re-meshed.
- Cd_mean_se = std of the last 100 iterations / sqrt(n_eff), n_eff = 100 / (1 + 2 * sum of autocorrelations) (Cd_se_neff). It covers iteration-to-iteration fluctuation only, not mesh, discretisation or turbulence-model error.
- This is OpenFOAM (not ANSYS) on a coarse mesh.
""" % (TAG.replace("urban", ""), len(DES), ", ".join(DES), TAG, tab, rng, val or "none of these designs")
open(OUT + "/README.md", "w").write(txt)
os.chdir(HOME)
if os.path.exists("for_component3_%s.zip" % TAG): os.remove("for_component3_%s.zip" % TAG)
subprocess.run(["zip", "-r", "-q", "for_component3_%s.zip" % TAG, "for_component3_" + TAG, "-x", "*.DS_Store"], check=True)
print("\nZIP:", HOME + "/for_component3_%s.zip" % TAG, os.path.getsize("for_component3_%s.zip" % TAG)//1024, "KB")
