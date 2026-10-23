import csv, glob, json, math, os, shutil, subprocess, sys
import numpy as np
if len(sys.argv) < 3: sys.exit("usage: python3 make_batch3.py <tag> <D061,D043,...>")
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
ok = True; rows = []
for d, sp in [(d, sp) for d in DES for sp in (100, 40)]:
    f = "npz_morph/sedan_%s_%dkmh.npz" % (d, sp)
    if not os.path.exists(f): print("MISSING", f); ok = False; continue
    z = np.load(f); n = len(z["area"]); msgs = []
    if n != 44279: msgs.append("faces %d" % n)
    for k in ("centre", "area", "normal", "p", "tau"):
        if not np.isfinite(z[k].astype(float)).all(): msgs.append("non-finite " + k)
    if (z["area"] <= 0).any(): msgs.append("area<=0")
    if np.abs(np.linalg.norm(z["normal"].astype(float), axis=1) - 1).max() > 1e-3: msgs.append("normals not unit")
    q = float(z["q_ref"]); A = float(z["Aref"]); a = z["area"].astype(float)
    F = (-(z["p"].astype(float)*a)[:, None]*z["normal"].astype(float)).sum(0) + (z["tau"].astype(float)*a[:, None]).sum(0)
    cd_sum = -F[2]/(q*A)
    ref = float(O[(d, sp)]["Cd_mean"]); dm = 100*(cd_sum-ref)/ref
    if abs(dm) > 1.0: msgs.append("Cd_sum differs from Cd_mean by %.2f%%" % dm)
    print("%s %3d km/h faces %d Cd_mean %.4f Cd(sum) %.4f (%.2f%%)  %s" % (d, sp, n, ref, cd_sum, dm, "FAIL: " + "; ".join(msgs) if msgs else "OK"))
    if msgs: ok = False
    shutil.copy(f, OUT + "/npz/sedan_%s_%dkmh.npz" % (d, sp))
    r = dict(O[(d, sp)])
    for k in cols: r.setdefault(k, "")
    conf = (p100.get(d, {}).get("morph_confidence") or r.get("morph_confidence") or "high")
    r["morph_confidence"] = conf; r["Cd_remesh"] = ""; r["remesh_diff_pct"] = ""
    for k in MAPK: r[k] = ""
    note = "FFD-morph run from the D000 base mesh, mesh-quality gate passed; NOT validated against a full re-mesh (no re-mesh run for this design)"
    r["notes"] = (r.get("notes", "") + "; " if r.get("notes") else "") + note
    for k in ("converged", "checkMesh_passed", "morph_passed"):
        if k in r: r[k] = "true" if str(r[k]).strip().lower() in ("true", "1", "1.0", "yes") else "false"
    r["speed_kmh"] = str(sp); r["scenario"] = "highway" if sp == 100 else "urban"
    r["Cd_label"] = "%.4f" % float(r["Cd_mean"])
    x = series(d, sp, False)
    if x: s_, ne = se_of(x); r["Cd_mean_se"] = "%.5f" % s_; r["Cd_se_neff"] = "%.1f" % ne
    else: r["Cd_mean_se"] = ""; r["Cd_se_neff"] = ""; print("WARNING: no Cd series for", d, sp)
    rows.append(r)
if not ok: sys.exit("\nSOME CHECKS FAILED / FILES MISSING - nothing zipped")
shutil.copy("npz_morph/baseline_faces.npz", OUT + "/npz/baseline_faces.npz")
with open(OUT + "/run_table_%s.csv" % TAG, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
rows.sort(key=lambda r: (r["design_id"], -int(r["speed_kmh"])))
with open(OUT + "/run_table_%s.csv" % TAG, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
lab = {(r["design_id"], r["speed_kmh"]): r for r in rows}
tab = "| Design | Cd_label 100 km/h (highway) | Cd_label 40 km/h (urban) | difference | Cd_mean_se (100 / 40) |\n|---|---|---|---|---|\n"
for d in DES:
    h, u = lab[(d, "100")], lab[(d, "40")]; a_, b_ = float(h["Cd_label"]), float(u["Cd_label"])
    tab += "| %s | %.4f | %.4f | %+.1f %% | %s / %s |\n" % (d, a_, b_, 100*(b_-a_)/a_, h["Cd_mean_se"], u["Cd_mean_se"])
txt = """# Batch %s - EV Aero-Optimizer, Component 2: new shapes at BOTH speeds

%d new designs (%s), each simulated at **highway = 100 km/h (27.78 m/s)** and **urban = 40 km/h (11.11 m/s)** (%d runs).

## Setup
OpenFOAM ESI v2606, simpleFoam, k-omega SST, full car, moving ground, rho 1.165 kg/m3, mu 1.86e-5, 1500 iterations, coarse mesh (about 0.57 M cells), FFD morph from the D000 base mesh (same 44,279 faces and face order as every other file). p is in Pa in the NPZ files.

## Files
- npz/sedan_<D>_100kmh.npz and npz/sedan_<D>_40kmh.npz : per-face data (centre, area, normal, p, tau, Cd, Aref, U, rho, q_ref, p_ref)
- npz/baseline_faces.npz : face order reference
- run_table_%s.csv : one row per run, same columns as the main run table (incl. Cd_label, Cd_mean_se, Cd_se_neff, scenario)

## Label
Use **Cd_label** (= Cd_mean, the last-100 mean; all these designs are morph runs that passed the mesh-quality gate). The `Cd` stored inside an NPZ is the last iteration and can differ from Cd_label by about 1 %%.

## Results
%s
## Honest notes
- These designs were NOT validated against a full re-mesh (validation was done for 7 other designs at 100 km/h only: D003 +0.2, D012 -0.3, D038 +1.5, D002 -2.3, D049 +3.0, D036 -6.7, D019 -11.8 %% morph vs re-mesh). Expect a typical morph error of a few percent; D019 and D036 showed that individual designs can be off by more than 5 %%.
- Cd_mean_se = std of the last 100 iterations / sqrt(n_eff); it covers iteration noise only (typical 0.4-1.4 %%), so differences of about 1 %% or less between speeds are within noise.
- OpenFOAM (not ANSYS), coarse mesh.
""" % (TAG, len(DES), ", ".join(DES), len(rows), TAG, tab)
open(OUT + "/README.md", "w").write(txt)
os.chdir(HOME)
if os.path.exists("for_component3_%s.zip" % TAG): os.remove("for_component3_%s.zip" % TAG)
subprocess.run(["zip", "-r", "-q", "for_component3_%s.zip" % TAG, "for_component3_" + TAG, "-x", "*.DS_Store"], check=True)
print("\nZIP:", HOME + "/for_component3_%s.zip" % TAG, os.path.getsize("for_component3_%s.zip" % TAG)//1024, "KB")
