# Replaces the D012 100 km/h data in the main package with the re-morphed run (same morph code as 40 km/h).
# Run inside ~/Documents/EV_Aero/openfoam after: python3 rebuild_run_table.py
import csv, glob, math, os, shutil, subprocess, sys
import numpy as np
HOME = os.path.expanduser("~/Documents/EV_Aero"); PKG = HOME + "/for_component3"
D, SP = "D012", 100
def rd(p): return list(csv.DictReader(open(p)))
def series(c):
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
def se_of(x):
    x = x[-100:]; n = len(x); m = sum(x)/n; v = sum((a-m)**2 for a in x)/(n-1); sd = math.sqrt(v)
    if v == 0: return sd, float(n)
    tau = 1.0
    for k in range(1, n//2):
        r = sum((x[i]-m)*(x[i+k]-m) for i in range(n-k))/((n-k)*v)
        if r <= 0: break
        tau += 2*r
    ne = max(1.0, n/tau); return sd/math.sqrt(ne), ne
new = [r for r in rd("run_table.csv") if r["design_id"] == D and int(float(r["speed_kmh"])) == SP]
if not new: sys.exit("no new D012 100 km/h row in run_table.csv - run python3 rebuild_run_table.py first")
new = new[0]
P = rd(PKG + "/run_table.csv"); cols = list(P[0].keys())
idx = [i for i, r in enumerate(P) if r["design_id"] == D and int(float(r["speed_kmh"])) == SP]
if len(idx) != 1: sys.exit("expected exactly one D012 100 km/h row in the package table")
r = P[idx[0]]; old = dict(r)
KEEP = {"design_id", "speed_kmh", "morph_confidence", "Cd_remesh", "remesh_diff_pct", "Cd_label", "Cd_mean_se", "Cd_se_neff", "scenario", "notes"} | {k for k in cols if k.startswith("map_")}
for k in cols:
    if k in KEEP or k not in new: continue
    r[k] = new[k]
r["Cd"] = new.get("Cd_last") or new.get("Cd") or r["Cd"]
r["Cd_mean"] = new.get("Cd_mean100") or new.get("Cd_mean") or r["Cd_mean"]
if "Cd_mean500" in cols: r["Cd_mean500"] = new.get("Cd_mean500") or new.get("Cd") or r.get("Cd_mean500", "")
for k in ("converged", "checkMesh_passed", "morph_passed"):
    if k in r and r[k] != "": r[k] = "true" if str(r[k]).strip().lower() in ("true", "1", "1.0", "yes") else "false"
r["Cd_label"] = "%.4f" % float(r["Cd_mean"])
if r.get("Cd_remesh"):
    r["remesh_diff_pct"] = "%.1f" % (100*(float(r["Cd_mean"]) - float(r["Cd_remesh"]))/float(r["Cd_remesh"]))
x = series("cases_morph/%s_%dkmh_coarse" % (D, SP))
if x: s, ne = se_of(x); r["Cd_mean_se"] = "%.5f" % s; r["Cd_se_neff"] = "%.1f" % ne
note = "re-morphed 2026-10-21 with the current morph code (same geometry as the 40 km/h run); earlier 100 km/h morph (older code) had Cd_mean %s" % old["Cd_mean"]
r["notes"] = (r.get("notes", "") + "; " if r.get("notes") else "") + note
f = "npz_morph/sedan_%s_%dkmh.npz" % (D, SP)
z = np.load(f); q = float(z["q_ref"]); A = float(z["Aref"]); a = z["area"].astype(float)
F = (-(z["p"].astype(float)*a)[:, None]*z["normal"].astype(float)).sum(0) + (z["tau"].astype(float)*a[:, None]).sum(0)
cds = -F[2]/(q*A); dm = 100*(cds-float(r["Cd_mean"]))/float(r["Cd_mean"])
print("faces", len(a), " Cd_mean %s  face-sum %.4f (%.2f %%)" % (r["Cd_mean"], cds, dm))
if len(a) != 44279 or abs(dm) > 1.0 or not np.isfinite(z["p"].astype(float)).all(): sys.exit("CHECK FAILED - package not changed")
os.makedirs(HOME + "/backup/pkg_before_D012_fix", exist_ok=True)
shutil.copy(PKG + "/run_table.csv", HOME + "/backup/pkg_before_D012_fix/run_table.csv")
shutil.copy(PKG + "/npz/sedan_%s_%dkmh.npz" % (D, SP), HOME + "/backup/pkg_before_D012_fix/")
shutil.copy(f, PKG + "/npz/sedan_%s_%dkmh.npz" % (D, SP))
with open(PKG + "/run_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(P)
print("D012 100 km/h: Cd_mean %s -> %s, Cd_label %s -> %s, remesh_diff_pct %s -> %s, SE %s" % (old["Cd_mean"], r["Cd_mean"], old["Cd_label"], r["Cd_label"], old.get("remesh_diff_pct"), r.get("remesh_diff_pct"), r.get("Cd_mean_se")))
os.chdir(HOME)
if os.path.exists("for_component3.zip"): os.remove("for_component3.zip")
subprocess.run(["zip", "-r", "-q", "for_component3.zip", "for_component3", "-x", "*.DS_Store"], check=True)
print("main package updated and re-zipped (backup of the old files in backup/pkg_before_D012_fix)")
