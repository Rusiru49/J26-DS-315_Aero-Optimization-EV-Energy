import csv, glob, json, os, re, shutil
import numpy as np
BASE = os.path.expanduser("~/Documents/EV_Aero")
P = BASE + "/for_component3/run_table.csv"; R = BASE + "/for_component3/README.md"
shutil.copy(P, "/tmp/run_table_backup.csv")
rows = list(csv.DictReader(open(P))); cols = list(rows[0].keys())
MAPCOLS = ["map_cd_err_pct", "map_p_err_cp", "map_tau_err_cp", "map_dist_mean_mm", "map_dist_p99_mm"]
for c in ["morph_confidence", "Cd_remesh", "remesh_diff_pct"] + MAPCOLS:
    if c not in cols: cols.append(c)
def remesh_cd(d):
    c = "cases_prod/%s_100kmh_coarse" % d
    if not os.path.exists(c + "/result.json"): return None, None          # run not finished / not exported
    fs = sorted(glob.glob(c + "/postProcessing/forceCoeffs*/*/forceCoeffs.dat") + glob.glob(c + "/postProcessing/forceCoeffs*/*/coefficient.dat"))
    if not fs: return None, None
    rr = [l.split() for l in open(fs[-1]) if l.strip() and not l.startswith("#")]
    if len(rr) < 100: return None, None
    return float(np.mean([float(x[1]) for x in rr[-100:]])), float(rr[-1][0])
for r in rows:
    d = r["design_id"]
    base = re.sub(r";? ?(low-confidence|re-mesh note).*$", "", r.get("notes", "") or "")
    rc, tl = remesh_cd(d) if d != "D000" else (None, None)
    extra = []
    if rc is not None:
        diff = 100*(float(r["Cd_mean"]) - rc)/rc
        r["Cd_remesh"] = "%.4f" % rc; r["remesh_diff_pct"] = "%.1f" % diff
        low = abs(diff) > 5.0
        if low: extra.append("low-confidence morph: morph differs from full re-mesh by %.1f percent (re-mesh Cd_mean %.4f)" % (diff, rc))
        if tl < 1499: extra.append("re-mesh note: force file ends at iteration %d although the solver reached 1500 (converged); Cd_remesh is the mean of the last 100 rows" % tl)
    else:
        r["Cd_remesh"] = ""; r["remesh_diff_pct"] = ""; low = False
    r["morph_confidence"] = "low" if low else "high"
    jp = "npz/sedan_%s_100kmh_maperr.json" % d
    j = json.load(open(jp)) if os.path.exists(jp) else {}
    for c in MAPCOLS: r[c] = j.get(c, "")
    if "notes" in r:
        r["notes"] = base
        for e in extra: r["notes"] = (r["notes"] + "; " if r["notes"] else "") + e
    print(r["design_id"], r["morph_confidence"], r["Cd_remesh"], r["remesh_diff_pct"], r["map_cd_err_pct"], r["map_p_err_cp"], r["map_tau_err_cp"], "" if tl is None else "last_t=%d" % tl)
with open(P, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
val = [(r["design_id"], r["remesh_diff_pct"], r["Cd_remesh"]) for r in rows if r["remesh_diff_pct"]]
txt = "\n## Validation summary (morph vs full re-mesh, same coarse settings)\n"
txt += "remesh_diff_pct = 100 * (Cd_mean of the morph - Cd_remesh) / Cd_remesh, with both Cd values being the mean of the last 100 iterations.\n"
txt += "Validated designs: " + ", ".join("%s %s %%" % (a, b) for a, b, c in val) + ".\n"
txt += "The map_* columns describe the mapping of the re-mesh field onto the 44,279 baseline faces: relative Cd error, area-weighted RMS error of p and |tau| divided by q_ref (evaluated on the re-mesh faces), and the mean and p99 mapping distance.\n"
s = open(R).read(); i = s.find("\n## Validation summary (morph vs full re-mesh")
if i >= 0: s = s[:i]
open(R, "w").write(s.rstrip("\n") + "\n" + txt)
print("README and run_table.csv updated")
