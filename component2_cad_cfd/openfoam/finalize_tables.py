import csv, glob, json, math, os, shutil, subprocess, sys
HOME = os.path.expanduser("~/Documents/EV_Aero")
PKG, B2, U1 = HOME + "/for_component3", HOME + "/for_component3_batch2", HOME + "/for_component3_urban1"
NEW = ["Cd_label", "Cd_mean_se", "Cd_se_neff", "scenario"]
MAPK = ["map_cd_err_pct", "map_p_err_cp", "map_tau_err_cp", "map_dist_mean_mm", "map_dist_p99_mm"]
def rd(p): return list(csv.DictReader(open(p)))
def wr(p, rows, cols):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
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
def add_cols(r, lookup=None):
    sp = int(float(r["speed_kmh"])); d = r["design_id"]
    conf = r.get("morph_confidence") or (lookup or {}).get((d, sp), ("", ""))[0]
    rm = r.get("Cd_remesh") or (lookup or {}).get((d, sp), ("", ""))[1]
    low = (conf == "low" and rm not in ("", None))
    r["Cd_label"] = "%.4f" % float(rm) if low else "%.4f" % float(r["Cd_mean"])
    r["scenario"] = "highway" if sp == 100 else "urban"
    x = series(d, sp, low)
    if x: s, ne = se_of(x); r["Cd_mean_se"] = "%.5f" % s; r["Cd_se_neff"] = "%.1f" % ne
    else: r["Cd_mean_se"] = ""; r["Cd_se_neff"] = ""; print("WARNING: no Cd series for", d, sp)
    return r
def show(r): print("%-5s %3s km/h %-9s conf %-4s Cd_mean %s Cd_label %s  SE %s (n_eff %s)" % (r["design_id"], r["speed_kmh"], r["scenario"], r.get("morph_confidence", "-"), r["Cd_mean"], r["Cd_label"], r["Cd_mean_se"], r["Cd_se_neff"]))
if not os.path.exists(PKG + "/run_table.csv"): sys.exit("missing " + PKG + "/run_table.csv")
P = rd(PKG + "/run_table.csv"); H = list(P[0].keys())
cols = H + [c for c in NEW if c not in H]
print("== main package"); 
for r in P: add_cols(r); show(r)
wr(PKG + "/run_table.csv", P, cols)
lookup = {(r["design_id"], int(float(r["speed_kmh"]))): (r.get("morph_confidence", ""), r.get("Cd_remesh", "")) for r in P}
p100 = {r["design_id"]: r for r in P if int(float(r["speed_kmh"])) == 100}
if os.path.exists(B2 + "/run_table_batch2.csv"):
    print("== batch2"); Bt = rd(B2 + "/run_table_batch2.csv"); hb = list(Bt[0].keys())
    for r in Bt: add_cols(r, lookup); show(r)
    wr(B2 + "/run_table_batch2.csv", Bt, hb + [c for c in NEW if c not in hb])
print("== urban batch 1")
O = {(r["design_id"], int(float(r["speed_kmh"]))): r for r in rd("run_table.csv")}
DES = ["D000", "D019", "D002", "D042", "D047", "D049"]; U = []
for d in DES:
    r = dict(O[(d, 40)]); b = p100.get(d)
    for k in cols: r.setdefault(k, "")
    r["morph_confidence"] = "low" if d == "D019" else ((b or {}).get("morph_confidence") or "high")
    r["Cd_remesh"] = ""; r["remesh_diff_pct"] = ""
    for k in MAPK: r[k] = ""
    note = ""
    if d == "D019":
        r["Cd_remesh"] = "%.4f" % float(r["Cd_mean"])
        if b and "morph_passed" in b: r["morph_passed"] = b["morph_passed"]
        jp = "npz/sedan_D019_40kmh_maperr.json"
        if os.path.exists(jp):
            j = json.load(open(jp))
            for k in MAPK: r[k] = j.get(k, "")
        note = "urban re-mesh run (full snappyHexMesh) at 40 km/h; label = this re-mesh Cd_mean; NPZ = re-mesh field mapped onto the morph faces (mapping Cd error %s %%); morph_confidence low as at 100 km/h; no 40 km/h morph run exists so remesh_diff_pct is blank" % r["map_cd_err_pct"]
    elif b and b.get("remesh_diff_pct"):
        note = "morph_confidence high from the 100 km/h validation (morph vs re-mesh %s %%); not re-meshed at 40 km/h" % b["remesh_diff_pct"]
    elif d != "D000":
        note = "morph run, mesh-quality gate passed; not re-meshed (no validation)"
    if "notes" in r and note: r["notes"] = (r["notes"] + "; " if r["notes"] else "") + note
    add_cols(r); show(r); U.append(r)
wr(U1 + "/run_table_urban1.csv", U, cols)
def cdl(d, sp):
    if sp == 40:
        for r in U:
            if r["design_id"] == d: return float(r["Cd_label"])
    if d in p100: return float(p100[d]["Cd_label"])
    if (d, 100) in O: return float(O[(d, 100)]["Cd_mean"])
tab = "| Design | Cd_label 40 km/h | Cd_label 100 km/h | difference |\n|---|---|---|---|\n"
for d in DES:
    a, h = cdl(d, 40), cdl(d, 100)
    tab += "| %s | %.4f | %s | %s |\n" % (d, a, "%.4f" % h if h else "-", "%+.1f %%" % (100*(a-h)/h) if h else "-")
LAB = """## Cd_label and standard error (columns added in every run table)
- **Cd_label** = the value to train on. Rule: Cd_label = Cd_remesh when morph_confidence is low, otherwise Cd_mean (last-100 mean).
  Validated high-confidence designs (D002, D003, D012, D038, D049) keep Cd_mean because their NPZ holds the morph field.
- **Cd_mean_se** = standard error of Cd_mean: std of the last 100 iterations divided by sqrt(n_eff), where n_eff = 100 / (1 + 2 * sum of the autocorrelations) (**Cd_se_neff**). It measures only the iteration-to-iteration fluctuation of the converged solution; it does not include mesh, discretisation or turbulence-model error. It can be larger for the oscillating designs (D039, D047).
- **scenario** = highway (100 km/h) or urban (40 km/h).
"""
def patch_readme(p, urban=False):
    if not os.path.exists(p): return
    s = open(p).read(); i = s.find("\n## Cd_label and standard error")
    if i >= 0:
        j = s.find("\n## ", i + 5); s = s[:i] + (s[j:] if j >= 0 else "")
    if urban:
        a = s.find("## Which Cd to use as the label"); b = s.find("## Results")
        if a >= 0 and b > a: s = s[:a] + "## Which Cd to use as the label\nUse the column **Cd_label** (see below). D000 and D019 are full re-mesh runs; D019's NPZ holds the re-mesh field mapped onto the morph faces (mapping Cd error about -2.8 %). D002, D042, D047, D049 are FFD-morph runs from the D000 base mesh.\n\n" + s[b:]
        a = s.find("## Results"); b = s.find("## Honest notes")
        if a >= 0 and b > a: s = s[:a] + "## Results\n" + tab + "\n" + s[b:]
    s = s.rstrip("\n") + "\n\n" + LAB
    open(p, "w").write(s)
patch_readme(PKG + "/README.md"); patch_readme(B2 + "/README_batch2.md"); patch_readme(U1 + "/README.md", True)
os.chdir(HOME)
for z in ("for_component3", "for_component3_batch2", "for_component3_urban1"):
    if os.path.isdir(z):
        if os.path.exists(z + ".zip"): os.remove(z + ".zip")
        subprocess.run(["zip", "-r", "-q", z + ".zip", z, "-x", "*.DS_Store"], check=True)
        print("ZIP", z + ".zip", os.path.getsize(z + ".zip")//1024, "KB")
