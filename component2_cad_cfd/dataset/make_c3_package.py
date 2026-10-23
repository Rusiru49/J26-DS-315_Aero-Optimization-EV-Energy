import csv, json, os, shutil, glob, sys, numpy as np
OUT = os.path.expanduser("~/Documents/EV_Aero/for_component3")
NPZ = "npz_morph"; DT = "design_table_sedan_v5.csv"
SHAPE = ["windshield_angle_deg","spoiler_angle_deg","boat_tail_angle_deg","hood_angle_deg","ground_clearance_mm"]
B1 = ["design_id","speed_kmh"]+SHAPE+["Cd","Cd_mean","Cd_std","Cd_p","Cd_f","Cl","Cl_p","Cl_f","A_frontal","q_ref","iterations","converged",
      "y_plus_mean","y_plus_max","cell_count","checkMesh_passed","morph_passed","final_res_p","final_res_U"]
EXTRA = ["Cd_mean500","Cd_drift_pct","Cd_sum_vs_OF_pct","morph_err_p99_mm","morph_max_disp_mm","max_nonortho","status","notes"]
ok_all = True
def chk(name, cond, detail=""):
    global ok_all; ok_all &= bool(cond); print(("  PASS  " if cond else "  FAIL  ") + name + (("  -> " + detail) if detail else ""))
def fl(x):
    try: return float(x)
    except: return float("nan")
def tf(x): return "true" if str(x).strip().lower() in ("true","1","1.0","yes") else "false"
rt = list(csv.DictReader(open("run_table.csv")))
qc = {}
if os.path.exists("qc_summary.csv"):
    for r in csv.DictReader(open("qc_summary.csv")):
        qc[(r.get("design_id"), str(r.get("speed_kmh")))] = r
os.makedirs(OUT + "/npz", exist_ok=True)
rows = []
for r in rt:
    if r.get("status", "OK") != "OK": continue
    o = {k: r.get(k, "") for k in B1+EXTRA}
    o["Cd"] = r.get("Cd_last") or r.get("Cd")
    o["Cd_mean"] = r.get("Cd_mean100") or r.get("Cd_mean")
    o["Cd_mean500"] = r.get("Cd_mean500") or r.get("Cd")
    o["speed_kmh"] = int(round(fl(r["speed_kmh"])))
    for k in ("converged","checkMesh_passed","morph_passed"): o[k] = tf(r.get(k))
    q = qc.get((r["design_id"], str(r["speed_kmh"]).replace(".0","")))
    o["notes"] = (q.get("flags") or q.get("notes") or "") if q else ""
    rows.append(o)
rows.sort(key=lambda o: (o["design_id"], o["speed_kmh"]))
with open(OUT + "/run_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=B1+EXTRA, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
m = {"WindshieldAngle":"windshield_angle_deg","HoodAngle":"hood_angle_deg","SpoilerAngle":"spoiler_angle_deg","BoatTailAngle":"boat_tail_angle_deg","GroundClearance":"ground_clearance_mm"}
dt = list(csv.DictReader(open(DT)))
dk = ["design_id"]+SHAPE+["is_corner","run_order"]
with open(OUT + "/design_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=dk); w.writeheader()
    for r in dt:
        o = {m[k]: r[k] for k in m}; o.update(design_id=r["design_id"], is_corner=("true" if r["is_corner"] in ("1","true","True") else "false"), run_order=r["run_order"]); w.writerow(o)
json.dump({"air_density_kg_m3":1.165,"dynamic_viscosity_Pa_s":1.86e-5,"speeds_kmh":[100,40],"speeds_m_s":[27.78,11.11],
 "turbulence_model":"k-omega SST (simpleFoam, steady RANS)","p_ref_Pa_gauge":0,"p_unit":"Pa (not divided by density)","averaging_window_iterations":100,
 "Cd":"final iteration","Cd_mean":"mean of last 100 iterations (same window as exported fields)","Cd_mean500":"mean of last 500 iterations",
 "full_car":True,"flow_direction":"-Z","normals":"outward from car into fluid","tau":"wall shear stress on the car surface (force of air on surface per area), Pa",
 "units":{"centre":"m (CAD frame)","area":"m2","A_frontal":"m2"},"solver":"OpenFOAM v2606","mesh":"coarse ~0.57 M cells (mesh independence: 0.57/1.02/1.86 M)","iterations":1500,
 "n_faces":44279}, open(OUT + "/case.json", "w"), indent=1)
for fn in ["export_case.py","qc_report.py","make_c3_package.py"]:
    if os.path.exists(fn): shutil.copy(fn, OUT)
if os.path.exists("qc_summary.csv"): shutil.copy("qc_summary.csv", OUT)
if os.path.exists(NPZ + "/baseline_faces.npz"): shutil.copy(NPZ + "/baseline_faces.npz", OUT + "/npz/")
for p in glob.glob(NPZ + "/sedan_D*_*kmh.npz"): shutil.copy(p, OUT + "/npz/")
print("\n=== check against Component 3 request ===")
cols = list(rows[0].keys()) if rows else []
chk("B1 all requested run_table columns present", all(k in cols for k in B1), str([k for k in B1 if k not in cols]))
chk("B6 speed_kmh is 100 or 40", all(o["speed_kmh"] in (100, 40) for o in rows))
chk("B6 design_id like D017", all(len(o["design_id"]) == 4 and o["design_id"][0] == "D" for o in rows))
chk("B6 booleans true/false", all(o[k] in ("true","false") for o in rows for k in ("converged","morph_passed")))
need = [k for k in B1 if k not in ("Cl_p","Cl_f","y_plus_max","Cd_p","Cd_f")]
bad = [(o["design_id"], k) for o in rows for k in need if str(o.get(k, "")).strip() in ("", "None", "nan")]
chk("B1 no empty required values", not bad, str(bad[:6]))
chk("B2 design_table has 70+ designs and is_corner/run_order", len(dt) >= 70 and "run_order" in dt[0])
nc = sum(1 for r in dt if r["is_corner"] in ("1","true","True"))
chk("B2 five corner designs flagged", nc == 5, "%d corners flagged (ask Member 1 / choose extreme designs)" % nc)
print("\n  per-face files")
base = np.load(OUT + "/npz/baseline_faces.npz") if os.path.exists(OUT + "/npz/baseline_faces.npz") else None
chk("A11 baseline_faces.npz (points, polys, poly_sizes)", base is not None and all(k in base.files for k in ("points","polys","poly_sizes")))
nref = len(base["centre"]) if base is not None else None
bd = {(o["design_id"], o["speed_kmh"]): o for o in rows}
for (d, s), o in sorted(bd.items()):
    p = OUT + "/npz/sedan_%s_%dkmh.npz" % (d, s)
    if not os.path.exists(p): chk("A2 %s %d km/h file" % (d, s), False, "missing"); continue
    z = np.load(p); n = len(z["area"]); good = True; msg = []
    for k, sh in (("centre",(n,3)),("area",(n,)),("normal",(n,3)),("p",(n,)),("tau",(n,3))):
        if k not in z.files or z[k].shape != sh or z[k].dtype != np.float32 or not np.isfinite(z[k]).all(): good = False; msg.append(k)
    if nref and n != nref: good = False; msg.append("faces %d != %d" % (n, nref))
    nn = np.linalg.norm(z["normal"], axis=1)
    if abs(nn.mean() - 1) > 0.01: good = False; msg.append("normals not unit")
    if not (z["area"] > 0).all(): good = False; msg.append("area<=0")
    if (fl(o["A_frontal"]) < 1 or fl(o["A_frontal"]) > 5): good = False; msg.append("A_frontal not m2? %s" % o["A_frontal"])
    if abs(fl(o.get("Cd_sum_vs_OF_pct"))) > 1.0: good = False; msg.append("Cd face-sum mismatch %s%%" % o.get("Cd_sum_vs_OF_pct"))
    chk("A2/A3/A10 %s %d km/h  (%d faces)" % (d, s, n), good, ", ".join(msg))
need2 = {(d, s) for d in {o["design_id"] for o in rows} for s in (100, 40)}
miss40 = sorted(d for d, s in need2 if (d, s) not in bd)
print("\n  still to come: %d design/speed rows missing (40 km/h not run yet)" % len(miss40))
print("\nOVERALL:", "ALL CHECKS PASS" if ok_all else "SOME CHECKS FAIL - see FAIL lines")
print("package folder:", OUT)
