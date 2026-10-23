import csv, glob, json, os, shutil, sys
import numpy as np
KEYS = ["design_id","speed_kmh","windshield_angle_deg","spoiler_angle_deg","boat_tail_angle_deg","hood_angle_deg","ground_clearance_mm",
        "Cd","Cd_mean","Cd_mean500","Cd_last","Cd_mean100","Cd_mean300","Cd_drift_pct","Cd_std","Cd_p","Cd_f","Cl","Cl_p","Cl_f","A_frontal","q_ref","iterations","converged","y_plus_mean","y_plus_max",
        "cell_count","checkMesh_passed","morph_passed","final_res_p","final_res_U","Cd_sum","Cd_sum_vs_OF_pct","map_method","map_mean_dist_mm",
        "map_cd_err_pct","morph_mode","morph_err_mean_mm","morph_err_p99_mm","morph_max_disp_mm","morph_face_shift_max_mm",
        "max_nonortho","max_skewness","n_faces","mesh","half_model","wall_minutes","status"]
SHAPE = {"WindshieldAngle":"windshield_angle_deg","SpoilerAngle":"spoiler_angle_deg","BoatTailAngle":"boat_tail_angle_deg","HoodAngle":"hood_angle_deg","GroundClearance":"ground_clearance_mm"}
table = {r["design_id"]: r for r in csv.DictReader(open("design_table_sedan_v5.csv"))}
rows = []
for rj in sorted(glob.glob("cases_prod/D000_100kmh_coarse/result.json") + glob.glob("cases_morph/*_coarse/result.json")):
    case = os.path.dirname(rj); r = json.load(open(rj)); d = r.get("design_id") or os.path.basename(case).split("_")[0]
    r["design_id"] = d; r["speed_kmh"] = r.get("speed_kmh", 100); r["mesh"] = "coarse"; r["status"] = "OK"
    for k, v in SHAPE.items():
        if d in table: r[v] = table[d][k]
    fs = sorted(glob.glob(case + "/postProcessing/forceCoeffs*/*/forceCoeffs.dat") + glob.glob(case + "/postProcessing/forceCoeffs*/*/coefficient.dat"))
    if fs:
        cd = np.array([float(l.split()[1]) for l in open(fs[-1]) if l.strip() and not l.startswith("#")])
        m5, m3, m1 = cd[-500:].mean(), cd[-300:].mean(), cd[-100:].mean()
        r.update(Cd=m5, Cd_mean=m1, Cd_mean500=m5, Cd_last=cd[-1], Cd_mean100=m1, Cd_mean300=m3, Cd_drift_pct=100 * abs(m5 - m3) / m5, Cd_std=cd[-500:].std(), iterations=len(cd))
        ok_res = (r.get("final_res_p") or 1) < 1e-3 and (r.get("final_res_U") or 1) < 1e-3
        r["converged"] = bool(ok_res and r["Cd_drift_pct"] < 1.0)
        if r.get("Cd_sum") is not None and m1: r["Cd_sum_vs_OF_pct"] = 100 * (r["Cd_sum"] - m1) / m1
    if r.get("morph_surface_ok") is not None and r.get("Cd_sum_vs_OF_pct") is not None and d != "D000":
        r["morph_passed"] = bool(r["morph_surface_ok"] and abs(r["Cd_sum_vs_OF_pct"]) < 1.0)
    rows.append(r)
if os.path.exists("run_table.csv"): shutil.copy("run_table.csv", "run_table_old.csv")
with open("run_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow(r)
print("wrote run_table.csv with %d cases:" % len(rows), " ".join(r["design_id"] for r in rows))
