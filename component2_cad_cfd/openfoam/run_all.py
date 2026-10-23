#!/usr/bin/env python3
"""Run many designs one after another (resumable). Start it INSIDE an OpenFOAM terminal.
usage examples
  python3 run_all.py --table design_table_sedan_v5.csv --stl-dir EV_Aero_sedan_cad --designs D000 --speeds 27.78 --mesh medium
  python3 run_all.py --table design_table_sedan_v5.csv --stl-dir EV_Aero_sedan_cad --speeds 27.78 --np 6 --clean
"""
import argparse, csv, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True); ap.add_argument("--stl-dir", required=True)
ap.add_argument("--designs", default="all"); ap.add_argument("--speeds", default="27.78,11.11")
ap.add_argument("--mesh", default="medium"); ap.add_argument("--np", type=int, default=6)
ap.add_argument("--iters", type=int, default=1500); ap.add_argument("--cases", default="cases"); ap.add_argument("--npz", default="npz")
ap.add_argument("--clean", action="store_true", help="delete mesh + heavy files after a successful export")
ap.add_argument("--full", action="store_true")
ap.add_argument("--morph-base", default=None, help="finished D000 case folder: every other design reuses ITS mesh, morphed (no snappyHexMesh)")
ap.add_argument("--base-design", default="D000"); ap.add_argument("--radius", default="1.2"); ap.add_argument("--ground-ramp", default="0.09")
a = ap.parse_args()
rows = sorted(csv.DictReader(open(a.table)), key=lambda r: int(r["run_order"]))
if a.designs != "all": rows = [r for r in rows if r["design_id"] in a.designs.split(",")]
speeds = [float(s) for s in a.speeds.split(",")]
log = "run_table.csv"; new = not os.path.exists(log)
SHAPE = {"WindshieldAngle":"windshield_angle_deg","SpoilerAngle":"spoiler_angle_deg","BoatTailAngle":"boat_tail_angle_deg",
         "HoodAngle":"hood_angle_deg","GroundClearance":"ground_clearance_mm"}
keys = ["design_id","speed_kmh","windshield_angle_deg","spoiler_angle_deg","boat_tail_angle_deg","hood_angle_deg","ground_clearance_mm",
        "Cd","Cd_mean","Cd_last","Cd_mean100","Cd_mean300","Cd_drift_pct","Cd_std","Cd_p","Cd_f","Cl","Cl_p","Cl_f","A_frontal","q_ref","iterations","converged","y_plus_mean","y_plus_max",
        "cell_count","checkMesh_passed","morph_passed","final_res_p","final_res_U","Cd_sum","Cd_sum_vs_OF_pct","map_method","map_mean_dist_mm",
        "map_cd_err_pct","morph_mode","morph_err_mean_mm","morph_err_p99_mm","morph_max_disp_mm","morph_face_shift_max_mm",
        "max_nonortho","max_skewness","n_faces","mesh","half_model","wall_minutes","status"]
if os.path.exists(log) and open(log).readline().strip() != ",".join(keys):        # an older run_table.csv with different columns: keep it aside, start a clean one
    import shutil; shutil.move(log, "run_table_old_%d.csv" % int(time.time())); new = True
f = open(log, "a", newline=""); wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
if new: wr.writeheader()
for r in rows:
    for U in speeds:
        d = r["design_id"]; name = "%s_%dkmh_%s" % (d, round(U*3.6), a.mesh); case = os.path.join(a.cases, name)
        if a.morph_base and d == a.base_design:
            print("skip (this is the base design, already run):", name); continue
        if os.path.exists(os.path.join(a.npz, "sedan_%s_%dkmh.npz" % (d, round(U*3.6)))) and (a.mesh == "medium" or a.morph_base):
            print("skip (done):", name); continue
        t0 = time.time(); print("=== %s  (run_order %s)" % (name, r["run_order"]), flush=True)
        cmd = [sys.executable, os.path.join(HERE, "make_case.py"), "--stl", os.path.join(a.stl_dir, "sedan_%s.stl" % d), "--design", d,
               "--speed", str(U), "--aref", r["A_frontal_m2"], "--mesh", a.mesh, "--np", str(a.np), "--out", a.cases, "--iters", str(a.iters)]
        if a.full: cmd.append("--full")
        res = {"design_id": d, "speed_kmh": round(U*3.6), "mesh": a.mesh}
        res.update({SHAPE[k]: r[k] for k in SHAPE})
        try:
            if a.morph_base and os.path.isdir(case):
                import shutil; shutil.rmtree(case)          # never mix files of an older attempt with a morph case
            subprocess.run(cmd, check=True)
            exp = [sys.executable, os.path.join(HERE, "export_case.py"), case, "--design", d, "--outdir", a.npz]
            if a.morph_base:
                subprocess.run([sys.executable, os.path.join(HERE, "morph_case.py"), "--base-case", a.morph_base,
                                "--base-stl", os.path.join(a.stl_dir, "sedan_%s.stl" % a.base_design),
                                "--target-stl", os.path.join(a.stl_dir, "sedan_%s.stl" % d), "--case", case, "--np", str(a.np),
                                "--radius", a.radius, "--ground-ramp", a.ground_ramp], check=True)
                exp.append("--morph")
            subprocess.run(["bash", os.path.join(case, "Allrun"), str(a.np)], check=True)
            out = subprocess.run(exp, check=True, capture_output=True, text=True).stdout
            res.update(json.loads(out)); res["status"] = "OK"
            res.update({SHAPE[k]: r[k] for k in SHAPE})
            if a.clean:
                subprocess.run("rm -rf %s/constant/polyMesh %s/[1-9]* %s/VTK %s/processor*" % (case, case, case, case), shell=True)
        except Exception as e:
            res["status"] = "FAILED: %s" % e
        res["wall_minutes"] = round((time.time()-t0)/60, 1)
        wr.writerow(res); f.flush(); print(res.get("status"), res.get("Cd"), res["wall_minutes"], "min", flush=True)
