import sys
p = sys.argv[1] if len(sys.argv) > 1 else "rebuild_run_table.py"
t = open(p).read()
if "scenario" in t: print("already patched"); sys.exit(0)
reps = [
 ("import csv, glob, json, os, shutil, sys", "import csv, glob, json, os, re, shutil, sys"),
 ('"half_model","wall_minutes","status"]', '"half_model","wall_minutes","status","scenario"]'),
 ('sorted(glob.glob("cases_prod/D000_100kmh_coarse/result.json") + glob.glob("cases_morph/*_coarse/result.json"))',
  'sorted(glob.glob("cases_prod/D000_*kmh_coarse/result.json") + glob.glob("cases_prod/D019_40kmh_coarse/result.json") + glob.glob("cases_morph/*_coarse/result.json"))'),
 ('r["design_id"] = d; r["speed_kmh"] = r.get("speed_kmh", 100); r["mesh"] = "coarse"; r["status"] = "OK"',
  'm_sp = re.search(r"_(\\d+)kmh_", case); r["design_id"] = d; r["speed_kmh"] = int(m_sp.group(1)) if m_sp else 100; r["mesh"] = "coarse"; r["status"] = "OK"\n'
  '    r["scenario"] = "highway" if r["speed_kmh"] == 100 else "urban"\n'
  '    if d != "D000" and case.startswith("cases_prod"):\n'
  '        for k in ("map_cd_err_pct", "map_method", "map_mean_dist_mm", "morph_mode"): r.pop(k, None)'),
]
for a, b in reps:
    if a not in t: print("PATTERN NOT FOUND:", a[:70]); sys.exit(1)
    t = t.replace(a, b)
open(p, "w").write(t); print("patched", p)
