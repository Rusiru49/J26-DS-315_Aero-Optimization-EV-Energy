import glob, os, re
rows = []
for f in sorted(glob.glob("cases_morph/*_coarse/log.checkMesh")):
    d = os.path.basename(os.path.dirname(f)).split("_")[0]
    t = open(f, errors="ignore").read()
    m = re.findall(r"non-orthogonality Max:\s*([\d.eE+-]+)", t)
    mx = float(m[-1]) if m else float("nan")
    m = re.findall(r"severely non-orthogonal \(> ?70 degrees\) faces:\s*(\d+)", t)
    n70 = int(m[-1]) if m else 0
    flag = (n70 > 50) or (mx > 72)
    rows.append((d, mx, n70, flag))
print("design  maxNonOrtho  faces>70deg  decision")
bad = []
for d, mx, n70, flag in rows:
    print("%-7s %-12.1f %-12d %s" % (d, mx, n70, "RE-MESH" if flag else "ok"))
    if flag: bad.append(d)
if bad:
    print("\nRe-mesh these (one at a time, ~27 min each):")
    print("time caffeinate -i python3 run_all.py --table design_table_sedan_v5.csv --stl-dir /Users/chanuka/Documents/EV_Aero_sedan_cad --designs %s --speeds 27.78 --mesh coarse --np 4 --iters 1500 --full --cases cases_prod --npz npz --clean" % ",".join(bad))
else:
    print("\nAll designs OK.")
