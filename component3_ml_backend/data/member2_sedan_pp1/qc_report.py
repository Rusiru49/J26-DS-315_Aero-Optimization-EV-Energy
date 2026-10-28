"""Quality-control summary of run_table.csv (Component 2).  python3 qc_report.py [run_table.csv] [design_table_sedan_v5.csv]
Prints one line per finished design and the flags to look at; writes qc_summary.csv."""
import csv, sys, numpy as np
rt = sys.argv[1] if len(sys.argv) > 1 else "run_table.csv"; dt = sys.argv[2] if len(sys.argv) > 2 else "design_table_sedan_v5.csv"
tab = {r["design_id"]: r for r in csv.DictReader(open(dt))}
last = {}
for r in csv.DictReader(open(rt)):
    last[(r["design_id"], r["speed_kmh"])] = r                       # newest row wins
f = lambda x: float(x) if x not in (None, "", "None") else np.nan
out = []; allcd = []
for (d, sp), r in sorted(last.items()):
    flags = []
    if not r["status"].startswith("OK"): flags.append("RUN FAILED")
    if r.get("converged") != "True": flags.append("not converged (mean drift/residuals)")
    if f(r.get("Cd_drift_pct")) > 1.0: flags.append("Cd drift %.1f%%" % f(r["Cd_drift_pct"]))
    if f(r.get("Cd_std")) / f(r.get("Cd_mean")) > 0.04: flags.append("large Cd oscillation")
    if abs(f(r.get("Cd_sum_vs_OF_pct"))) > 1.0: flags.append("face-sum Cd mismatch")
    if r.get("morph_mode") and r.get("morph_passed") != "True": flags.append("morph check failed")
    if f(r.get("morph_err_p99_mm")) > 5.0: flags.append("surface err p99 %.1f mm" % f(r["morph_err_p99_mm"]))
    if f(r.get("max_nonortho")) > 80: flags.append("non-orth %.0f" % f(r["max_nonortho"]))
    if d in tab and d != "D000":
        t = tab[d]; dws = abs(float(t["WindshieldAngle"]) - 28); dh = abs(float(t["HoodAngle"]) - 10)
        if dws > 3 or dh > 3: flags.append("windshield/hood change (dWS %.1f, dHood %.1f): flow-sensitive region" % (dws, dh))
    cd = f(r.get("Cd_mean")); allcd.append(cd)
    out.append([d, sp, "%.4f" % cd, "%.2f" % f(r.get("Cd_drift_pct")), r.get("converged"), r.get("wall_minutes"), "; ".join(flags)])
    print("%-5s %4s km/h  Cd %.4f  drift %.2f%%  conv %-5s  %s min  %s" % (d, sp, cd, f(r.get("Cd_drift_pct")), r.get("converged"), r.get("wall_minutes"), "; ".join(flags)))
a = np.array(allcd); print("\n%d designs; Cd min %.4f  mean %.4f  max %.4f" % (len(a), np.nanmin(a), np.nanmean(a), np.nanmax(a)))
with open("qc_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["design_id", "speed_kmh", "Cd", "Cd_drift_pct", "converged", "wall_min", "flags"]); w.writerows(out)
print("wrote qc_summary.csv")
