#!/usr/bin/env python3
"""Run this INSIDE a finished pilot case folder:   python3 ../../check_feasibility.py [--swap-gb 0.3] [--iters 1500]
It reads log.checkMesh, log.snappyHexMesh, log.simpleFoam and tells you if 140 runs are possible on this Mac."""
import argparse, re, os
ap = argparse.ArgumentParser(); ap.add_argument("--swap-gb", type=float, default=None, help="Swap Used from Activity Monitor (GB)")
ap.add_argument("--iters", type=int, default=1500); ap.add_argument("--runs", type=int, default=140); ap.add_argument("--hours-per-night", type=float, default=10)
a = ap.parse_args()
def rd(n): return open(n, errors="ignore").read() if os.path.exists(n) else ""
c, s, m = rd("log.checkMesh"), rd("log.simpleFoam"), rd("log.snappyHexMesh")
cells = re.findall(r"^\s+cells:\s+(\d+)\s*$", c, re.M)      # the mesh-stats line (first match), not 'number of cells: N' of a failed check
cells = int(cells[0]) if cells else None
steps = re.findall(r"^Time = (\d+)", s, re.M); clock = re.findall(r"ClockTime = (\d+) s", s)
mclock = re.findall(r"ClockTime = (\d+) s", m)
print("Cells                  :", cells)
if not steps or not clock: print("Cannot find the solver time in log.simpleFoam - did the run finish?"); raise SystemExit
n, t = int(steps[-1]), int(clock[-1]); spi = t/n
# meshing time = time between the end of blockMesh and the end of snappyHexMesh (file times)
try: mesh_s = int(os.path.getmtime("log.snappyHexMesh") - os.path.getmtime("log.blockMesh"))
except OSError: mesh_s = 0
run_h = (spi*a.iters + mesh_s)/3600.0
total_h = run_h*a.runs
print("Iterations done        :", n, "in", t, "s  ->", round(spi, 2), "s per iteration")
print("Meshing time           :", round(mesh_s/60, 1), "min")
print("One full run (%d it)  : %.2f hours" % (a.iters, run_h))
print("%d runs               : %.0f hours = %.0f nights of %d h" % (a.runs, total_h, total_h/getattr(a,"hours_per_night"), getattr(a,"hours_per_night")))
print()
if a.swap_gb is not None and a.swap_gb > 1.0:
    print("VERDICT: RED - swap > 1 GB, mesh is too big for 8 GB RAM. Use a coarser mesh or half-car."); raise SystemExit
if run_h < 2: v = "GREEN - all designs at both speeds are possible on this Mac."
elif run_h < 4: v = "YELLOW - 100 km/h for all 70 designs, 40 km/h for only some (tell supervisors)."
else: v = "RED - too slow for this Mac. Use half-car, or SLIIT lab / cloud."
print("VERDICT:", v)
if a.swap_gb is None: print("(Also check Activity Monitor: Swap Used must be < 1 GB; re-run with --swap-gb X)")
