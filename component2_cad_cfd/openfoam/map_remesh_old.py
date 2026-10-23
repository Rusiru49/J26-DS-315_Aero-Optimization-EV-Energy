import json, os, shutil, sys
import numpy as np
from scipy.spatial import cKDTree
args = [a for a in sys.argv[1:] if not a.startswith("--")]
D = args[0] if args else "D019"
WRITE = "--write" in sys.argv
tp = "npz/sedan_%s_100kmh_true.npz" % D; mp = "npz_morph/sedan_%s_100kmh.npz" % D
for f in (tp, mp):
    if not os.path.exists(f):
        print(D, "skipped, file missing:", f); sys.exit(0)
T = np.load(tp); M = np.load(mp)
q = float(T["q_ref"]); A = float(T["Aref"])
def cd_of(p, tau, area, nrm):
    F = (-(p*area)[:, None]*nrm).sum(axis=0) + (tau*area[:, None]).sum(axis=0)
    return float(-F[2]/(q*A))
Tc = T["centre"].astype(float); Mc = M["centre"].astype(float)
dist, idx = cKDTree(Tc).query(Mc)                 # forward: each baseline-topology face takes the nearest re-mesh face
pm, tm = T["p"][idx], T["tau"][idx]
cd_true = cd_of(T["p"].astype(float), T["tau"].astype(float), T["area"].astype(float), T["normal"].astype(float))
cd_map = cd_of(pm.astype(float), tm.astype(float), M["area"].astype(float), M["normal"].astype(float))
cd_err = 100*(cd_map-cd_true)/cd_true
_, j = cKDTree(Mc).query(Tc)                      # backward: evaluate the mapped field at the re-mesh face centres
a = T["area"].astype(float)
pb, p0 = pm[j].astype(float), T["p"].astype(float)
tb, t0 = np.linalg.norm(tm[j].astype(float), axis=1), np.linalg.norm(T["tau"].astype(float), axis=1)
p_err = float(np.sqrt((a*(pb-p0)**2).sum()/a.sum())/q)
t_err = float(np.sqrt((a*(tb-t0)**2).sum()/a.sum())/q)
res = dict(map_cd_err_pct=round(float(cd_err), 2), map_p_err_cp=round(p_err, 4), map_tau_err_cp=round(t_err, 5),
           map_dist_mean_mm=round(float(dist.mean()*1000), 1), map_dist_p99_mm=round(float(np.percentile(dist, 99)*1000), 1))
json.dump(res, open("npz/sedan_%s_100kmh_maperr.json" % D, "w"))
print(D, res)
if WRITE:
    if abs(cd_err) < 5.0:
        pkg = os.path.expanduser("~/Documents/EV_Aero/for_component3/npz")
        shutil.copy(mp, pkg + "/sedan_%s_100kmh_morph.npz" % D)
        np.savez_compressed(pkg + "/sedan_%s_100kmh.npz" % D, centre=M["centre"], area=M["area"], normal=M["normal"],
                            p=pm.astype("f4"), tau=tm.astype("f4"), Cd=float(T["Cd"]), Aref=A, U=float(T["U"]), rho=float(T["rho"]), q_ref=q, p_ref=0.0)
        print("WRITTEN to", pkg)
    else:
        print("NOT WRITTEN: Cd map error above 5 percent")
