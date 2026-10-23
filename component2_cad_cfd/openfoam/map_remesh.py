import glob, json, os, shutil, sys
import numpy as np
from scipy.spatial import cKDTree
args = [a for a in sys.argv[1:] if not a.startswith("--")]
D = args[0] if args else "D019"
WRITE = "--write" in sys.argv
SP = 100
for a in sys.argv[1:]:
    if a.startswith("--speed="): SP = int(a.split("=")[1])
tp = "npz/sedan_%s_%dkmh_true.npz" % (D, SP)
mp_own = "npz_morph/sedan_%s_%dkmh.npz" % (D, SP)
mp_100 = "npz_morph/sedan_%s_100kmh.npz" % D
mp = mp_own if os.path.exists(mp_own) else mp_100
for f in (tp, mp):
    if not os.path.exists(f):
        print(D, "skipped, file missing:", f); sys.exit(0)
T = np.load(tp); M = np.load(mp)
q = float(T["q_ref"]); A = float(T["Aref"])
def cd_of(p, tau, area, nrm):
    F = (-(p*area)[:, None]*nrm).sum(axis=0) + (tau*area[:, None]).sum(axis=0)
    return float(-F[2]/(q*A))
Tc = T["centre"].astype(float); Mc = M["centre"].astype(float)
dist, idx = cKDTree(Tc).query(Mc)
pm, tm = T["p"][idx], T["tau"][idx]
cd_true = cd_of(T["p"].astype(float), T["tau"].astype(float), T["area"].astype(float), T["normal"].astype(float))
cd_map = cd_of(pm.astype(float), tm.astype(float), M["area"].astype(float), M["normal"].astype(float))
cd_err = 100*(cd_map-cd_true)/cd_true
_, j = cKDTree(Mc).query(Tc)
a = T["area"].astype(float)
pb, p0 = pm[j].astype(float), T["p"].astype(float)
tb, t0 = np.linalg.norm(tm[j].astype(float), axis=1), np.linalg.norm(T["tau"].astype(float), axis=1)
p_err = float(np.sqrt((a*(pb-p0)**2).sum()/a.sum())/q)
t_err = float(np.sqrt((a*(tb-t0)**2).sum()/a.sum())/q)
res = dict(map_cd_err_pct=round(float(cd_err), 2), map_p_err_cp=round(p_err, 4), map_tau_err_cp=round(t_err, 5),
           map_dist_mean_mm=round(float(dist.mean()*1000), 1), map_dist_p99_mm=round(float(np.percentile(dist, 99)*1000), 1))
json.dump(res, open("npz/sedan_%s_%dkmh_maperr.json" % (D, SP), "w"))
print(D, SP, "km/h", res)
label = float(T["Cd"])
if SP != 100:
    fs = glob.glob("cases_prod/%s_%dkmh_coarse/postProcessing/**/coefficient.dat" % (D, SP), recursive=True)
    if fs:
        cd = [float(l.split()[1]) for l in open(fs[0]) if l.strip() and not l.startswith("#")]
        label = sum(cd[-100:])/100
        print("label = mean of last 100 iterations = %.4f (solver value in file %.4f)" % (label, float(T["Cd"])))
if WRITE:
    if abs(cd_err) < 5.0:
        pkg = os.path.expanduser("~/Documents/EV_Aero/for_component3/npz")
        for _a in sys.argv[1:]:
            if _a.startswith("--pkg="): pkg = os.path.expanduser(_a.split("=", 1)[1])
        os.makedirs(pkg, exist_ok=True)
        if os.path.exists(mp_own): shutil.copy(mp_own, pkg + "/sedan_%s_%dkmh_morph.npz" % (D, SP))
        np.savez_compressed(pkg + "/sedan_%s_%dkmh.npz" % (D, SP), centre=M["centre"], area=M["area"], normal=M["normal"],
                            p=pm.astype("f4"), tau=tm.astype("f4"), Cd=label, Aref=A, U=float(T["U"]), rho=float(T["rho"]), q_ref=q, p_ref=0.0)
        print("WRITTEN to", pkg)
    else:
        print("NOT WRITTEN: Cd map error above 5 percent")
