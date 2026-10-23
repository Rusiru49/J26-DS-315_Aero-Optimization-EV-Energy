#!/usr/bin/env python3
"""After a case has finished: read Cd, mesh/solver quality, and per-face p / tau on the car surface
and write   <outdir>/sedan_<design>_<kmh>kmh.npz   +   result.json in the case folder.
needs: python3 + numpy
usage: python3 export_case.py cases/D000_100kmh_medium --design D000 --outdir npz
"""
import argparse, glob, json, os, re, sys
import numpy as np

RHO = 1.165

def read_vtk(path):
    tok = open(path).read().split()
    i = 0; pts = None; polys = []; cell = {}; section = None; ncell = None
    n = len(tok)
    def num(k, cnt): return np.array(tok[k:k+cnt], dtype=float)
    while i < n:
        t = tok[i]
        if t == "POINTS":
            npt = int(tok[i+1]); pts = num(i+3, 3*npt).reshape(-1, 3); i += 3 + 3*npt
        elif t == "POLYGONS":
            m = int(tok[i+1]); tot = int(tok[i+2]); j = i+3
            for _ in range(m):
                c = int(tok[j]); polys.append([int(x) for x in tok[j+1:j+1+c]]); j += 1 + c
            i = j
        elif t == "CELL_DATA": section = "cell"; ncell = int(tok[i+1]); i += 2
        elif t == "POINT_DATA": section = "point"; i += 2
        elif t == "FIELD":
            na = int(tok[i+2]); j = i+3
            for _ in range(na):
                name, nc, nt = tok[j], int(tok[j+1]), int(tok[j+2]); j += 4
                vals = num(j, nc*nt).reshape(nt, nc); j += nc*nt
                if section == "cell": cell[name] = vals
            i = j
        elif t in ("SCALARS", "VECTORS"):
            name = tok[i+1]; nc = 3 if t == "VECTORS" else (int(tok[i+3]) if len(tok) > i+3 and tok[i+3].isdigit() else 1)
            j = i + (3 if t == "VECTORS" else 3)
            if t == "SCALARS":
                if tok[j].isdigit(): j += 1
                j += 2     # LOOKUP_TABLE default
            vals = num(j, nc*ncell).reshape(ncell, nc); j += nc*ncell
            if section == "cell": cell[name] = vals
            i = j
        else: i += 1
    return pts, polys, cell

def read_vtp(path):
    """XML PolyData (.vtp, ASCII or uncompressed base64) -> pts, polys, cell dict (same output as read_vtk)."""
    import xml.etree.ElementTree as ET, base64, struct
    root = ET.parse(path).getroot()
    hdr = 8 if root.get("header_type", "UInt32") == "UInt64" else 4
    def arr(da):
        t = da.get("type", "Float64"); nc = int(da.get("NumberOfComponents", "1"))
        dt = {"Float32": "f4", "Float64": "f8", "Int32": "i4", "Int64": "i8", "UInt32": "u4", "UInt64": "u8", "UInt8": "u1"}[t]
        if da.get("format", "ascii") == "ascii":
            v = np.array(da.text.split(), dtype=dt)
        else:
            raw = base64.b64decode(da.text.strip()); n = struct.unpack("<I" if hdr == 4 else "<Q", raw[:hdr])[0]
            v = np.frombuffer(raw[hdr:hdr+n], dtype="<"+dt)
        return v.reshape(-1, nc) if nc > 1 else v
    pc = root.find(".//Piece")
    pts = arr(pc.find("Points/DataArray")).astype(float).reshape(-1, 3)
    polys_el = {d.get("Name"): arr(d) for d in pc.find("Polys").findall("DataArray")}
    conn, off = polys_el["connectivity"], polys_el["offsets"]; polys = []; s = 0
    for o in off: polys.append([int(x) for x in conn[s:o]]); s = int(o)
    cell = {}
    cd = pc.find("CellData")
    if cd is not None:
        for d in cd.findall("DataArray"):
            v = arr(d).astype(float); cell[d.get("Name")] = v.reshape(len(polys), -1)
    return pts, polys, cell

def face_geometry(pts, polys):
    centre = np.zeros((len(polys), 3)); area = np.zeros(len(polys)); normal = np.zeros((len(polys), 3))
    for k, f in enumerate(polys):
        P = pts[f]; c0 = P.mean(axis=0); Av = np.zeros(3); cs = np.zeros(3); At = 0.0
        for a in range(len(f)):
            p1, p2 = P[a], P[(a+1) % len(f)]
            av = 0.5*np.cross(p1-c0, p2-c0); ar = np.linalg.norm(av)
            Av += av; cs += ar*(c0+p1+p2)/3.0; At += ar
        area[k] = np.linalg.norm(Av); normal[k] = Av/max(area[k], 1e-30); centre[k] = cs/max(At, 1e-30)
    return centre, area, normal

def last_float(rx, text, group=1):
    m = re.findall(rx, text); return float(m[-1] if not isinstance(m[-1], tuple) else m[-1][group-1]) if m else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("case"); ap.add_argument("--design", required=True)
    ap.add_argument("--outdir", default="npz"); ap.add_argument("--baseline", default="D000"); ap.add_argument("--avg", type=int, default=500)
    ap.add_argument("--morph", action="store_true", help="case uses the morphed base mesh: car faces correspond to the baseline faces by index (no nearest-point mapping)")
    a = ap.parse_args(); case = a.case.rstrip("/")
    info = dict(l.split(None, 1) for l in open(case+"/case_info.txt").read().splitlines() if l.strip())
    U = float(info["speed_m_s"]); half = info["half_model"].strip() == "True"
    aref_full = float(info["Aref_full"]); q = 0.5*RHO*U*U
    res = {"design_id": a.design, "speed_m_s": U, "mesh": info["mesh"].strip(), "half_model": half}

    # --- Cd / Cl from forceCoeffs (mean of the last `avg` iterations)
    fc = sorted(glob.glob(case+"/postProcessing/forceCoeffs1/*/*.dat"))
    if fc:
        hdr = None; rows = []
        for ln in open(fc[-1]):
            if ln.startswith("#"):
                if "Time" in ln: hdr = ln.lstrip("# ").split()
            elif ln.strip(): rows.append([float(x) for x in ln.split()])
        rows = np.array(rows); col = {h: j for j, h in enumerate(hdr)}
        res["Cd"] = float(rows[-a.avg:, col["Cd"]].mean()); res["Cl"] = float(rows[-a.avg:, col["Cl"]].mean())
        res["Cd_last"] = float(rows[-1, col["Cd"]]); res["Cd_std_lastavg"] = float(rows[-a.avg:, col["Cd"]].std())
        res["iterations"] = int(rows[-1, 0])
        m5 = rows[-500:, col["Cd"]].mean(); m3 = rows[-300:, col["Cd"]].mean(); m1 = rows[-100:, col["Cd"]].mean()
        res["Cd_mean100"] = float(m1); res["Cd_mean300"] = float(m3)
        res["Cd_drift_pct"] = float(100.0 * abs(m5 - m3) / m5)     # mean-stability: last-500 mean vs last-300 mean
    # --- logs: residuals, cells, mesh quality, y+
    def rd(n):
        p = os.path.join(case, n); return open(p, errors="ignore").read() if os.path.exists(p) else ""
    s = rd("log.simpleFoam"); c = rd("log.checkMesh")
    res["final_res_p"] = last_float(r"Solving for p, Initial residual = ([\d.eE+-]+)", s)
    res["final_res_U"] = last_float(r"Solving for Ux, Initial residual = ([\d.eE+-]+)", s)
    res["final_res_k"] = last_float(r"Solving for k, Initial residual = ([\d.eE+-]+)", s)
    res["converged_flag"] = "SIMPLE solution converged" in s
    m0 = re.findall(r"^\s+cells:\s+(\d+)\s*$", c, re.M)       # mesh-stats line, not a failed-check count
    res["cells"] = int(m0[0]) if m0 else 0
    stars = [l for l in c.splitlines() if l.strip().startswith("***")]
    # the extra -allGeometry checks 'concave' and 'small determinant' only warn; every other failure makes the mesh fail
    res["mesh_ok"] = ("Mesh OK" in c) or (len(stars) > 0 and all(("oncave" in l or "determinant" in l) for l in stars))
    nc = re.findall(r"Concave cells.*number of cells:\s*(\d+)", c); nd = re.findall(r"small determinant.*number of cells:\s*(\d+)", c)
    res["n_concave_cells"] = int(nc[-1]) if nc else 0; res["n_smalldet_cells"] = int(nd[-1]) if nd else 0
    res["max_nonortho"] = last_float(r"non-orthogonality Max:\s*([\d.eE+-]+)", c) or last_float(r"Max non-orthogonality = ([\d.eE+-]+)", c)
    res["max_skewness"] = last_float(r"Max skewness = ([\d.eE+-]+)", c)
    m = re.findall(r"patch car\S* y\+ : min = ([\d.eE+-]+), max = ([\d.eE+-]+), average = ([\d.eE+-]+)", s)
    if m: res["yplus_min"], res["yplus_max"], res["yplus_avg"] = (float(x) for x in m[-1])

    # --- per-face data on the car surface
    vtk = sorted(glob.glob(case+"/VTK/**/*.vtk", recursive=True) + glob.glob(case+"/VTK/**/*.vtp", recursive=True), key=lambda p: (len(p), p))
    vtk = [v for v in vtk if "car" in os.path.basename(v).lower() or "car" in os.path.dirname(v).lower()]
    vtk = [v for v in vtk if "internal" not in v.lower()]
    # newest time step last (VTK/<case>_<iter>/...): sort by iteration number found in the path
    vtk.sort(key=lambda p: [int(x) for x in re.findall(r"(\d+)", p)] or [0])
    if vtk:
        pts, polys, cell = (read_vtp if vtk[-1].endswith(".vtp") else read_vtk)(vtk[-1])
        res["vtk_file"] = os.path.relpath(vtk[-1], case)
        centre, area, nrm = face_geometry(pts, polys)
        p = (cell.get("pMean", cell.get("p"))[:, 0])*RHO                        # kinematic -> Pa
        tau = cell.get("wallShearStressMean", cell.get("wallShearStress"))*RHO  # kinematic -> Pa
        tau = -tau            # OpenFOAM(ESI) wallShearStress is the stress ON THE FLUID; flip so tau = shear force per area ON THE CAR (sum(tau*A)_x = friction drag > 0)
        res["tau_convention"] = "force per area exerted by the air on the car surface (Pa)"
        nrm = -nrm            # OpenFOAM patch normals point OUT of the fluid; store normals pointing OUT of the car (into the fluid)
        if half:              # mirror y_of -> -y_of to get the full car
            c2 = centre.copy(); c2[:, 1] *= -1; n2 = nrm.copy(); n2[:, 1] *= -1; t2 = tau.copy(); t2[:, 1] *= -1
            centre = np.vstack([centre, c2]); nrm = np.vstack([nrm, n2]); tau = np.vstack([tau, t2])
            p = np.concatenate([p, p]); area = np.concatenate([area, area])
        # force on the car, in the OpenFOAM frame (x = flow direction, z = up)
        Fp = (-(p*area)[:, None]*nrm).sum(axis=0); Ff = (tau*area[:, None]).sum(axis=0)
        res["Cd_sum"] = float((Fp[0]+Ff[0])/(q*aref_full)); res["Cl_sum"] = float((Fp[2]+Ff[2])/(q*aref_full))
        res["Cd_p"] = float(Fp[0]/(q*aref_full)); res["Cd_f"] = float(Ff[0]/(q*aref_full))
        res["Cl_p"] = float(Fp[2]/(q*aref_full)); res["Cl_f"] = float(Ff[2]/(q*aref_full))
        res["n_faces"] = int(len(area)); res["surface_area_m2"] = float(area.sum())
        ref_cd = res.get("Cd_mean100") or res.get("Cd")   # the per-face fields (fieldAverage) cover the last 100 iterations -> compare with the last-100 mean
        if ref_cd: res["Cd_sum_vs_OF_pct"] = 100.0*(res["Cd_sum"]-ref_cd)/ref_cd   # face-sum Cd vs forceCoeffs mean over the last iterations
        # convert to the CAD (Fusion) frame the Component 3 contract uses: air flows toward -Z, Y up, X lateral, metres
        def cad(v): return np.stack([-v[:, 1], v[:, 2], -v[:, 0]], axis=1)
        centre_c, nrm_c, tau_c = cad(centre), cad(nrm), cad(tau)
        tag = "sedan_%s_%dkmh" % (a.design, round(U*3.6)); os.makedirs(a.outdir, exist_ok=True)
        common = dict(Cd=res.get("Cd_mean", res.get("Cd", np.nan)), Aref=aref_full, U=U, rho=RHO, q_ref=q, p_ref=0.0)
        np.savez_compressed(os.path.join(a.outdir, tag+"_true.npz"), centre=centre_c.astype("f4"), area=area.astype("f4"),
                            normal=nrm_c.astype("f4"), p=p.astype("f4"), tau=tau_c.astype("f4"), **common)
        base_path = os.path.join(a.outdir, "baseline_faces.npz")
        if a.design == a.baseline and not os.path.exists(base_path):
            conn = np.array([len(f) for f in polys])      # polygon sizes; full connectivity saved for plots (A11)
            np.savez_compressed(base_path, centre=centre_c.astype("f4"), area=area.astype("f4"), normal=nrm_c.astype("f4"),
                                points=(cad(pts)).astype("f4"), polys=np.array([i for f in polys for i in f]), poly_sizes=conn,
                                half_model=half)
        if a.morph and os.path.exists(base_path) and a.design != a.baseline:
            B = np.load(base_path); nb = len(B["area"])
            if len(area) != nb:
                res["error"] = "morph: %d car faces but baseline has %d" % (len(area), nb); res["morph_passed"] = False
                res["map_method"] = "morph_index"
            else:
                shift = np.linalg.norm(centre_c - B["centre"].astype(float), axis=1)
                res["map_method"] = "morph_index"; res["morph_face_shift_max_mm"] = float(shift.max()*1000)
                mrep = {}
                if os.path.exists(case+"/morph_report.json"): mrep = json.load(open(case+"/morph_report.json"))
                for k in ("morph_mode", "morph_surface_ok", "err_mean_mm", "err_p99_mm", "err_max_mm", "max_disp_mm"):
                    if k in mrep: res[k if k.startswith("morph_") else "morph_"+k] = mrep[k]
                res["morph_passed"] = bool(shift.max() < 0.5 and mrep.get("morph_surface_ok", False)
                                           and abs(res.get("Cd_sum_vs_OF_pct", 99.0)) < 1.0)
                np.savez_compressed(os.path.join(a.outdir, tag+".npz"), centre=centre_c.astype("f4"), area=area.astype("f4"),
                                    normal=nrm_c.astype("f4"), p=p.astype("f4"), tau=tau_c.astype("f4"), **common)
        elif os.path.exists(base_path) and a.design != a.baseline:
            try:
                from scipy.spatial import cKDTree
                B = np.load(base_path); bc = B["centre"].astype(float)
                dist, idx = cKDTree(centre_c).query(bc)
                pm, tm = p[idx], tau_c[idx]
                bn, ba = B["normal"].astype(float), B["area"].astype(float)
                # same drag-sum check but with the baseline faces (shows the interpolation error)
                Fm = (-(pm*ba)[:, None]*bn).sum(axis=0) + (tm*ba[:, None]).sum(axis=0)
                cdm = float(-Fm[2]/(q*aref_full))      # drag acts toward -Z in the CAD frame
                res["map_method"] = "remesh_mapped_to_baseline"; res["map_mean_dist_mm"] = float(dist.mean()*1000)
                res["map_cd_err_pct"] = 100.0*(cdm-res["Cd_sum"])/res["Cd_sum"]
                res["morph_passed"] = bool(abs(res["map_cd_err_pct"]) < 5.0)
                np.savez_compressed(os.path.join(a.outdir, tag+".npz"), centre=B["centre"], area=B["area"], normal=B["normal"],
                                    p=pm.astype("f4"), tau=tm.astype("f4"), **common)
            except ImportError:
                res["error"] = "scipy missing: pip3 install scipy"
        else:
            res["map_method"] = "baseline_self" if a.design == a.baseline else "no_baseline_yet"
            res["morph_passed"] = a.design == a.baseline
            np.savez_compressed(os.path.join(a.outdir, tag+".npz"), centre=centre_c.astype("f4"), area=area.astype("f4"),
                                normal=nrm_c.astype("f4"), p=p.astype("f4"), tau=tau_c.astype("f4"), **common)
        res["npz"] = os.path.join(a.outdir, tag+".npz")
    else:
        res["error"] = "no VTK surface file found; files in VTK/: " + ", ".join(os.path.relpath(x, case) for x in glob.glob(case+"/VTK/**/*.*", recursive=True)[:12])
    # --- contract columns (Component 3 run_table.csv)
    res["speed_kmh"] = int(round(U*3.6)); res["A_frontal"] = aref_full; res["q_ref"] = q
    # Cd (dataset value) = time-average over the last `avg` iterations (steady RANS on a bluff body oscillates +-2-3 %, so the last-iteration value is NOT used)
    res["Cd_mean"] = res.get("Cd"); res["Cd_std"] = res.get("Cd_std_lastavg")
    res["y_plus_mean"] = res.get("yplus_avg"); res["y_plus_max"] = res.get("yplus_max")
    res["cell_count"] = res.get("cells"); res["checkMesh_passed"] = res.get("mesh_ok")
    ok_res = (res.get("final_res_p") or 1) < 1e-3 and (res.get("final_res_U") or 1) < 1e-3
    ok_cd = res.get("Cd_drift_pct") is not None and res["Cd_drift_pct"] < 1.0      # the running mean must have settled (<1 % between last-300 and last-500 mean)
    res["converged"] = bool(ok_res and ok_cd)
    json.dump({"rho": RHO, "mu": 1.86e-5, "speeds_m_s": [27.78, 11.11], "turbulence": "kOmegaSST", "p_ref": 0.0,
               "avg_last_iterations": a.avg, "half_model_mirrored": half, "frame": "CAD: X lateral, Y up, Z along car, air toward -Z, metres"},
              open(case+"/case.json", "w"), indent=1)
    json.dump(res, open(case+"/result.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

if __name__ == "__main__":
    main()
