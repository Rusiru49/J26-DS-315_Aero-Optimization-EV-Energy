"""Build cfd2d_sedan_standin.ipynb from cfd2d.py (run: python colab/make_notebook.py)."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
module_src = open(os.path.join(HERE, "cfd2d.py"), encoding="utf-8").read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Component 3: 2D CFD stand-in for Member 2's sedan data

This notebook runs **real CFD (OpenFOAM, steady RANS, k-omega SST)** on a **2D side profile** of a sedan with the
same 5 shape parameters as Member 2's car (windshield, spoiler, boat-tail, hood angle, ground clearance), at
100 and 40 km/h, and saves the results in **Member 2's exact file format** to your Google Drive.

It is NOT Member 2's data: it is 2D, so the drag values differ from a real 3D car. It lets us test the
Component 3 pipeline on real flow physics before his data arrives.

**How to use (simple):**
1. Run the cells **one by one from the top** (Shift+Enter). The first cell asks for Google Drive access: allow it.
2. Cell 5 is a **trial** (6 runs, about 10-20 minutes). Send the printed table to Claude before continuing.
3. Cell 7 is the **full run** (142 runs, a few hours). Keep this browser tab open. If Colab disconnects,
   run cells 1, 2, 3, 4 again and then cell 7: it continues where it stopped.
4. Cell 8 makes one small zip file in Google Drive. Download only that file.

**WiFi data:** everything heavy happens in Google's cloud. Only the final zip (a few MB) is downloaded.

---
**සිංහලෙන්:** ඉහළ සිට cells එකින් එක run කරන්න (Shift+Enter). පළමු cell එක Google Drive අවසරය ඉල්ලනවා, ඉඩ දෙන්න.
Cell 5 trial එක (විනාඩි 10-20). එහි table එක Claude ට යවන්න. Cell 7 සම්පූර්ණ run එක (පැය කිහිපයක්), tab එක විවෘතව තියන්න.
Disconnect වුණොත් cells 1, 2, 3, 4 නැවත run කරලා cell 7 run කරන්න, නතර වුණ තැනින් පටන් ගන්නවා.
Cell 8 Google Drive එකේ කුඩා zip ගොනුවක් හදනවා, ඒක විතරක් download කරන්න.
"""),
    md("## Cell 1: Google Drive and folders"),
    code("""
from google.colab import drive
drive.mount('/content/drive')

import os
OUT_DIR = '/content/drive/MyDrive/component3_cfd2d/sedan2d'   # small results files go to your Google Drive
WORK_DIR = '/content/work'                                    # big OpenFOAM folders stay in Colab and are deleted
os.makedirs(OUT_DIR, exist_ok=True)
print('Results folder:', OUT_DIR)
"""),
    md("## Cell 2: Install OpenFOAM (5-10 minutes, in Google's cloud)"),
    code("""
%%bash
if ls /usr/lib/openfoam/openfoam*/etc/bashrc >/dev/null 2>&1; then
  echo "OpenFOAM already installed"
else
  curl -s https://dl.openfoam.com/add-debian-repo.sh | bash > /tmp/repo.log 2>&1 || (tail -20 /tmp/repo.log; exit 1)
  apt-get update -qq > /dev/null 2>&1
  PKG=$(apt-cache search openfoam | grep -oE '^openfoam2[0-9]{3}-default' | sort | tail -1)
  if [ -z "$PKG" ]; then echo "No OpenFOAM package found"; tail -20 /tmp/repo.log; exit 1; fi
  echo "Installing $PKG ..."
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $PKG > /tmp/of_install.log 2>&1 || (tail -30 /tmp/of_install.log; exit 1)
fi
ls -d /usr/lib/openfoam/openfoam*
"""),
    md("## Cell 3: Write the CFD code (no need to read it)"),
    code("%%writefile cfd2d.py\n" + module_src),
    md("## Cell 4: Design table and settings"),
    code("""
import importlib, cfd2d
importlib.reload(cfd2d)
dt = cfd2d.make_design_table()
dt.to_csv(f'{OUT_DIR}/design_table.csv', index=False)
cfd2d.write_case_json(OUT_DIR)
print('OpenFOAM found:', cfd2d.openfoam_bashrc())
print(len(dt), 'designs (D000 baseline + 70 LHS designs)')
print(dt.head())
"""),
    md("## Cell 5: TRIAL (baseline + 2 extreme designs, both speeds = 6 runs, about 10-20 minutes)\nSend the printed table to Claude."),
    code("""
trial = dt.sort_values('run_order')['design_id'].tolist()[:3]
cfd2d.run_many(dt, trial, OUT_DIR, WORK_DIR, workers=2, keep_case=True)
rt = cfd2d.assemble_run_table(OUT_DIR)
cols = ['design_id', 'speed_kmh', 'Cd_mean', 'Cd_std', 'Cd_p', 'Cd_f', 'Cl', 'y_plus_mean', 'final_res_p', 'converged', 'checkMesh_passed', 'cell_count']
print(rt[[c for c in cols if c in rt.columns]].to_string())
"""),
    md("## Cell 6: ONLY if the trial shows FAILED: print the end of the logs (send this to Claude)"),
    code("""
import glob
for log in sorted(glob.glob(f'{WORK_DIR}/*/log.*')):
    print('=====', log)
    print(open(log).read()[-1500:])
"""),
    md("## Cell 6c: Drag history of the trial runs (helps choose the number of iterations)"),
    code("""
import glob, re, os, numpy as np
for case in sorted(glob.glob(f'{WORK_DIR}/*kmh')):
    cd, _ = cfd2d.read_force_history(case)
    if cd is None:
        print(os.path.basename(case), 'no history'); continue
    pts = [250, 500, 750, 1000, 1500, 2000, 2500, 3000]
    print(os.path.basename(case), ' '.join(f'{n}:{cd[min(n, len(cd)) - 1]:.4f}' for n in pts),
          '| std last 300:', round(float(np.std(cd[-300:])), 5), '| residuals p, Ux:', cfd2d.read_residuals(f'{case}/log.simpleFoam'))
"""),
    md("## Cell 6d: ONLY when Claude asks: delete ALL saved results (rows and face files)"),
    code("""
import glob, os
for f in glob.glob(f'{OUT_DIR}/rows/*.json') + glob.glob(f'{OUT_DIR}/faces/*.npz'):
    os.remove(f)
print('All saved results deleted')
"""),
    md("## Cell 6b: ONLY when Claude asks: forget FAILED runs so they run again"),
    code("""
import json, glob, os
for f in glob.glob(f'{OUT_DIR}/rows/*.json'):
    if 'Cd_mean' not in json.load(open(f)):
        os.remove(f); print('removed', f)
"""),
    md("## Cell 7: FULL RUN (142 runs, a few hours; resumes if interrupted)\nKeep this tab open."),
    code("""
all_ids = dt.sort_values('run_order')['design_id'].tolist()
cfd2d.run_many(dt, all_ids, OUT_DIR, WORK_DIR, workers=2)
"""),
    md("## Cell 8: Collect the results into one small zip file in Google Drive"),
    code("""
import shutil
rt = cfd2d.assemble_run_table(OUT_DIR)
print(len(rt), 'runs saved;', int(rt['converged'].fillna(False).astype(bool).sum()), 'converged')
zip_path = shutil.make_archive('/content/drive/MyDrive/component3_cfd2d/sedan2d_results', 'zip', OUT_DIR)
print('Download this one file from Google Drive:', zip_path, f'({os.path.getsize(zip_path) / 1e6:.1f} MB)')
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "cfd2d_sedan_standin.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out)
