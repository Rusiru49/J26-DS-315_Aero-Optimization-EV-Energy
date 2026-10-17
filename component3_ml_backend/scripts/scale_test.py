"""
Size test: does Workflow 1 cope with Member 2's expected file size?

Member 2's files are estimated at 10-20 MB each, i.e. roughly 250,000-450,000
faces. This builds a dummy sedan dataset with N_FACES faces per file (same
format, 70 designs + baseline, two speeds) and times each Workflow 1 step and
its peak Python memory. The physics is the invented dummy, so only time and
memory matter here, not accuracy.

Usage: python scripts/scale_test.py [n_faces]   (default about 260,000)
Writes data/scale_test_sedan/ (git-ignored) and outputs/scale_test.json
"""

import json
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np

from component3 import config
import component3.data.dummy_generator as gen
OUT = os.path.join(config.COMPONENT_ROOT, "data", "scale_test_sedan")


def stage(name, fn, log):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    secs = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    log.append({"stage": name, "seconds": round(secs, 1), "peak_python_MB": round(peak)})
    print(f"  {name:45s} {secs:8.1f} s   peak {peak:8.0f} MB", flush=True)
    return result


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 260_000
    gen.NZ, gen.NTHETA = 400, max(8, target // (2 * 400))
    n_faces = 2 * gen.NZ * gen.NTHETA
    log = []
    print(f"Scale test with {n_faces:,} faces per file")
    if not os.path.exists(os.path.join(OUT, "run_table.csv")):
        stage("make the big dummy dataset (142 files)", lambda: gen.generate_dataset(out_dir=OUT), log)
    size_mb = sum(os.path.getsize(os.path.join(OUT, "faces", f)) for f in os.listdir(os.path.join(OUT, "faces"))) / 1e6
    print(f"  dataset on disk: {size_mb:,.0f} MB ({size_mb / 142:.1f} MB per file)")

    from component3.data.loader import CFDDataset
    from component3.data.checker import check_dataset
    from component3.workflow1.helper import label_shapes, select_labeller, train_helper_gp
    from component3.workflow1.augmentation import propose_shapes
    from component3.workflow1.pod import fit_pod
    from component3.workflow1.training import train_and_select

    stage("data checker (all files)", lambda: check_dataset(OUT), log)
    ds = stage("load the dataset (100 km/h)", lambda: CFDDataset(OUT, 100), log)
    pod = stage("POD + geometry model + calibrations", lambda: fit_pod(ds), log)
    choice = stage("choose the labeller (5-fold CV)", lambda: select_labeller(ds), log)
    (kind, method), _ = choice
    gp = stage(f"helper GP ({kind})", lambda: train_helper_gp(ds, pod, kind), log)
    shapes = propose_shapes(config.N_SYNTHETIC_SHAPES)
    labels = stage(f"label 1,200 synthetic shapes ({method})", lambda: label_shapes(gp, pod, shapes, method), log)

    import pandas as pd
    real = ds.real_runs()[config.SHAPE_PARAM_NAMES + ["Cd"]].assign(source="real")
    syn = pd.DataFrame(shapes).assign(Cd=labels, source="synthetic")
    aug = pd.concat([real, syn], ignore_index=True)
    stage("train XGBoost / RF / ANN", lambda: train_and_select(aug, ds.holdout_runs()), log)

    t1 = time.perf_counter()
    stage("one live drag prediction (deployed model)", lambda: None, log)
    total = sum(r["seconds"] for r in log if not r["stage"].startswith("make"))
    print(f"\nWorkflow 1 steps total (excluding making the data): {total:.0f} s")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUTS_DIR, "scale_test.json"), "w") as fh:
        json.dump({"n_faces": n_faces, "dataset_MB": size_mb, "stages": log, "total_seconds": total}, fh, indent=2)


if __name__ == "__main__":
    main()
