"""
Phase 4 deliverable: fit POD + the helper GP on the placeholder dataset,
then validate both against held-out placeholder runs.

Usage:
    python scripts/run_pod_validation.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np

from component3 import config
from component3.data.loader import load_cfd_dataset
from component3.workflow1.pod import fit_pod, validate_on_holdout, explained_variance_ratio
from component3.workflow1.helper import train_helper_gp, label_shape


def main():
    dataset = load_cfd_dataset(source="placeholder")
    print(f"Loaded {len(dataset.real_runs())} real (training) runs, "
          f"{len(dataset.holdout_runs())} held out.\n")

    pod_model = fit_pod(dataset)
    evr = explained_variance_ratio(dataset, pod_model)
    print(f"POD: {config.N_POD_MODES} modes explain {evr * 100:.1f}% of field variance "
          f"in the training runs.\n")

    records, summary = validate_on_holdout(dataset, pod_model)
    print("POD-only reconstruction on held-out runs (field -> project -> reconstruct Cd):")
    print(f"  mean |error| = {summary['mean_abs_error']:.5f}, "
          f"max |error| = {summary['max_abs_error']:.5f}  (Cd units; 0.001 = 1 drag count)\n")

    gp = train_helper_gp(dataset, pod_model)
    holdout = dataset.holdout_runs()
    gp_errors = []
    for _, row in holdout.iterrows():
        params = {n: row[n] for n in config.SHAPE_PARAM_NAMES}
        predicted_cd = label_shape(gp, pod_model, params)
        gp_errors.append(abs(predicted_cd - row["Cd"]))
    gp_errors = np.array(gp_errors)
    print("Full path (5 params -> GP -> POD reconstruction -> Cd) on held-out runs:")
    print(f"  mean |error| = {gp_errors.mean():.5f}, max |error| = {gp_errors.max():.5f}\n")

    print("Note: this validates pipeline correctness against the placeholder's own "
          "ground truth, NOT real drag accuracy — that validation happens in Phase 9 "
          "once Member 2 delivers real CFD data.")


if __name__ == "__main__":
    main()
