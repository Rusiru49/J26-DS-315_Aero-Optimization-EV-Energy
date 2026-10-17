"""
Generative augmentation: propose new in-range shape combinations and label
each with a physics-consistent Cd via the POD + helper-GP path (never an
invented value from the generator itself).

LHS (Latin Hypercube Sampling) is the only implemented production path, not
a fallback bolted on — the reference spec explicitly flags ~80 real rows as
too few to train CTGAN reliably, and names space-filling sampling as the
designated fallback with no training step needed. CTGAN/SDV would need a
heavy new dependency (drags in torch) for a path the spec itself says is
unlikely to work at this sample size; if it's ever wanted, it belongs in a
one-off experiment for the report's limitations discussion, not here.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

from component3 import config
from component3.workflow1.helper import label_shapes


def propose_shapes(n_shapes=config.N_SYNTHETIC_SHAPES, seed=config.RANDOM_STATE + 10):
    """
    LHS-sample n_shapes new parameter combinations, confined to the real
    design-space box in config.SHAPE_PARAM_BOUNDS — by construction, every
    sample is in-range, so no separate filtering step is needed (unlike a
    generator such as CTGAN, which can propose out-of-box combinations that
    must be discarded).
    """
    names = config.SHAPE_PARAM_NAMES
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit_samples = sampler.random(n=n_shapes)
    bounds_lo = np.array([config.SHAPE_PARAM_BOUNDS[n][0] for n in names])
    bounds_hi = np.array([config.SHAPE_PARAM_BOUNDS[n][1] for n in names])
    scaled = qmc.scale(unit_samples, bounds_lo, bounds_hi)
    return [dict(zip(names, row)) for row in scaled]


def build_augmented_dataset(cfd_dataset, pod_model, gp, n_synthetic=config.N_SYNTHETIC_SHAPES, method=None):
    """
    Combine the real (non-holdout) runs with n_synthetic physics-labelled
    synthetic rows into one "augmented" dataset, tagged by source. Held-out
    real runs are deliberately excluded here — they are reserved for final
    surrogate evaluation only (workflow1/training.py), never for training.

    Returns a DataFrame with columns: the 5 shape params, Cd, source
    ("real" or "synthetic").
    """
    real_runs = cfd_dataset.real_runs()
    real_rows = real_runs[config.SHAPE_PARAM_NAMES + ["Cd"]].copy()
    real_rows["source"] = "real"

    proposed = propose_shapes(n_synthetic)
    labels = label_shapes(gp, pod_model, proposed, method)
    synthetic_rows = [{**params, "Cd": float(cd), "source": "synthetic"} for params, cd in zip(proposed, labels)]
    synthetic_df = pd.DataFrame(synthetic_rows)

    augmented = pd.concat([real_rows, synthetic_df], ignore_index=True)
    return augmented
