"""
Plot the physics labeller's predicted surface pressure for held-out designs
next to the real CFD pressure (what a direct drag model cannot give).

Usage: python scripts/plot_pressure_prediction.py [data_dir] [speed_kmh]
Writes outputs/pressure_prediction_<folder>.png
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from component3 import config
from component3.data.loader import load_cfd_dataset
from component3.workflow1.helper import _params_to_array, select_labeller, train_helper_gp
from component3.workflow1.pod import fit_pod, shape_features


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.COMPONENT_ROOT, "data", "cfd2d_sedan")
    speed = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ds = load_cfd_dataset(source, speed)
    (kind, method, n_modes), _ = select_labeller(ds)
    pod = fit_pod(ds, n_modes)
    gp = train_helper_gp(ds, pod, kind)
    hold = ds.holdout_runs().sort_values("Cd")
    picks = [hold.iloc[0], hold.iloc[len(hold) // 2], hold.iloc[-1]]

    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 3.2 * len(picks)), sharex=True)
    for ax, row in zip(axes, picks):
        faces = ds.get_faces(row["run_id"])
        true_cp = ds.get_field(row["run_id"])
        amps = gp.predict(_params_to_array(row)[None, :])
        pred_cp = pod.reconstruct_field(amps[0])
        pred_cd = pod.cd_batch(amps, shape_features(row)[None, :], method)[0]
        idx = np.arange(len(true_cp))
        ax.plot(idx, true_cp, "k-", lw=1.4, label="real CFD")
        ax.plot(idx, pred_cp, "r--", lw=1.2, label="predicted from the 5 shape values")
        ax.set_ylabel("Cp")
        ax.set_title(f"{row['run_id']} (not used in training): Cd real {row['Cd']:.3f}, labeller {pred_cd:.3f}  "
                     f"({abs(pred_cd - row['Cd']) * 1000:.1f} drag counts)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    axes[-1].set_xlabel("face number around the car (nose -> roof -> rear -> underbody)")
    fig.suptitle(f"Surface pressure on held-out designs: real CFD vs physics labeller ({kind} helper, {method} drag)")
    plt.tight_layout()
    out = os.path.join(config.OUTPUTS_DIR, f"pressure_prediction_{os.path.basename(os.path.normpath(source))}.png")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    plt.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
