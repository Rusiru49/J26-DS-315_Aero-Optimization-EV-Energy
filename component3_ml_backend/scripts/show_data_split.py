"""
Training and test data sizes of Workflow 1, read from the saved run (no retraining).

Usage:  python scripts/show_data_split.py [dataset folder name]     (default: member2_sedan_pp1)

Shows:
  - the real CFD data (designs, speeds, runs) and which designs can be tested;
  - one cross-validation fold: real training runs, synthetic training rows, real test runs;
  - all folds: how many designs each fold tests;
  - the final model's training set.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import pandas as pd

from component3 import config


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "member2_sedan_pp1"
    card = json.load(open(os.path.join(config.MODELS_DIR, tag, "model_card.json")))
    speeds = card["speeds_kmh"]
    n_designs = card["n_designs"][str(speeds[0])]
    n_real = sum(card["n_designs"].values())
    preds = {s: pd.read_csv(os.path.join(config.OUTPUTS_DIR, f"workflow1_{tag}_{s}kmh_predictions.csv")) for s in speeds}
    summary = {s: json.load(open(os.path.join(config.OUTPUTS_DIR, f"workflow1_{tag}_{s}kmh.json"))) for s in speeds}
    oof = preds[speeds[0]]
    n_testable = oof["run_id"].nunique()
    n_repeats, n_folds = oof["repeat"].nunique(), oof["fold"].nunique()
    per_fold = oof.groupby(["repeat", "fold"])["run_id"].nunique()

    print(f"=== Workflow 1 data sizes: {tag} ===\n")
    print("1. Real CFD data (Member 2)")
    print(f"   {n_designs} designs x {len(speeds)} speeds ({', '.join(f'{s} km/h' for s in speeds)}) = {n_real} runs")
    print(f"   testable designs: {n_testable}  (the other {n_designs - n_testable} - baseline, extreme corners, "
          f"low-confidence morphs - always train)\n")

    print(f"2. Cross-validation by design: {n_folds} groups x {n_repeats} repeats = {n_folds * n_repeats} folds")
    print("   each fold, a group of designs is hidden at BOTH speeds and used only as the test set\n")
    print(f"   {'':24s}{'train (real)':>14s}{'train (synthetic)':>19s}{'test (real only)':>18s}")
    for t in sorted(per_fold.unique()):
        test_runs = t * len(speeds)
        synth = sum(round(summary[s]["synthetic_label_checks"]["kept"] * summary[s]["synthetic_label_checks"]["n_proposed"])
                    for s in speeds)
        print(f"   fold testing {t} designs  {n_real - test_runs:>10d} runs{synth:>14,d} rows{test_runs:>12d} runs")
    print("   (synthetic rows: about this many per fold, after the per-label checks; they are labelled by a")
    print("    labeller fitted on that fold's training designs only)\n")

    print("3. Over all folds")
    print(f"   designs tested per fold: {', '.join(str(v) for v in per_fold.to_numpy())}")
    print(f"   every testable design is tested once per repeat: {n_testable} designs x {len(speeds)} speeds = "
          f"{n_testable * len(speeds)} test runs per repeat, {n_testable * len(speeds) * n_repeats} test predictions "
          f"in total\n")

    kept = card.get("synthetic_kept_per_speed", {})
    n_synth = sum(kept.values()) if card.get("uses_augmentation") else 0
    print(f"4. Final model ({card['drag_model']}), retrained after the choice")
    print(f"   real CFD runs:        {n_real}")
    for s in speeds:
        if card.get("uses_augmentation"):
            print(f"   synthetic, {s:>3d} km/h:  {kept[str(s)]:,} of {card['n_synthetic_per_speed']:,} passed the checks")
    print(f"   total training rows:  {n_real + n_synth:,}")
    print("   test set: none left - its expected error is the cross-validated error above "
          "(the frozen copy is kept for a blind test on new shapes)")


if __name__ == "__main__":
    main()
