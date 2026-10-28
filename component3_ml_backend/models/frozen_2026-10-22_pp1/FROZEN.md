# Frozen drag model, PP1 (2026-10-22)

This is a copy of `models/member2_sedan_pp1/` as presented at PP1. **Do not retrain or overwrite it.**

- Trained on Member 2's PP1 delivery: 30 sedan shapes x 2 speeds (40, 100 km/h).
- Model: XGBoost + physics-labelled augmentation; inputs = 5 shape values + speed in km/h.
- Cross-validated error: 9.6 drag counts at 40 km/h, 8.9 at 100 km/h (see model_card.json). Synthetic labels pass per-label physics checks and carry a calibrated uncertainty.

**Purpose: an independent (blind) test.** When Member 2 delivers NEW shapes (not among the 30), predict
their Cd with THIS model before any retraining (inputs: the 5 shape values + speed_kmh, in the order of
`model_card.json`), compare with the CFD `Cd_label`, and report the mean absolute error on the new
shapes. That is the honest, independent accuracy; if it is close to 9-10 drag counts, the
cross-validation estimate is confirmed. (A small script for this is still to be written.)

Known from the sanity check (`outputs/logs/model_sanity_check.txt`): in a few far corners of the design
box, away from every CFD design, the model predicts Cd slightly outside the CFD range (down to about
0.297). Shapes the optimiser picks in such corners need a confirmation CFD run.
