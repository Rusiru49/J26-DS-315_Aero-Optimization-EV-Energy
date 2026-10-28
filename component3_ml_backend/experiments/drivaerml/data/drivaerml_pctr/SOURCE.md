# DrivaerML-PCTR files (third-party data)

- Source: DrivaerML-PCTR, https://huggingface.co/datasets/Jrhoss/DrivaerML-PCTR, a processed version of DrivAerML (N. Ashton et al., arXiv:2408.11969, 2024; https://huggingface.co/datasets/neashton/drivaerml). Each run: 100,000 randomly sampled surface cell centres (per-run normalised coordinates) with CpMeanTrim, pMeanTrim, pPrime2MeanTrim and wall-shear-stress magnitude.
- License: CC BY-SA 4.0, inherited from DrivAerML (credit the authors; anything shared that is derived from this data must keep the same license).
- Downloaded 2026-10-13: 80 runs chosen at random (seed 2026) from the 484 available, listed in `chosen_80.txt` (`<split> <run_id>`); files `raw/boundary_<id>.pt`, about 3.6 MB each, 276 MB in total.
- The files are PyTorch pickles. They are read ONLY with `scripts/drivaerml_pt_reader.py`, a restricted reader that allows just the tensor-rebuild function, the storage types and OrderedDict, so no code from the files can run.
- `mapped_fields.npz` is made by `scripts/run_drivaerml_pod.py`: every run's Cp mapped onto 20,000 reference points.
