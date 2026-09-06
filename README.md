# Driving Cycle Construction - Prototype

A working, runnable prototype of your methodology, using **synthetic** GPS +
OBD-II data so you can validate the pipeline structure before you have real
vehicle logs.

```
python3 main.py
```

This regenerates synthetic data, runs all 10 pipeline stages, and writes
results to `outputs/` and `figures/`.

## Files

| File | Purpose |
|---|---|
| `synthetic_data.py` | Generates raw GPS (1 Hz) + OBD-II (5 Hz) logs for 5 road classes, with realistic noise, GPS dropout, and sensor jitter |
| `pipeline.py` | All 10 pipeline stages as standalone functions |
| `main.py` | Orchestrates the full run + produces validation plots |
| `outputs/` | Feature matrix, representative segments, constructed cycles (CSV) |
| `figures/` | Constructed cycle plots, cluster feature-space plot, classification confusion matrix |

## Pipeline stage → code mapping

1. **Data Synchronization** — `pipeline.sync_gps_obd()` — `pandas.merge_asof`
   aligns the 1 Hz GPS stream and 5 Hz OBD stream onto one timeline (nearest
   timestamp, 0.5 s tolerance).
2. **Data Preprocessing** — `pipeline.preprocess()` — drops physically
   implausible speed jumps (>5 m/s² accel), smooths with Savitzky-Golay,
   recomputes acceleration/jerk.
3. **Route Identification** — `pipeline.identify_routes()` — placeholder for
   map-matching (real version: OSMnx + a Hidden Markov map-matcher against
   OpenStreetMap). Currently just chunks each trip into ~60 s route segments.
4. **Road Environment Classification** — `pipeline.classify_road_env()` —
   rule-based on observed mean speed + stop density per segment. **This is
   the weakest link in the prototype** (see "Known limitation" below) and is
   exactly where real road-tag data (OSM `highway=*` tags) will help most.
5. **Adaptive Segmentation** — `pipeline.adaptive_segmentation()` — splits
   each trip into micro-trips at idle points; caps long high-speed segments
   so a single feature vector doesn't span mixed behavior.
6. **Feature Extraction** — `pipeline.extract_features()` — 14 standard
   kinematic features per segment (avg/max/std speed, accel/decel stats,
   idle %, stops/km, etc.) — see `pipeline.FEATURE_COLS`.
7. **K-Means Clustering** — `pipeline.cluster_segments()` — run
   independently within each road class; k chosen automatically via
   silhouette score.
8. **Representative Segments** — `pipeline.select_representative_segments()`
   — picks the segment closest to each cluster centroid (medoid).
9. **Cycle Construction** — `pipeline.construct_cycle()` — concatenates
   representative segments (repeated in proportion to cluster size) with a
   short linear blend at splice points to avoid unrealistic speed jumps.
10. **Statistical Validation** — `pipeline.validate_cycle()` — compares the
    constructed cycle's avg speed, max speed, idle %, and avg acceleration
    against the full segment pool for that road class, with a 10% relative
    error pass/fail flag (standard driving-cycle acceptance criterion).

## Known limitation (by design, worth noting in your methodology writeup)

The Stage 4 classifier is a deliberately simple baseline (mean speed + stop
density thresholds). On synthetic data it reaches ~50% agreement with ground
truth — mainly because "rural" and "suburban/urban" speed distributions
overlap by construction. This is not a bug to silently fix by tuning
thresholds to my synthetic profiles (that would just overfit to fake data and
tell you nothing about the real world). It demonstrates precisely why your
real pipeline needs genuine **Route Identification** — map-matching to OSM
road tags (`highway=residential/trunk/motorway`, speed limits, etc.) — rather
than inferring road class purely from kinematics. Swap in real map-matching
here and this stage should become far more reliable.

## Swapping in real data

Replace the Stage-0 synthetic generation with your real logs, keeping the
same schema:

- GPS: `trip_id, timestamp, lat, lon, gps_speed_kmh`
- OBD-II: `trip_id, timestamp, obd_speed_kmh, rpm, throttle_pct, engine_load_pct`

Everything from `sync_gps_obd()` onward runs unchanged.

## Tuning knobs worth exploring next

- `adaptive_segmentation()`: `min_segment_s`, idle detection threshold, and
  the max-length cap per road class — currently fixed constants, could be
  made road-class-adaptive per your diagram's "Adaptive Segmentation" intent.
- `cluster_segments()`: `k_range` and whether to cluster per-road-class
  (current) vs. globally then filter.
- `construct_cycle()`: currently caps segment repetition at 3x and uses a
  fixed smoothing window — both are reasonable literature defaults but not
  optimized against any real target cycle length (e.g. targeting exactly
  1200s / 20min per class, as many driving-cycle studies do).
