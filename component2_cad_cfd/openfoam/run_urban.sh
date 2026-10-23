#!/bin/bash
# usage:  bash run_urban.sh <1|2|3>      (urban = 40 km/h). Safe to stop (Ctrl+C) and start again: finished designs are skipped.
COMMON="--table design_table_sedan_v5.csv --stl-dir /Users/chanuka/Documents/EV_Aero_sedan_cad --mesh coarse --np 4 --iters 1500 --full"
MORPH="--cases cases_morph --npz npz_morph --morph-base cases_prod/D000_100kmh_coarse --clean"
case "$1" in
 1) echo "Urban step 1: 10 morph designs at 40 km/h, then D036 re-mesh at 40 km/h"
    time caffeinate -i python3 run_all.py $COMMON $MORPH --designs D003,D004,D005,D008,D012,D015,D037,D038,D039,D059 --speeds 11.11
    time caffeinate -i python3 run_all.py $COMMON --cases cases_prod --npz npz --clean --designs D036 --speeds 11.11 ;;
 2) echo "Urban step 2: 7 new designs at 100 and 40 km/h"
    time caffeinate -i python3 run_all.py $COMMON $MORPH --designs D061,D043,D033,D068,D026,D031,D021 --speeds 27.78,11.11 ;;
 3) echo "Optional step 3: 6 more designs at 100 and 40 km/h"
    time caffeinate -i python3 run_all.py $COMMON $MORPH --designs D045,D064,D034,D065,D058,D023 --speeds 27.78,11.11 ;;
 *) echo "usage: bash run_urban.sh <1|2|3>"; exit 1 ;;
esac
echo; echo "Step $1 finished. Now send me the output of:  python3 rebuild_run_table.py && python3 qc_report.py"
