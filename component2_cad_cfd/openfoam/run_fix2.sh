#!/bin/bash
COMMON="--table design_table_sedan_v5.csv --stl-dir /Users/chanuka/Documents/EV_Aero_sedan_cad --mesh coarse --np 4 --iters 1500 --full"
mkdir -p ~/Documents/EV_Aero/backup/D012_old100
mv cases_morph/D012_100kmh_coarse ~/Documents/EV_Aero/backup/D012_old100/case_old
mv npz_morph/sedan_D012_100kmh.npz ~/Documents/EV_Aero/backup/D012_old100/sedan_D012_100kmh_old_moved.npz
echo "##### 1/2  D012 re-morph at 100 km/h"
time caffeinate -i python3 run_all.py $COMMON --cases cases_morph --npz npz_morph --morph-base cases_prod/D000_100kmh_coarse --clean --designs D012 --speeds 27.78
echo "##### 2/2  D061 re-mesh at 40 km/h"
time caffeinate -i python3 run_all.py $COMMON --cases cases_prod --npz npz --clean --designs D061 --speeds 11.11
echo "Done. Send me a screenshot of this window."
