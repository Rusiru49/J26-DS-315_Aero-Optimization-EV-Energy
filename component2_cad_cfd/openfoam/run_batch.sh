#!/bin/bash
# usage:  bash run_batch.sh <batch number 1-10>     (run inside the openfoam2606 terminal, in ~/Documents/EV_Aero/openfoam)
# Runs 6-7 designs by FFD morph, ~27 min each.  Safe to stop (Ctrl+C) and start again: finished designs are skipped.
case "$1" in
 1) D=D019,D038,D002,D037,D049,D012,D003;;
 2) D=D005,D015,D039,D036,D004,D059,D008;;
 3) D=D061,D047,D043,D033,D068,D026,D031;;
 4) D=D042,D021,D045,D064,D034,D065,D058;;
 5) D=D023,D027,D054,D016,D028,D055,D030;;
 6) D=D040,D032,D044,D067,D041,D035,D007;;
 7) D=D053,D010,D060,D020,D056,D001,D066;;
 8) D=D050,D062,D022,D069,D051,D025,D013;;
 9) D=D006,D014,D011,D009,D024,D057,D046;;
 10) D=D029,D048,D017,D018,D052,D063;;
 *) echo "usage: bash run_batch.sh <1-10>"; exit 1;;
esac
echo "Batch $1 : $D"
RUN="python3 run_all.py"
COMMON="--table design_table_sedan_v5.csv --stl-dir /Users/chanuka/Documents/EV_Aero_sedan_cad --speeds 27.78 --mesh coarse --np 4 --iters 1500 --full --cases cases_morph --npz npz_morph --morph-base cases_prod/D000_100kmh_coarse"
if [ "$1" = "1" ]; then
  # D019 and D012 are kept WITHOUT --clean (their flow fields are needed for the ParaView pictures); the rest is cleaned to save disk
  time caffeinate -i $RUN $COMMON --designs D019,D012
  D=D038,D002,D037,D049,D003
fi
time caffeinate -i python3 run_all.py --table design_table_sedan_v5.csv --stl-dir /Users/chanuka/Documents/EV_Aero_sedan_cad \
  --designs $D --speeds 27.78 --mesh coarse --np 4 --iters 1500 --full --cases cases_morph --npz npz_morph \
  --morph-base cases_prod/D000_100kmh_coarse --clean
echo; echo "Batch $1 finished.  Now run:  python3 qc_report.py"
