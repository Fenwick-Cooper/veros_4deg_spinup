#!/bin/zsh
# Every run behind Description/method_comparison.tex, one after another (never in parallel,
# so that wall-clock times are comparable). Run from the repository root:
#
#     code/newton_krylov_anderson/run_all.sh
#
# Takes about 5 hours. Needs the 2000-year reference run (code/run_2000yr_reference.py and
# code/analyse_2000yr_reference.py) to have been made first. Set VEROS_PYTHON to the Python
# of the environment in which Veros is installed if it is not the `python` on the path.
# Output: data/intermediate/newton_krylov_anderson/ (<tag>.log, <tag>.csv, summary.jsonl,
# run_all.log, the timing files, conventional_curve.npz, comparison.md).
# The figure needs matplotlib: see README.md.
P=${VEROS_PYTHON:-python}
D=code/newton_krylov_anderson
OUT=data/intermediate/newton_krylov_anderson
mkdir -p $OUT
: > $OUT/run_all.log
rm -f $OUT/summary.jsonl
stamp() { echo "=== $1 $(date +%Y-%m-%dT%H:%M:%S)" >> $OUT/run_all.log; }
run() { local tag=$1; shift; stamp $tag; echo "$tag"; $P "$@" --tag $tag > $OUT/$tag.log 2>&1; }
# the methods (the start-up of each is the gap between these time stamps minus its own clock)
run L3      $D/anderson_outer.py
run L2      $D/anderson_outer.py --cycles 20:30y:0.125,20:60y:2,20:60y:4,20:2000y:32
run C_k0    $D/nk_emulator.py --k 0
run C_k5f   $D/nk_emulator.py --k 5 --freeze-closures
run D_plain $D/anderson_outer.py --extra 3
run D_aa3   $D/anderson_outer.py --extra 3 --aa-m 3 --aa-start 4
run A_m10   $D/anderson_veros.py --tau 360 --m 10 --max-wall 3600
run B_k20   $D/nk_veros.py --tau 360 --k 20 --max-wall 3600
stamp done
# conventional Veros: speed, full run time, and the curve
echo "conventional Veros"
$P $D/measure_speed.py > $OUT/measure_speed.log 2>&1
rm -f $OUT/conventional_timing.jsonl
$P $D/time_conventional.py --days 3714 --no-output > $OUT/time_conventional.log 2>&1
$P $D/time_conventional.py --days 3714 >> $OUT/time_conventional.log 2>&1
$P $D/conventional_curve.py > $OUT/conventional_curve.log 2>&1
# the table
$P $D/startup_times.py
$P $D/compare.py
