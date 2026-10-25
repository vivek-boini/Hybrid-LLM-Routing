#!/usr/bin/env bash
# Evaluate everything on the TEST split. Run from the repo root after train_all.sh.
#   DATA=data/mix.npz PROFILE=profiles/7b.json LAM=5 bash scripts/eval_all.sh
set -e
DATA=${DATA:-data/mix.npz}; LAM=${LAM:-5}; PROF=${PROFILE:+--profile $PROFILE}
RL='--rl Baseline-RL=runs/baseline/best.pt:expert_only:flat --rl +State=runs/state/best.pt:full:flat --rl QoS-RL=runs/qos/best.pt:full:flat'
BASE="--data $DATA --n-req 500 --seeds 10 --policies rr,sqf,br,greedy $PROF"
mkdir -p results
python -m qos.cli eval $BASE --lam $LAM $RL --out results/main.csv
python -m qos.cli eval $BASE --lam $LAM $RL --sweep lam=2,3,5,7,10 --out results/lam.csv
python -m qos.cli eval $BASE --lam $LAM $RL --sweep L_ms=20,25,30,35,40 --out results/L.csv
python -m qos.cli eval $BASE --lam $LAM $RL --workload bursty --out results/bursty.csv
python -m qos.cli eval --data $DATA --lam $LAM --n-req 500 --seeds 10 --policies br,greedy $PROF --pred-s none --pred-d none --out results/abl_nopred.csv
python -m qos.cli eval --data $DATA --lam $LAM --n-req 500 --seeds 10 --policies br,greedy $PROF --pred-s oracle --pred-d oracle --out results/abl_oracle.csv
for f in main lam L bursty; do python -m qos.plot_results results/$f.csv; done
