#!/usr/bin/env bash
# Train the three RL configs that mirror the paper's ablation. Run from the repo root.
#   DATA=data/mix.npz PROFILE=profiles/7b.json LAM=5 STEPS=300000 bash scripts/train_all.sh
set -e
DATA=${DATA:-data/mix.npz}; LAM=${LAM:-5}; STEPS=${STEPS:-300000}; NREQ=${NREQ:-300}
PROF=${PROFILE:+--profile $PROFILE}
COMMON="--data $DATA --lam $LAM --n-req $NREQ --steps $STEPS --update-every 1 --eval-every 10000 $PROF"
mkdir -p runs
python -m qos.cli train $COMMON --state expert_only --reward baseline --out runs/baseline > runs/baseline.log 2>&1 &
python -m qos.cli train $COMMON --state full        --reward baseline --out runs/state    > runs/state.log    2>&1 &
python -m qos.cli train $COMMON --state full        --reward qos      --out runs/qos      > runs/qos.log      2>&1 &
# optional DeepSets variant (slower):
# python -m qos.cli train $COMMON --state full --reward qos --encoder deepsets --out runs/qos_ds > runs/qos_ds.log 2>&1 &
wait
echo "done. best checkpoints: runs/*/best.pt"
