#!/bin/bash
# Phase A after dataset prep: wait for prep, merge, check, train all three
# models sequentially on GPU 0 only, validate, export. Usage: run_phase_a.sh EPOCHS
set -u
export CUDA_VISIBLE_DEVICES=0
EPOCHS=${1:-36}
cd ~/clarity-v04 && . .venv/bin/activate
while pgrep -f "code/prepare_v04.py" > /dev/null; do sleep 30; done
D=data-v04
python code/merge_manifests.py $D > logs/merge.txt
python code/check_dataset.py $D/manifest.jsonl --deep > logs/dataset_stats.txt 2>&1
mkdir -p weights eval packages
cd code
rm -f ../logs/train_*.jsonl
for k in Packed8 Packed16 Unpacked8; do
  python train.py ../$D/manifest.jsonl $k --epochs $EPOCHS --output ../weights/${k,,}.pt \
    --log ../logs/train_$k.jsonl > ../logs/train_$k.log 2>&1
done
for k in Packed8 Packed16 Unpacked8; do
  w=../weights/${k,,}.pt
  {
    echo "== $k"
    python validate.py probes $w
    python validate.py identity ../$D/manifest.jsonl $w --split val
    python validate.py predict ../$D/manifest.jsonl $w --out ../eval/${k,,} --split val
    for by in "" mode clip rate; do
      echo "-- by=${by:-resolution}"
      python validate.py compare ../$D/manifest.jsonl --baseline identity --new ../eval/${k,,} --split val --by "$by"
    done
    python export.py $w --out ../packages 2>&1 | grep mlpackage
  } > ../logs/val_$k.log 2>&1
done
echo PHASE_A_DONE > ../logs/phase_a.done
