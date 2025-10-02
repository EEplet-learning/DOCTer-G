#!/bin/bash
epoch=100
seed=1234
trainseed=99

python ./run_DOC/run_DOC.py --seed 2025 --trainseed 56  --exptype "DG" --model "MulTmp" --cuda 0 --param 0  --batch_size 512 --epochs 200 --p1 0.06 --p3 0.8 --th 4 --clip_value 1  --lr 0.0005 &
wait
echo "DONE"