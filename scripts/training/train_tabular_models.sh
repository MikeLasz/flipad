#!/bin/bash
DEV=0

for data in "whitewine" "redwine"
do
    CUDA_VISIBLE_DEVICES=$DEV python3 scripts/training/train_klwgan.py --data $data --load_model
    CUDA_VISIBLE_DEVICES=$DEV python3 scripts/training/train_sdv_models.py --data $data --model copulagan --load_model
    CUDA_VISIBLE_DEVICES=$DEV python3 scripts/training/train_sdv_models.py --data $data --model tvae --load_model
    CUDA_VISIBLE_DEVICES=$DEV python3 scripts/training/train_sdv_models.py --data $data --model ctgan --load_model
done