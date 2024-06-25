#!/bin/bash
# Set CUDA DEVICE:
DEV=0

# Set Data Root:
DATA_ROOT=/USERSPACE/DATASETS/

# CELEBA
for iter in {1..5}
do
    CUDA_VISIBLE_DEVICE=$DEV python3 scripts/training/train_dcgan.py $DATA_ROOT --cuda --dataset celeba --niter 10 #50 
    CUDA_VISIBLE_DEVICE=$DEV python3 scripts/training/train_wgangp.py $DATA_ROOT --cuda --dataset celeba --niter 200 --b1 0.0 --b2 0.9 --nz 100 
    CUDA_VISIBLE_DEVICE=$DEV python3 scripts/training/train_lsgan.py $DATA_ROOT --cuda --dataset celeba --niter 100 --num_layers 4 
    CUDA_VISIBLE_DEVICE=$DEV python3 scripts/training/train_ebgan.py $DATA_ROOT --cuda --dataset celeba --niter 100 
done

# LSUN
for iter in {1..5}
do
    CUDA_VISIBLE_DEVICE=$DEV python3 scripts/training/train_dcgan.py $DATA_ROOT --cuda --dataset lsun --niter 2 #10 
    CUDA_VISIBLE_DEVICE=$DEV python3 scripts/training/train_wgangp.py $DATA_ROOT --cuda --dataset lsun --niter 10 --b1 0.0 --b2 0.9 --nz 100 
    CUDA_VISIBLE_DEVICE=$DEV python3 scripts/training/train_lsgan.py $DATA_ROOT --cuda --dataset lsun --niter 10 --num_layers 4  
    CUDA_VISIBLE_DEVICE=$DEV python3 scripts/training/train_ebgan.py $DATA_ROOT --cuda --dataset lsun --niter 5 
done
