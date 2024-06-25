#!/bin/bash
DEV=3
nepochs=100
train_dir="stylegan2"
real_dir="real"
checkpoint="trained_models/stylegan2/stylegan2-ffhq-256x256.pkl"
output_dir="output"

for _ in 0 1 2 3 4
do
    dataset="ffhq_${_}"
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --data-path ../deep_algebraic_attribution/data --train-dir $train_dir  --real-dir $real_dir --model stylegan2 --checkpoint $checkpoint --num-train 10000 --num-val 1000 --num-test 1000 --lr 0.0005 --n-epochs $nepochs --net-name cifar10_biglenet --batch-size 32 --feat dct --num_select_channels 3 --output-path $output_dir --downsampling center_crop --dataset $dataset
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --data-path ../deep_algebraic_attribution/data --train-dir $train_dir  --real-dir $real_dir --model stylegan2 --checkpoint $checkpoint --num-train 10000 --num-val 1000 --num-test 1000 --lr 0.0005 --n-epochs $nepochs --net-name cifar10_biglenet --batch-size 32 --feat raw --num_select_channels 3 --output-path $output_dir --dataset $dataset
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_baselines.py --data-path ../deep_algebraic_attribution/data --train-dir $train_dir --real-dir $real_dir --model stylegan2 --checkpoint $checkpoint --num-train 10000 --num-val 1000 --num-test 1000 --output-path $output_dir --dataset $dataset --attr fingerprint
done
