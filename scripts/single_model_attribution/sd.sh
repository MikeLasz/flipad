#!/bin/bash
DEV=0
nepochs=100
output_path="output"
num_train=2000
num_val=100
num_test=200
batch_size=32

for _ in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --avoid-caching --train-dir stable-diffusion-2-1-base --real-dir stable-diffusion-v1-4 --model stablediffusion --checkpoint stabilityai/stable-diffusion-2-1-base --num-train $num_train --num-val $num_val --num-test $num_test --lr 0.0005 --n-epochs $nepochs --rec-alpha 0.00001 --rec-max-iter 1000 --net-name cifar10_biglenet --batch-size $batch_size --feat act --num_select_channels 3 --output-path $output_path --downsampling center_crop --dataset coco2014train --rec-lr 0.01
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --avoid-caching --train-dir stable-diffusion-2-1-base --real-dir stable-diffusion-v1-4 --model stablediffusion --checkpoint stabilityai/stable-diffusion-2-1-base --num-train $num_train --num-val $num_val --num-test $num_test --lr 0.0005 --n-epochs $nepochs --rec-alpha 0.00001 --rec-max-iter 1000 --net-name cifar10_biglenet --batch-size $batch_size --feat raw --num_select_channels 3 --output-path $output_path --downsampling center_crop --dataset coco2014train
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --avoid-caching --train-dir stable-diffusion-2-1-base --real-dir stable-diffusion-v1-4 --model stablediffusion --checkpoint stabilityai/stable-diffusion-2-1-base --num-train $num_train --num-val $num_val --num-test $num_test --lr 0.0005 --n-epochs $nepochs --rec-alpha 0.00001 --rec-max-iter 1000 --net-name cifar10_biglenet --batch-size $batch_size --feat dct --num_select_channels 3 --output-path $output_path --downsampling center_crop --dataset coco2014train

    # baselines
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_baselines.py  --attr fingerprint --train-dir stable-diffusion-2-1-base --real-dir stable-diffusion-v1-4 --model stablediffusion --checkpoint-path stabilityai/stable-diffusion-2-1-base --dataset coco2014train --num-train $num_train --num-val $num_val --num-test $num_test --output-path $output_path --batch-size 128
done