#!/bin/bash
DEV=1
output_path="output"

# CELEBA
declare -A checkpoint_paths
checkpoint_paths["dcgan"]="trained_models/dcgan/celeba/nz=100_niter=50_model=1/checkpoints/netG_epoch_49.pth"
checkpoint_paths["wgangp"]="trained_models/wgangp/celeba/nz=100_niter=200_model=1/checkpoints/netG_epoch_199.pth"
checkpoint_paths["lsgan"]="trained_models/lsgan/celeba/nz=100_niter=100_model=1/checkpoints/netG_epoch_99.pth"
checkpoint_paths["ebgan"]="trained_models/ebgan/celeba/nz=100_niter=100_model=1/checkpoints/netG_epoch_99.pth"
n_epochs=50
for dataset in "celeba_0" "celeba_1" "celeba_2" "celeba_3" "celeba_4"
do
    for generator in "dcgan" "wgangp" "lsgan" "ebgan"
    do
        train_dir="${generator}_1"
        checkpoint_path=${checkpoint_paths[$generator]}
        # max downsamping FLIPAD
        CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --test-models same --avoid-caching --train-dir $train_dir --real-dir real --dataset $dataset --model $generator --checkpoint-path $checkpoint_path --rec-lr 0.025 --rec-max-iter 10000 --rec-alpha 0.0005 --num-train 10000 --lr 0.0005 --n-epochs $n_epochs --output-path $output_path --downsampling max --net-name lenet_64channels --num_select_channels 64
        # bicubic downsampling RAWPAD
        CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --test-models same --avoid-caching --train-dir $train_dir --real-dir real --dataset $dataset --model $generator --checkpoint-path $checkpoint_path --num-train 10000 --lr 0.0005 --feat raw --net-name cifar10_LeNet --n-epochs $n_epochs --output-path $output_path
        # center crop DCTPAD
        CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --test-models same --avoid-caching --train-dir $train_dir --real-dir real --dataset $dataset --model $generator --checkpoint-path $checkpoint_path --num-train 10000 --lr 0.0005 --feat dct --net-name cifar10_LeNet --n-epochs $n_epochs --output-path $output_path --downsampling center_crop
    done
done

# LSUN
declare -A checkpoint_paths
checkpoint_paths["dcgan"]="trained_models/dcgan/lsun/nz=100_niter=10_model=1/checkpoints/netG_epoch_9.pth"
checkpoint_paths["wgangp"]="trained_models/wgangp/lsun/nz=100_niter=10_model=1/checkpoints/netG_epoch_9.pth"
checkpoint_paths["lsgan"]="trained_models/lsgan/lsun/nz=100_niter=10_model=1/checkpoints/netG_epoch_9.pth"
checkpoint_paths["ebgan"]="trained_models/ebgan/lsun/nz=100_niter=5_model=1/checkpoints/netG_epoch_4.pth"

for dataset in "lsun_0" "lsun_1" "lsun_2" "lsun_3" "lsun_4"
do
    for generator in "dcgan" "wgangp" "lsgan" "ebgan"
    do
        train_dir="${generator}_1"
        checkpoint_path=${checkpoint_paths[$generator]}
        # FLIPAD Max downsmpling
        CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --test-models same --avoid-caching --train-dir $train_dir --real-dir real --dataset $dataset --model $generator --checkpoint-path $checkpoint_path --rec-lr 0.00075 --rec-max-iter 10000 --rec-alpha 0.0005 --num-train 10000 --lr 0.0005 --n-epochs $n_epochs --output-path $output_path --downsampling max --net-name lenet_64channels --num_select_channels 64
        # RAWPAD bicubic downsampling:
        CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --test-models same --avoid-caching --train-dir $train_dir --real-dir real --dataset $dataset --model $generator --checkpoint-path $checkpoint_path --num-train 10000 --lr 0.0005 --feat raw --net-name cifar10_LeNet --n-epochs $n_epochs --output-path $output_path
        # DCTPAD center crop:
        CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --test-models same --avoid-caching --train-dir $train_dir --real-dir real --dataset $dataset --model $generator --checkpoint-path $checkpoint_path --num-train 10000 --lr 0.0005 --feat dct --net-name cifar10_LeNet --n-epochs $n_epochs --output-path $output_path --downsampling center_crop
    done
done