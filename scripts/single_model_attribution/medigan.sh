#!/bin/bash
DEV=3
output_path="output"

n_epochs=50
num_train=10000
rec_alpha=0.001
for dataset in "breastmass_0" "breastmass_1" "breastmass_2" "breastmass_3" "breastmass_4"
do
    checkpoint_path="trained_models/dcgan_bcdr/model_state_dict.pt"
    # max downsamping FLIPAD
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --train-dir dcgan_bcdr --real-dir wgangp_bcdr --dataset $dataset --model medigan_dcgan --checkpoint-path $checkpoint_path --rec-lr 0.05 --rec-max-iter 10000 --rec-alpha $rec_alpha --num-train $num_train --lr 0.0005 --n-epochs $n_epochs --output-path $output_path --downsampling max --net-name "lenet_64channels_medigan" --num_select_channels 64 --feat act
    # bicubic downsampling RAWPAD
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --train-dir dcgan_bcdr --real-dir wgangp_bcdr --dataset $dataset --model medigan_dcgan --checkpoint-path $checkpoint_path --num-train $num_train --lr 0.0005 --feat raw --net-name lenet_128x128_medigan --n-epochs $n_epochs --output-path $output_path
    # center crop DCTPAD
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad.py --train-dir dcgan_bcdr --real-dir wgangp_bcdr --dataset $dataset --model medigan_dcgan --checkpoint-path $checkpoint_path --num-train $num_train --lr 0.0005 --feat dct --net-name lenet_128x128_medigan --n-epochs $n_epochs --output-path $output_path --downsampling center_crop

    # Fingerprint
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_baselines.py --attr fingerprint --train-dir dcgan_bcdr --real-dir wgangp_bcdr --dataset $dataset --model medigan_dcgan --checkpoint-path $checkpoint_path --num-train $num_train --output-path $output_path
done
