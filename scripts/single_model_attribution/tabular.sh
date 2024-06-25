#!/bin/bash
DEV=0

output_path="output"
num_train=100000
num_test=10000
num_val=10000
rec_lr=0.001
n_epochs=100
num_dimensions=100
rep_dim=32
real_dir="tvae"
# Redwine:
rec_alpha=0.0005
for j in "0" "1" "2" "3" "4"
do
    checkpoint_path="trained_models/klwgan-hinge/redwine_${j}.pkl"
    dataset="redwine_${j}"
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad_tabular.py --train-dir klwgan-hinge --real-dir $real_dir --dataset $dataset --checkpoint-path $checkpoint_path --feat act --num-train $num_train --num-test $num_test --num-val $num_val --model klwgan --net-name mlp --rec-alpha $rec_alpha --output-path $output_path --n-epochs $n_epochs --rep-dim $rep_dim --mlp-hdims 512 1024 512 256 128 --num-select-dimensions $num_dimensions --rec-lr $rec_lr
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad_tabular.py --train-dir klwgan-hinge --real-dir $real_dir --dataset $dataset --checkpoint-path $checkpoint_path --feat raw --num-train $num_train --num-test $num_test --num-val $num_val --model klwgan --net-name mlp --rec-alpha $rec_alpha --output-path $output_path --n-epochs $n_epochs --rep-dim $rep_dim --mlp-hdims 512 1024 512 256 128 --num-select-dimensions $num_dimensions --rec-lr $rec_lr
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_baselines.py --attr l2_inversion --train-dir klwgan-hinge --real-dir $real_dir --dataset $dataset --model klwgan --checkpoint-path $checkpoint_path --output-path $output_path --batch-size 256
done


# Whitewine:
rec_alpha=0.0001
for j in "0" "1" "2" "3" "4" 
do
    checkpoint_path="trained_models/klwgan-hinge/whitewine_${j}.pkl"
    dataset="whitewine_${j}"
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad_tabular.py --train-dir klwgan-hinge --real-dir $real_dir --dataset $dataset --checkpoint-path $checkpoint_path --feat act --num-train $num_train --num-test $num_test --num-val $num_val --model klwgan --net-name mlp --rec-alpha $rec_alpha --output-path $output_path --n-epochs $n_epochs --rep-dim $rep_dim --mlp-hdims 512 1024 512 256 128 --num-select-dimensions $num_dimensions --rec-lr $rec_lr
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_deepsad_tabular.py --train-dir klwgan-hinge --real-dir $real_dir --dataset $dataset --checkpoint-path $checkpoint_path --feat raw --num-train $num_train --num-test $num_test --num-val $num_val --model klwgan --net-name mlp --rec-alpha $rec_alpha --output-path $output_path --n-epochs $n_epochs --rep-dim $rep_dim --mlp-hdims 512 1024 512 256 128 --num-select-dimensions $num_dimensions --rec-lr $rec_lr
    CUDA_VISIBLE_DEVICES=$DEV python3 sma_baselines.py --attr l2_inversion --train-dir klwgan-hinge --real-dir $real_dir --dataset $dataset --model klwgan --checkpoint-path $checkpoint_path --output-path $output_path --batch-size 256
done

