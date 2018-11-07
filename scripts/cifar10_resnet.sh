#!/bin/bash


change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90
resnet20 54
}

pruning(){
CUDA_VISIBLE_DEVICES=$1 python pruning_cifar10_resnet.py  ./data/cifar.python --dataset cifar10 --arch resnet110 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--rate 0.7 \
--layer_begin 0  --layer_end 324 --layer_inter 3 --epoch_prune 1
}


pruning 0 ./logs/cifar10_resnet110_norm2_0_324_3_rate0.7


