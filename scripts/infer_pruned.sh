#!/usr/bin/env bash


python infer_pruned.py /data/yahe/imagenet/ImageNet2012  -a resnet50  --workers 4  --batch-size 64 \
--resume  /data/yahe/imagenet/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar \
--small_model ./logs/ResNet50_pruned_rate0.3.pt \
-e --eval_small --save_dir ./logs/infer_small_model/

python infer_pruned.py /data/yahe/imagenet/ImageNet2012  -a resnet50  --workers 4  --batch-size 64 \
--resume  /data/yahe/imagenet/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar \
--small_model  ./logs/ResNet50_pruned_rate0.3.pt \
-e  --save_dir ./logs/infer_model_withzero/


