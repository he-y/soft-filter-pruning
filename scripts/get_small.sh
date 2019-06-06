
python ./utils/get_small_model.py /data/yahe/imagenet/ImageNet2012 -a resnet50  --workers 12 \
--resume  /data/yahe/imagenet/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar \
--save_dir ./small/resnet50/ --batch-size 64 \
--get_small
