
python gpu_time.py $DOME_HOME/datasets/ILSVRC2012  -a resnet50  --workers 40  --resume  ./snapshots/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar  \
	--save_dir ./snapshots/resnet50-baseline/ --batch-size 64 --big_small
