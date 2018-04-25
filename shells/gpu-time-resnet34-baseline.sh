
python gpu_time.py $DOME_HOME/datasets/ILSVRC2012  -a resnet34  --workers 40  \
	--resume  ./snapshots/resnet34-rate-0.7/checkpoint.resnet34.2018-01-05-5277.pth.tar  \
	--save_dir ./snapshots/resnet34-baseline-time/ --batch-size 64  --big_small
