
python gpu_time.py $DOME_HOME/datasets/ILSVRC2012  -a resnet101  --workers 40  \
	--resume  ./snapshots/resnet101-rate-0.7/checkpoint.resnet101.2018-01-07-4965.pth.tar  \
	--save_dir ./snapshots/resnet101-0.7-time/ --batch-size 64 
