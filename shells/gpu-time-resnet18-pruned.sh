
python gpu_time.py $DOME_HOME/datasets/ILSVRC2012  -a resnet18  --workers 40  \
	--resume  ./snapshots/resnet18-rate-0.7/checkpoint.resnet18.2018-01-08-6905.pth.tar \
	--save_dir ./snapshots/resnet18-0.7-time/ --batch-size 64
