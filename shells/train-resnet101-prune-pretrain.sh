
python pruning_train.py $DOME_HOME/datasets/ILSVRC2012 -a resnet101 --save_dir ./snapshots/Pretrain-resnet101-rate-0.7 \
	--rate 0.7 --layer_begin 0 --layer_end 309 --layer_inter 3 --workers 36 \
	--use_pretrain --lr 0.01
