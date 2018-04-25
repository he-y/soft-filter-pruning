
python pruning_train.py $DOME_HOME/datasets/ILSVRC2012 -a resnet50 --save_dir ./snapshots/resnet50-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 156 --layer_inter 3 --workers 36
