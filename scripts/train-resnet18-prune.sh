
python pruning_train.py $DOME_HOME/datasets/ILSVRC2012 -a resnet18 --save_dir ./snapshots/resnet18-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 57 --layer_inter 3 --workers 36
