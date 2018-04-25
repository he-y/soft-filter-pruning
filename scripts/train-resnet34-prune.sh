
python pruning_train.py $DOME_HOME/datasets/ILSVRC2012 -a resnet34 --save_dir ./snapshots/resnet34-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 105 --layer_inter 3 --workers 36
