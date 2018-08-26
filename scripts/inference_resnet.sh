infer_resnet18(){
python pruning_train.py /path/to/ImageNet2012 -a resnet18 --resume /path/to/checkpoint.resnet18.pth.tar -e --save_dir ./infer_log --workers 24 
}

infer_resnet34(){
python pruning_train.py /path/to/ImageNet2012 -a resnet34 --resume /path/to/checkpoint.resnet34.pth.tar -e --save_dir ./infer_log --workers 24 
}

infer_resnet50(){
python pruning_train.py /path/to/ImageNet2012 -a resnet50 --resume /path/to/checkpoint.resnet50.pth.tar -e --save_dir ./infer_log --workers 24 
}

infer_resnet101(){
python pruning_train.py /path/to/ImageNet2012 -a resnet101 --resume /path/to/checkpoint.resnet101.pth.tar -e --save_dir ./infer_log --workers 24 
}



infer_resnet50


