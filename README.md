# Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks
The PyTorch implementation for [this paper]()

## Requirements
- PyThon 3.6
- PyTorch >= 0.1.0
- TorchVision = 0.3.0

## Training ImageNet



### Usage of Pruning Training:
We train each model from scratch by default. If you wish to train the model with pre-trained models, please use the options `--use_pretrain --lr 0.01`.

### Usage of Pruning Rate Decay:
We train each model for stable pruning rate by default. If you wish to train the model with exponential pruning rate, please use the options `--use_rate_decay`.

### Usage of Initial with Pruned Model:
We use unpruned model as initial model by default. If you wish to initial with pruned model, please use the options `--use_sparse --sparse path_to_pruned_model`.

Run resnet152: 
```bash
python pruning_train.py -a resnet152 --save_dir ./snapshots/resnet152-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 462 --layer_inter 3  /path/to/Imagenet2012
```

Run resnet101: 
```bash
python pruning_train.py -a resnet101 --save_dir ./snapshots/resnet101-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 309 --layer_inter 3  /path/to/Imagenet2012
```

Run resnet50: 
```bash
python pruning_train.py -a resnet50  --save_dir ./snapshots/resnet50-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 156 --layer_inter 3  /path/to/Imagenet2012
```
Run resnet34: 
```bash
python pruning_train.py -a resnet34  --save_dir ./snapshots/resnet34-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 105 --layer_inter 3  /path/to/Imagenet2012
```
Run resnet18: 
```bash
python pruning_train.py -a resnet18  --save_dir ./snapshots/resnet18-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 57 --layer_inter 3  /path/to/Imagenet2012
```

### Usage of Normal Training:
Run resnet(100 epochs): 
```bash
python original_train.py -a resnet50 --save_dir ./snapshots/resnet50-baseline  /path/to/Imagenet2012 --workers 36
```

### Usage of GPU Time:
See the directory `shells`. Enable `big_small` to test baseline model, unable `big_small` to test pruned model.

### Scripts to reproduce the results in our paper
To train the ImageNet model with / without pruning, see the directory `shells`

The trained models with log files can be found in [Google Drive](https://drive.google.com/drive/folders/1lPhInbd7v3HjK9uOPW_VNjGWWm7kyS8e?usp=sharing)


## Notes

#### torchvision version
We use the torchvision of 0.3.0. If the version of your torchvision is 0.2.0, then the `transforms.RandomResizedCrop` should be `transforms.RandomSizedCrop` and the `transforms.Resize` should be `transforms.Scale`.

#### why use 100 epochs for training
This can obtain a sight accuracy improvement.

#### This implementation refers to [ResNeXt-DenseNet](https://github.com/D-X-Y/ResNeXt-DenseNet)

## Citation
