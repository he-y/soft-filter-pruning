# Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks
The PyTorch implementation for [our IJCAI 2018 paper](https://www.ijcai.org/proceedings/2018/0309.pdf).
This implementation is based on [ResNeXt-DenseNet](https://github.com/D-X-Y/ResNeXt-DenseNet).

## Requirements
- Python 3.6
- PyTorch 0.3.1
- TorchVision 0.3.0

## Training ImageNet

### Usage of Pruning Training:
We train each model from scratch by default. If you wish to train the model with pre-trained models, please use the options `--use_pretrain --lr 0.01`.

Run Pruning Training ResNet (depth 152,101,50,34,18) on Imagenet:
(the `layer_begin` and `layer_end` is the index of the first and last conv layer, `layer_inter` choose the conv layer instead of BN layer): 
```bash
python pruning_train.py -a resnet152 --save_dir ./snapshots/resnet152-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 462 --layer_inter 3  /path/to/Imagenet2012

python pruning_train.py -a resnet101 --save_dir ./snapshots/resnet101-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 309 --layer_inter 3  /path/to/Imagenet2012

python pruning_train.py -a resnet50  --save_dir ./snapshots/resnet50-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 156 --layer_inter 3  /path/to/Imagenet2012

python pruning_train.py -a resnet34  --save_dir ./snapshots/resnet34-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 105 --layer_inter 3  /path/to/Imagenet2012

python pruning_train.py -a resnet18  --save_dir ./snapshots/resnet18-rate-0.7 --rate 0.7 --layer_begin 0 --layer_end 57 --layer_inter 3  /path/to/Imagenet2012
```

### Usage of Initial with Pruned Model:
We use unpruned model as initial model by default. If you wish to initial with pruned model, please use the options `--use_sparse --sparse path_to_pruned_model`.

### Usage of Normal Training:
Run resnet(100 epochs): 
```bash
python original_train.py -a resnet50 --save_dir ./snapshots/resnet50-baseline  /path/to/Imagenet2012 --workers 36
```

### Scripts to reproduce the results in our paper
To train the ImageNet model with / without pruning, see the directory `scripts` (we use 8 GPUs for training).

### Inference the pruned model
```bash
sh scripts/inference_resnet.sh
```
The trained models with log files can be found in [Google Drive](https://drive.google.com/drive/folders/1lPhInbd7v3HjK9uOPW_VNjGWWm7kyS8e?usp=sharing)

## Notes

#### Torchvision Version
We use the torchvision of 0.3.0. If the version of your torchvision is 0.2.0, then the `transforms.RandomResizedCrop` should be `transforms.RandomSizedCrop` and the `transforms.Resize` should be `transforms.Scale`.

#### Why use 100 epochs for training
This can improve the accuracy slightly.

## Citation
```
@inproceedings{he2018soft,
  title     = {Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks},
  author    = {He, Yang and Kang, Guoliang and Dong, Xuanyi and Fu, Yanwei and Yang, Yi},
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
  pages     = {2234--2240},
  year      = {2018}
}
```
