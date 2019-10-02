# Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks
The PyTorch implementation for [our IJCAI 2018 paper](https://www.ijcai.org/proceedings/2018/0309.pdf).
This implementation is based on [ResNeXt-DenseNet](https://github.com/D-X-Y/ResNeXt-DenseNet).

## Updates:
The journal version of this work [Asymptotic Soft Filter Pruning for Deep Convolutional Neural Networks](https://ieeexplore.ieee.org/document/8816678) is available now, code coming soon.


## Table of Contents

- [Requirements](#requirements)
- [Models and log files](#models-and-log-files)
- [Training ImageNet](#training-imagenet)
  - [Usage of Pruning Training](#usage-of-pruning-training)
  - [Usage of Initial with Pruned Model](#usage-of-initial-with-pruned-model)
  - [Usage of Normal Training](#usage-of-normal-training)
  - [Inference the pruned model with zeros](#inference-the-pruned-model-with-zeros)
  - [Inference the pruned model without zeros](#inference-the-pruned-model-without-zeros)
  - [Get small model](#get-small-model)
  - [Scripts to reproduce the results in our paper](#scripts-to-reproduce-the-results-in-our-paper)
- [Training Cifar-10](#training-cifar-10)
- [Notes](#notes)
  - [Torchvision Version](#torchvision-version)
  - [Why use 100 epochs for training](#why-use-100-epochs-for-training)
  - [Process of ImageNet dataset](#process-of-imagenet-dataset)
  - [FLOPs Calculation](#flops-calculation)
- [Citation](#citation)


## Requirements
- Python 3.6
- PyTorch 0.3.1
- TorchVision 0.2.0

## Models and log files
The trained models with log files can be found in [Google Drive](https://drive.google.com/drive/folders/1lPhInbd7v3HjK9uOPW_VNjGWWm7kyS8e?usp=sharing).

The pruned model without zeros: [Release page](https://github.com/he-y/soft-filter-pruning/releases/tag/ResNet50_pruned).

## Training ImageNet

#### Usage of Pruning Training
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

#### Usage of Initial with Pruned Model
We use unpruned model as initial model by default. If you wish to initial with pruned model, please use the options `--use_sparse --sparse path_to_pruned_model`.

#### Usage of Normal Training
Run resnet(100 epochs): 
```bash
python original_train.py -a resnet50 --save_dir ./snapshots/resnet50-baseline  /path/to/Imagenet2012 --workers 36
```

#### Inference the pruned model with zeros
```bash
sh scripts/inference_resnet.sh
```

#### Inference the pruned model without zeros
```bash
sh scripts/infer_pruned.sh
```
The pruned model without zeros could be downloaded at the [Release page](https://github.com/he-y/soft-filter-pruning/releases/tag/ResNet50_pruned).

#### Get small model
Get the model without zeros.
In the below script, change the path of the resume model to the pruned-model with zeros, then both the big model (with zero) and small model (without zero) will be saved. This script support ResNet of depth 18, 34, 50, 101.
```bash
sh scripts/get_small.sh
```


#### Scripts to reproduce the results in our paper
To train the ImageNet model with / without pruning, see the directory `scripts` (we use 8 GPUs for training).

## Training Cifar-10
```bash
sh scripts/cifar10_resnet.sh
```
Please be care of the hyper-parameter [`layer_end`](https://github.com/he-y/soft-filter-pruning/blob/master/scripts/cifar10_resnet.sh#L4-L9) for different layer of ResNet.

## Notes

#### Torch Version
We use the torch of 0.3.1. If the version of your torch is 0.2.0, then the `transforms.RandomResizedCrop` should be `transforms.RandomSizedCrop` and the `transforms.Resize` should be `transforms.Scale`.

#### Why use 100 epochs for training
This can improve the accuracy slightly.

#### Process of ImageNet dataset
We follow the [Facebook process of ImageNet](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).
Two subfolders ("train" and "val") are included in the "/path/to/ImageNet2012".
The correspding code is [here](https://github.com/he-y/soft-filter-pruning/blob/master/pruning_train.py#L129-L130).

#### FLOPs Calculation
Refer to the [file](https://github.com/he-y/soft-filter-pruning/blob/master/utils/cifar_resnet_flop.py).

## Citation
```
@inproceedings{he2018soft,
  title     = {Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks},
  author    = {He, Yang and Kang, Guoliang and Dong, Xuanyi and Fu, Yanwei and Yang, Yi},
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
  pages     = {2234--2240},
  year      = {2018}
}

@article{he2019asymptotic,
  title={Asymptotic Soft Filter Pruning for Deep Convolutional Neural Networks}, 
  author={He, Yang and Dong, Xuanyi and Kang, Guoliang and Fu, Yanwei and Yan, Chenggang and Yang, Yi}, 
  journal={IEEE Transactions on Cybernetics}, 
  year={2019}, 
  volume={}, 
  number={}, 
  pages={1-11}, 
  doi={10.1109/TCYB.2019.2933477}, 
  ISSN={2168-2267}, 
}
```

