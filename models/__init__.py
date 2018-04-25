"""The models subpackage contains definitions for the following model
architectures:
-  `ResNeXt` for CIFAR10 CIFAR100
You can construct a model with random weights by calling its constructor:
.. code:: python
    import models
    resnet20 = models.ResNet20(num_classes)
    resnet32 = models.ResNet32(num_classes)


.. ResNext: https://arxiv.org/abs/1611.05431
"""

from .resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from .preresnet import preresnet20, preresnet32, preresnet44, preresnet56, preresnet110
from .caffe_cifar import caffe_cifar

from .imagenet_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .alexnet import alexnet
from .vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn 

from .imagenet_resnet_small import resnet18_small, resnet34_small, resnet50_small, resnet101_small, resnet152_small
