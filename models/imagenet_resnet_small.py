import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch

__all__ = ['ResNet', 'resnet18_small', 'resnet34_small', 'resnet50_small', 'resnet101_small', 'resnet152_small']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, plane_expand, full_demension, index, batch, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.full_demension = full_demension
        self.index = Variable(index).cuda()
        # self.out = torch.autograd.Variable(
        #     torch.rand(batch, self.full_demension, 64 * 56 // self.full_demension, 64 * 56 // self.full_demension),
        #     volatile=True).cuda()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # setting: for real input without index match
        # out += residual
        # out = self.relu(out)

        # setting: for real input with index match
        residual.index_add_(0, self.index, out)
        out = self.relu(residual)

        # setting: for fake input
        # out = self.relu(self.out)

        return out


class Bottleneck(nn.Module):
    # expansion is not accurately equals to 4
    expansion = 4

    def __init__(self, inplanes, planes, plane_expand, full_demension, index, batch, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # setting: for accuracy expansion
        self.conv3 = nn.Conv2d(planes, plane_expand, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(plane_expand)

        # setting: expansion = 4
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)

        # setting: original resnet
        # self.conv3 = nn.Conv2d(planes, full_demension * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(full_demension * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.full_demension = full_demension
        self.index = Variable(index).cuda()
        # self.out = torch.autograd.Variable(
        #     torch.rand(batch, self.full_demension * 4, 64 * 56 // self.full_demension, 64 * 56 // self.full_demension),
        #     volatile=True).cuda()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # print("conv1 size", out.size())

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print("conv2 size", out.size())
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # print("conv3 size", out.size())
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # setting: for real input without index match
        # out += residual
        # out = self.relu(out)

        # setting: for real input with index match
        # print('index', self.index.size())
        residual.index_add_(1, self.index, out)
        out = self.relu(residual)
        # print("out size", out.size())

        # setting: for fake input
        # out = self.relu(self.out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, batch, index, rate=[64, 64, 64 * 4, 128, 128 * 4, 256, 256 * 4, 512, 512 * 4],
                 num_classes=1000):
        self.inplanes = rate[0]
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, rate[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(rate[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # setting: expansion = 4
        # self.layer1 = self._make_layer(block, rate[1], rate[1] * 4, 64, index, batch, layers[0])
        # self.layer2 = self._make_layer(block, rate[2], rate[2] * 4, 128, index, batch, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, rate[3], rate[3] * 4, 256, index, batch, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, rate[4], rate[4] * 4, 512, index, batch, layers[3], stride=2)

        # setting: accurate expansion
        index_layer1 = {key: index[key] for key in index.keys() if 'layer1' in key}
        index_layer2 = {key: index[key] for key in index.keys() if 'layer2' in key}
        index_layer3 = {key: index[key] for key in index.keys() if 'layer3' in key}
        index_layer4 = {key: index[key] for key in index.keys() if 'layer4' in key}

        self.layer1 = self._make_layer(block, rate[1], rate[2], 64, index_layer1, batch, layers[0])
        self.layer2 = self._make_layer(block, rate[3], rate[4], 128, index_layer2, batch, layers[1], stride=2)
        self.layer3 = self._make_layer(block, rate[5], rate[6], 256, index_layer3, batch, layers[2], stride=2)
        self.layer4 = self._make_layer(block, rate[7], rate[8], 512, index_layer4, batch, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, plane_expand, full_demension, index, batch, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            print("downsample:", self.inplanes, full_demension, block.expansion)
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, full_demension * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(full_demension * block.expansion),
            )
        # setting: accurate expansion
        index_block_0_dict = {key: index[key] for key in index.keys() if '0.conv3' in key}
        index_block_0_value = list(index_block_0_dict.values())[0]
        # print(index_block_0_value)
        layers = []
        layers.append(
            block(self.inplanes, planes, plane_expand, full_demension, index_block_0_value, batch, stride, downsample))
        #        self.inplanes = planes * block.expansion
        self.inplanes = full_demension * block.expansion

        for i in range(1, blocks):
            index_block_i_dict = {key: index[key] for key in index.keys() if (str(i) + '.conv3') in key}
            index_block_i_value = list(index_block_i_dict.values())[0]
            layers.append(block(self.inplanes, planes, plane_expand, full_demension, index_block_i_value, batch))
            # print(index_block_i_value.size())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #        print("x1 size",x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #        print("x2 size",x.size())

        x = self.layer1(x)
        #        print("x3 size",x.size())
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_small(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_small(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_small(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_small(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_small(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
