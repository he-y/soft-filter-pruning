# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os,sys
import shutil
import pdb, time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import convert_secs2time, time_string, time_file_str
# from models import print_log
import models
import random
import numpy as np
import copy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--save_dir', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=5, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# compress rate
parser.add_argument('--rate', type=float, default=0.9, help='compress rate of model')
parser.add_argument('--layer_begin', type=int, default=3, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=3, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--skip_downsample', type=int, default=1, help='compress layer of model')
parser.add_argument('--get_small', dest='get_small', action='store_true', help='whether a big or small model')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

args.prefix = time_file_str()


def main():
    best_prec1 = 0

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'gpu-time.{}.{}.log'.format(args.arch, args.prefix)), 'w')

    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](pretrained=False)
    print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("Skip downsample : {}".format(args.skip_downsample), log)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            state_dict = checkpoint['state_dict']
            state_dict = remove_module_dict(state_dict)
            model.load_state_dict(state_dict)
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Scale(256),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.get_small:
        big_path = os.path.join(args.save_dir, "big_model.pt")
        torch.save(model, big_path)
        small_model = get_small_model(model.cpu())
        # small_model = torch.load('small_model.pt')
        small_path = os.path.join(args.save_dir, "small_model.pt")
        torch.save(small_model, small_path)

        if args.use_cuda:
            model = model.cuda()
            small_model = small_model.cuda()
        print('evaluate: big')
        print('big model accu', validate(val_loader, model, criterion, log))

        print('evaluate: small')
        print('small model accu', validate(val_loader, small_model, criterion, log))


def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        if args.use_cuda:
            input, target = input.cuda(), target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5), log)

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg), log)

    return top1.avg


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def remove_module_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def import_sparse(model):
    checkpoint = torch.load('/data/yahe/imagenet/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("sparse_model_loaded")
    return model


def check_channel(tensor):
    size_0 = tensor.size()[0]
    size_1 = tensor.size()[1] * tensor.size()[2] * tensor.size()[3]
    tensor_resize = tensor.view(size_0, -1)
    # indicator: if the channel contain all zeros
    channel_if_zero = np.zeros(size_0)
    for x in range(0, size_0, 1):
        channel_if_zero[x] = np.count_nonzero(tensor_resize[x].cpu().numpy()) != 0
    # indices = (torch.LongTensor(channel_if_zero) != 0 ).nonzero().view(-1)

    indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])
    # indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])

    zeros = (channel_if_zero == 0).nonzero()[0]
    indices_zero = torch.LongTensor(zeros) if zeros != [] else []

    return indices_zero, indices_nonzero


def extract_para(big_model):
    '''
    :param model:
    :param batch_size:
    :return: num_for_construc: number of remaining filter,
             [conv1,stage1,stage1_expend,stage2,stage2_expend,stage3,stage3_expend,stage4,stage4_expend]

             kept_filter_per_layer: number of remaining filters for every layer
             kept_index_per_layer: filter index of remaining channel
             model: small model
    '''
    item = list(big_model.state_dict().items())
    print("length of state dict is", len(item))
    try:
        assert len(item) in [102, 182, 267, 522]
        print("state dict length is one of 102, 182, 267, 522")
    except AssertionError as e:
        print("False state dict")

    # indices_list = []
    kept_index_per_layer = {}
    kept_filter_per_layer = {}
    pruned_index_per_layer = {}

    for x in range(0, len(item) - 2, 5):
        indices_zero, indices_nonzero = check_channel(item[x][1])
        # indices_list.append(indices_nonzero)
        pruned_index_per_layer[item[x][0]] = indices_zero
        kept_index_per_layer[item[x][0]] = indices_nonzero
        kept_filter_per_layer[item[x][0]] = indices_nonzero.shape[0]

    # add 'module.' if state_dict are store in parallel format
    # state_dict = ['module.' + x for x in state_dict]

    if len(item) == 102 or len(item) == 182:
        basic_block_flag = ['conv1.weight',
                            'layer1.0.conv1.weight', 'layer1.0.conv2.weight',
                            'layer2.0.conv1.weight', 'layer2.0.conv2.weight',
                            'layer3.0.conv1.weight', 'layer3.0.conv2.weight',
                            'layer4.0.conv1.weight', 'layer4.0.conv2.weight']
        constrct_flag = basic_block_flag
        block_flag = "conv2"
    elif len(item) == 267 or len(item) == 522:
        bottle_block_flag = ['conv1.weight',
                             'layer1.0.conv1.weight', 'layer1.0.conv3.weight',
                             'layer2.0.conv1.weight', 'layer2.0.conv3.weight',
                             'layer3.0.conv1.weight', 'layer3.0.conv3.weight',
                             'layer4.0.conv1.weight', 'layer4.0.conv3.weight']
        constrct_flag = bottle_block_flag
        block_flag = "conv3"

    # number of nonzero channel in conv1, and four stages
    num_for_construct = []
    for key in constrct_flag:
        num_for_construct.append(kept_filter_per_layer[key])

    index_for_construct = dict(
        (key, value) for (key, value) in kept_index_per_layer.items() if block_flag in key)
    bn_value = get_bn_value(big_model, block_flag, pruned_index_per_layer)
    if len(item) == 102:
        small_model = models.resnet18_small(index=kept_index_per_layer, bn_value=bn_value,
                                            num_for_construct=num_for_construct)
    if len(item) == 182:
        small_model = models.resnet34_small(index=kept_index_per_layer, bn_value=bn_value,
                                            num_for_construct=num_for_construct)
    if len(item) == 267:
        small_model = models.resnet50_small(index=kept_index_per_layer, bn_value=bn_value,
                                            num_for_construct=num_for_construct)
    if len(item) == 522:
        small_model = models.resnet101_small(index=kept_index_per_layer, bn_value=bn_value,
                                             num_for_construct=num_for_construct)
    return kept_index_per_layer, pruned_index_per_layer, block_flag, small_model


def get_bn_value(big_model, block_flag, pruned_index_per_layer):
    big_model.eval()
    bn_flag = "bn3" if block_flag == "conv3" else "bn2"
    key_bn = [x for x in big_model.state_dict().keys() if "bn3" in x]
    layer_flag_list = [[x[0:6], x[7], x[9:12], x] for x in key_bn if "weight" in x]
    # layer_flag_list = [['layer1', "0", "bn3",'layer1.0.bn3.weight']]
    bn_value = {}

    for layer_flag in layer_flag_list:
        module_bn = big_model._modules.get(layer_flag[0])._modules.get(layer_flag[1])._modules.get(layer_flag[2])
        num_feature = module_bn.num_features
        act_bn = module_bn(Variable(torch.zeros(1, num_feature, 1, 1)))

        index_name = layer_flag[3].replace("bn", "conv")
        index = Variable(torch.LongTensor(pruned_index_per_layer[index_name]))
        act_bn = torch.index_select(act_bn, 1, index)

        select = Variable(torch.zeros(1, num_feature, 1, 1))
        select.index_add_(1, index, act_bn)

        bn_value[layer_flag[3]] = select
    return bn_value


def get_small_model(big_model):
    indice_dict, pruned_index_per_layer, block_flag, small_model = extract_para(big_model)
    big_state_dict = big_model.state_dict()
    small_state_dict = {}
    keys_list = list(big_state_dict.keys())
    # print("keys_list", keys_list)
    for index, [key, value] in enumerate(big_state_dict.items()):
        # all the conv layer excluding downsample layer
        flag_conv_ex_down = not 'bn' in key and not 'downsample' in key and not 'fc' in key
        # downsample conv layer
        flag_down = 'downsample.0' in key
        # value for 'output' dimension: all the conv layer including downsample layer
        if flag_conv_ex_down or flag_down:
            small_state_dict[key] = torch.index_select(value, 0, indice_dict[key])
            conv_index = keys_list.index(key)
            # 4 following bn layer, bn_weight, bn_bias, bn_runningmean, bn_runningvar
            for offset in range(1, 5, 1):
                bn_key = keys_list[conv_index + offset]
                small_state_dict[bn_key] = torch.index_select(big_state_dict[bn_key], 0, indice_dict[key])
            # value for 'input' dimension
            if flag_conv_ex_down:
                # first layer of first block
                if 'layer1.0.conv1.weight' in key:
                    small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict['conv1.weight'])
                # just conv1 of block, the input dimension should not change for shortcut
                elif not "conv1" in key:
                    conv_index = keys_list.index(key)
                    # get the last con layer
                    key_for_input = keys_list[conv_index - 5]
                    # print("key_for_input", key, key_for_input)
                    small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict[key_for_input])
            # only the first downsample layer should change as conv1 reduced
            elif 'layer1.0.downsample.0.weight' in key:
                small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict['conv1.weight'])
        elif 'fc' in key:
            small_state_dict[key] = value

    if len(set(big_state_dict.keys()) - set(small_state_dict.keys())) != 0:
        print("different keys of big and small model",
              sorted(set(big_state_dict.keys()) - set(small_state_dict.keys())))
        for x, y in zip(small_state_dict.keys(), small_model.state_dict().keys()):
            if small_state_dict[x].size() != small_model.state_dict()[y].size():
                print("difference with model and dict", x, small_state_dict[x].size(),
                      small_model.state_dict()[y].size())

    small_model.load_state_dict(small_state_dict)

    return small_model


if __name__ == '__main__':
    main()
