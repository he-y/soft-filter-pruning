def cifar_resnet_flop(layer=110, prune_rate=1):
    '''
    :param layer: the layer of Resnet for Cifar, including 110, 56, 32, 20
    :param prune_rate: 1 means baseline
    :return: flop of the network
    '''
    flop = 0
    channel = [16, 32, 64]
    width = [32, 16, 8]

    stage = int(layer / 3)
    for index in range(0, layer, 1):
        if index == 0:  # first conv layer before block
            flop += channel[0] * width[0] * width[0] * 9 * 3 * prune_rate
        elif index in [1, 2]:  # first block of first stage
            flop += channel[0] * width[0] * width[0] * 9 * channel[0] * (prune_rate ** 2)
        elif 2 < index <= stage:  # other blocks of first stage
            if index % 2 != 0:
                # first layer of block, only output channal reduced, input channel remain the same
                flop += channel[0] * width[0] * width[0] * 9 * channel[0] * (prune_rate)
            elif index % 2 == 0:
                # second layer of block, both input and output channal reduced
                flop += channel[0] * width[0] * width[0] * 9 * channel[0] * (prune_rate ** 2)
        elif stage < index <= stage * 2:  # second stage
            if index % 2 != 0:
                flop += channel[1] * width[1] * width[1] * 9 * channel[1] * (prune_rate)
            elif index % 2 == 0:
                flop += channel[1] * width[1] * width[1] * 9 * channel[1] * (prune_rate ** 2)
        elif stage * 2 < index <= stage * 3:  # third stage
            if index % 2 != 0:
                flop += channel[2] * width[2] * width[2] * 9 * channel[2] * (prune_rate)
            elif index % 2 == 0:
                flop += channel[2] * width[2] * width[2] * 9 * channel[2] * (prune_rate ** 2)

    # offset for dimension change between blocks
    offset1 = channel[1] * width[1] * width[1] * 9 * channel[1] * prune_rate - channel[1] * width[1] * width[1] * 9 * \
              channel[0] * prune_rate
    offset2 = channel[2] * width[2] * width[2] * 9 * channel[2] * prune_rate - channel[2] * width[2] * width[2] * 9 * \
              channel[1] * prune_rate
    flop = flop - offset1 - offset2
    # print(flop)
    return flop


def cal_cifar_resnet_flop(layer, prune_rate):
    '''
    :param layer:  the layer of Resnet for Cifar, including 110, 56, 32, 20
    :param prune_rate: 1 means baseline
    :return:
    '''
    pruned_flop = cifar_resnet_flop(layer, prune_rate)
    baseline_flop = cifar_resnet_flop(layer, 1)

    print(
        "pruning rate of layer {:d} is {:.1f}, pruned FLOP is {:.0f}, "
        "baseline FLOP is {:.0f}, FLOP reduction rate is {:.4f}"
        .format(layer, prune_rate, pruned_flop, baseline_flop, 1 - pruned_flop / baseline_flop))


if __name__ == '__main__':
    layer_list = [110, 56, 32, 20]
    pruning_rate_list = [0.9, 0.8, 0.7]
    for layer in layer_list:
        for pruning_rate in pruning_rate_list:
            cal_cifar_resnet_flop(layer, pruning_rate)
