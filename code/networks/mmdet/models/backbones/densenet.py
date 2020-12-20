import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os

from mmcv.runner import load_checkpoint
#from mmdet.utils import get_root_logger
#from ..registry import BACKBONES
from ..builder import BACKBONES

# features_2 = 0
# features_4 = 0
# features_6 = 0
# features_8 = 0

class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


@BACKBONES.register_module
class DenseNet(nn.Module):
    # num_init_features, growth_rate, block_config
    arch_settings = {
        121: (64, 32, (6, 12, 24, 16)),
        161: (96, 48, (6, 12, 36, 24)),
        169: (64, 32, (6, 12, 32, 32)),
        201: (64, 32, (6, 12, 48, 32)),
    }
    # firstpoolfeatures = []

    def __init__(self, depth=121,
                 num_stages=4,
                 frozen_stages=-1,
                 norm_eval=True,
                 bn_size=4,
                 drop_rate=0
                 ):
        super(DenseNet, self).__init__()
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        num_init_features, growth_rate, block_config = self.arch_settings[depth]

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        _DenseBlock
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self._freeze_stages()

    def _freeze_stages(self):
        # 'Not implemented yet.'
        return

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger, isolddensenetload=True)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        features = self.features.conv0(x)
        features = self.features.norm0(features)
        features = self.features.relu0(features)
        features = self.features.pool0(features)
        # outs.append(features)
        # print('----------------------')
        # print(features.shape)   #torch.Size([2, 64, 400, 592])
        # print(features)
        # self.firstpoolfeatures = features.detach()
        features = self.features.denseblock1(features)
        features = self.features.transition1(features)
        outs.append(features)
        features = self.features.denseblock2(features)
        features = self.features.transition2(features)
        outs.append(features)
        features = self.features.denseblock3(features)
        features = self.features.transition3(features)
        outs.append(features)
        features = self.features.denseblock4(features)
        features = self.features.norm5(features)
        outs.append(features)
        # print(outs)
        # print(outs.shape)

        # global features_2, features_4, features_6, features_8

        # features_1 = self.features.denseblock1(features)
        # features_2 = self.features.transition1(features_1)
        # outs.append(features_2)
        # features_3 = self.features.denseblock2(features_2)
        # features_4 = self.features.transition2(features_3)
        # outs.append(features_4)
        # features_5 = self.features.denseblock3(features_4)
        # features_6 = self.features.transition3(features_5)
        # outs.append(features_6)
        # features_7 = self.features.denseblock4(features_6)
        # features_8 = self.features.transition4(features_7)
        # outs.append(features_8)
        return tuple(outs)



    def train(self, mode=True):
        super(DenseNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    for param in m.parameters():
                        param.requires_grad = True
                    m.eval()
