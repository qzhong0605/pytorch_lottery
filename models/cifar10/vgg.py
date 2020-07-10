'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from models import base_model


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(base_model.HookModule):
    def __init__(self, vgg_name, device, name, num_classes=10):
        super(VGG, self).__init__(device, name)
        vgg_cfg = cfg[vgg_name]
        self._num_classes = num_classes
        self.features = self._make_layers(vgg_cfg)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(vgg_cfg[-2], num_classes)
        )

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def adjust_modules(self, bn_mask):
        """adjust the channels according to the `bn_mask`, which is 0-1 tensor

        Args:
            bn_mask: a list of the mask for scale on batch normalization
        """
        import numpy as np
        import copy
        old_module = copy.deepcopy(self)

        new_cfg = []
        layer_bn_idx = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                new_cfg.append(int(torch.sum(bn_mask[layer_bn_idx]).item()))
                layer_bn_idx += 1
            elif isinstance(module, nn.MaxPool2d):
                new_cfg.append('M')
        self.features = self._make_layers(new_cfg)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(new_cfg[-2], self._num_classes)
        )

        # fill the weights after modifying the network architecture
        layer_bn_idx = 0
        start_mask = torch.ones(3)
        end_mask = bn_mask[layer_bn_idx]
        for m0, m1 in zip(old_module.modules(), self.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                bn_indices = np.squeeze(np.argwhere(end_mask))
                m1.weight.data = m0.weight[bn_indices.tolist()].clone().to(self._device)
                m1.bias.data = m0.bias[bn_indices.tolist()].clone().to(self._device)
                m1.running_mean = m0.running_mean[bn_indices.tolist()].clone().to(self._device)
                m1.running_var = m0.running_var[bn_indices.tolist()].clone().to(self._device)

                start_mask = end_mask
                layer_bn_idx += 1
                if layer_bn_idx < len(bn_mask):
                    end_mask = bn_mask[layer_bn_idx]
            elif isinstance(m0, nn.Conv2d):
                ch_indices = np.squeeze(np.argwhere(start_mask))
                filter_indices = np.squeeze(np.argwhere(end_mask))
                w1 = m0.weight[::, ch_indices.tolist(), ::, ::].clone()
                m1.weight.data = w1[filter_indices, ::, ::, ::].clone().to(self._device)
                m1.bias.data = m0.bias[filter_indices].clone().to(self._device)
            elif isinstance(m0, nn.Linear):
                fc_indices = np.squeeze(np.argwhere(start_mask))
                m1.weight.data = m0.weight[::,fc_indices].clone().to(self._device)
                m1.bias.data = m0.bias.clone().to(self._device)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out

def build_vgg11(device):
    return VGG('VGG11', device, 'vgg11').to(device)

def build_vgg13(device):
    return VGG('VGG13', device, 'vgg13').to(device)

def build_vgg16(device):
    return VGG('VGG16', device, 'vgg16').to(device)

def build_vgg19(device):
    return VGG('VGG19', device, 'vgg19').to(device)
