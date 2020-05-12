from models.cifar10 import resnet
from models.cifar10 import mobilenet
from models.cifar10 import mobilenetv2
from models.cifar10 import densenet
from models.cifar10 import efficientnet
from models.cifar10 import regnet
from models.cifar10 import shufflenet
from models.cifar10 import vgg
from models.cifar10 import resnext

# a list models for cifar10 dataset
__model__ = {
    'resnet18' : resnet.build_resnet18,
    'resnet34' : resnet.build_resnet34,
    'resnet50' : resnet.build_resnet50,
    'resnet101' : resnet.build_resnet101,
    'resnet152' : resnet.build_resnet152,

    'mobilenet' : mobilenet.build_mobilenet,
    'mobilenetv2' : mobilenetv2.build_mobilenetv2,

    'shufflenetg2': shufflenet.build_shufflenetg2,
    'shufflenetg3': shufflenet.build_shufflenetg3,

    'densenet121': densenet.build_densenet121,
    'densenet161': densenet.build_densenet161,
    'densenet169': densenet.build_densenet169,
    'densenet201': densenet.build_densenet201,

    'efficientnetb0': efficientnet.build_efficientnetb0,
    'regnetx_200mf': regnet.build_regnetx_200mf,

    'vgg11': vgg.build_vgg11,
    'vgg13': vgg.build_vgg13,
    'vgg16': vgg.build_vgg16,
    'vgg19': vgg.build_vgg19,

    'resnext29_2x64d' : resnext.build_resnext29_2x64d,
    'resnext29_4x64d' : resnext.build_resnext29_4x64d,
    'resnext29_8x64d' : resnext.build_resnext29_8x64d,
    'resnext29_32x64d': resnext.build_resnext29_32x4d,
}

