from models.cifar10 import resnet
from models.cifar10 import mobilenet
from models.cifar10 import mobilenetv2
from models.cifar10 import densenet
from models.cifar10 import efficientnet

# a list models for cifar10 dataset
__model__ = {
    'resnet18' : resnet.build_resnet18,
    'resnet34' : resnet.build_resnet34,
    'resnet50' : resnet.build_resnet50,
    'resnet101' : resnet.build_resnet101,
    'resnet152' : resnet.build_resnet152,

    'mobilenet' : mobilenet.build_mobilenet,
    'mobilenetv2' : mobilenetv2.build_mobilenetv2,

    'densenet121': densenet.build_densenet121,
    'densenet161': densenet.build_densenet161,
    'densenet169': densenet.build_densenet169,
    'densenet201': densenet.build_densenet201,

    'efficientnetb0': efficientnet.build_efficientnetb0,
}

