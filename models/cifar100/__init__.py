from models.cifar100 import mobilenet as mobilenet_cifar100
from models.cifar100 import resnet as resnet_cifar100
from models.cifar100 import densenet as densenet_cifar100
from models.cifar100 import mobilenetv2 as mobilenetv2_cifar100

# a list of models for cifar100 dataset
__model__ = {
    'mobilenet' : mobilenet_cifar100.build_mobilenet,
    'mobilenetv2' : mobilenetv2_cifar100.build_mobilenetv2,
    'resnet18' : resnet_cifar100.build_resnet18,

    'densenet121' : densenet_cifar100.build_densenet121,
}
