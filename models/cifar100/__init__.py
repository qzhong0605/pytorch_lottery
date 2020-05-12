from models.cifar100 import mobilenet as mobilenet_cifar100
from models.cifar100 import resnet as resnet_cifar100
from models.cifar100 import densenet as densenet_cifar100
from models.cifar100 import mobilenetv2 as mobilenetv2_cifar100
from models.cifar100 import efficientnet as efficientnet_cifar100
from models.cifar100 import shufflenet as shufflenet_cifar100
from models.cifar100 import vgg as vgg_cifar100

# a list of models for cifar100 dataset
__model__ = {
    'mobilenet' : mobilenet_cifar100.build_mobilenet,
    'mobilenetv2' : mobilenetv2_cifar100.build_mobilenetv2,
    'resnet18' : resnet_cifar100.build_resnet18,

    'densenet121' : densenet_cifar100.build_densenet121,
    'densenet169' : densenet_cifar100.build_densenet169,
    'densenet201' : densenet_cifar100.build_densenet201,
    'densenet161' : densenet_cifar100.build_densenet169,

    'efficientnetb0': efficientnet_cifar100.build_efficientnetb0,

    'shufflenetg2' : shufflenet_cifar100.build_shufflenetg2,
    'shufflenetg3' : shufflenet_cifar100.build_shufflenetg3,

    'vgg11' : vgg_cifar100.build_vgg11,
    'vgg13' : vgg_cifar100.build_vgg13,
    'vgg16' : vgg_cifar100.build_vgg16,
    'vgg19' : vgg_cifar100.build_vgg19,
}
