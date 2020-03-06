from models.cifar10 import resnet

# a list models for cifar10 dataset
__model__ = {
    'resnet18' : resnet.build_resnet18,
    'resnet34' : resnet.build_resnet34,
    'resnet50' : resnet.build_resnet50,
    'resnet101' : resnet.build_resnet101,
    'resnet152' : resnet.build_resnet152,
}

