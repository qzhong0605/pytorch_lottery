from models.imagenet import vgg
from models.imagenet import resnet

# a list of models for tiny imagenet dataset

__model__ = {
    'vgg11' : vgg.build_vgg11,
    'vgg11bn' : vgg.build_vgg11bn,
    'vgg13' : vgg.build_vgg13,
    'vgg13bn' : vgg.build_vgg13bn,
    'vgg16' : vgg.build_vgg16,
    'vgg16bn' : vgg.build_vgg16bn,
    'vgg19' : vgg.build_vgg19,
    'vgg19bn' : vgg.build_vgg19bn,

    'resnet18' : resnet.build_resnet18,
    'resnet34' : resnet.build_resnet34,
    'resnet50' : resnet.build_resnet50,
    'resnet101' : resnet.build_resnet101,
    'resnet152' : resnet.build_resnet152,
    'resnext50_32x4d' : resnet.build_resnext50_32x4d,
    'resnext101_32x8d' : resnet.build_resnext101_32x8d,
    'wide_resnet50_2' : resnet.build_wide_resnet50_2,
    'build_wide_resnet101_2' : resnet.build_wide_resnet101_2,
}
