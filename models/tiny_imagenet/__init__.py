from models.tiny_imagenet import vgg

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
}
