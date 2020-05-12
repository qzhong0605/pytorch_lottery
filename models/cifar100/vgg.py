from ..cifar10.vgg import VGG

def build_vgg11(device):
    return VGG('VGG11', device, 'vgg11', num_classes=100).to(device)

def build_vgg13(device):
    return VGG('VGG13', device, 'vgg13', num_classes=100).to(device)

def build_vgg16(device):
    return VGG('VGG16', device, 'vgg16', num_classes=100).to(device)

def build_vgg19(device):
    return VGG('VGG19', device, 'vgg19', num_classes=100).to(device)
