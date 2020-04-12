from ..cifar10.resnet import ResNet
from ..cifar10.resnet import BasicBlock, BottleNeck

def build_resnet18(device):
    return ResNet(BasicBlock, [2,2,2,2], device, 'resnet18', num_classes=100).to(device)

def build_resnet34(device):
    return ResNet(BasicBlock, [3,4,6,3], device, 'resnet34', num_classes=100).to(device)

def build_resnet50(device):
    return ResNet(BottleNeck, [3,4,6,3], device, 'resnet50', num_classes=100).to(device)

def build_resnet101(device):
    return ResNet(BottleNeck, [3,4,23,3], device, 'resnet101', num_classes=100).to(device)

def build_resnet152(device):
    return ResNet(BottleNeck, [3,8,36,3], device, 'resnet152', num_classes=100).to(device)
