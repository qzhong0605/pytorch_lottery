from ..cifar10.mobilenetv2 import MobileNetV2

def build_mobilenetv2(device):
    return MobileNetV2(device, 'mobilenetv2', num_classes=100).to(device)
