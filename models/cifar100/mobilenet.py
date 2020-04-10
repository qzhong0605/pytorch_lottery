from ..cifar10.mobilenet import MobileNet

def build_mobilenet(device):
    return MobileNet(device, 'mobilenet', num_classes=100).to(device)
