from ..cifar10.densenet import DenseNet, Bottleneck

def build_densenet121(device):
    return DenseNet(Bottleneck, device, 'densenet121', [6,12,24,16], growth_rate=32, num_classes=100).to(device)

def build_densenet169(device):
    return DenseNet(Bottleneck, device, 'densenet169', [6,12,32,32], growth_rate=32, num_classes=100).to(device)

def build_densenet201(device):
    return DenseNet(Bottleneck, device, 'densenet201', [6,12,48,32], growth_rate=32, num_classes=100).to(device)

def build_densenet161(device):
    return DenseNet(Bottleneck, device, 'densenet161', [6,12,36,24], growth_rate=48, num_classes=100).to(device)
