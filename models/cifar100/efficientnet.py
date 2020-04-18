from ..cifar10.efficientnet import EfficientNet

def build_efficientnetb0(device):
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_planes': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
    }
    return EfficientNet(cfg, device, 'efficientnetb0', num_classes=100).to(device)


