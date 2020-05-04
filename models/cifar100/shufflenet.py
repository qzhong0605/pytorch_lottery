from ..cifar10.shufflenet import ShuffleNet

def build_shufflenetg2(device):
    cfg = {
        'out_planes': [200,400,800],
        'num_blocks': [4,8,4],
        'groups': 2
    }
    return ShuffleNet(cfg, device, 'shufflenetg2', num_classes=100).to(device)

def build_shufflenetg3(device):
    cfg = {
        'out_planes': [240,480,960],
        'num_blocks': [4,8,4],
        'groups': 3
    }
    return ShuffleNet(cfg, device, 'shufflenetg3', num_classes=100).to(device)
