import torch
import argparse
import time
import re
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parameter import Parameter

module_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))

quant_module_names = sorted(name for name in models.quantization.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='evaluate models from torchvsion on imagenet dataset')
parser.add_argument('-a', '--arch', default='resnet18', choices=module_names,
                    help='model architecture: ' +
                        ' | '.join(module_names) +
                        ' and quantization architecture: ' +
                        ' | '.join(quant_module_names) +
                        ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size')
parser.add_argument('--checkpoint', type=str, required=True, help='the local model file')
parser.add_argument('--gpu', type=int, default=0, help='the gpu id, where to run the model')
parser.add_argument('--data', type=str, required=True, help='the dir for imagenet dataset')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-q', '--quantize', dest='quantize', action='store_true',
                    help='use the quantized model')
parser.add_argument('--skip-module', action='store_true',
                    help='skip the whole module of network')
# these weights are skipped to be loaded into network
skip_weights = []

args = parser.parse_args()

def mask_filter(weight:torch.Tensor, filters=[]):
    """mask a list of `filter` dimension of weights and set them to zero"""
    mask = torch.ones(weight.shape)
    mask[::, filters] = 0.
    old_data = weight.data.cpu()
    new_data = old_data * mask
    weight.data = new_data.to(weight.device)

def mask_number(weight:torch.Tensor, numbers=[]):
    """mask a list of `number` dimension of weights and set them to zero"""
    mask = torch.ones(weight.shape)
    mask[numbers] = 0.
    old_data = weight.data.cpu()
    new_data = old_data * mask
    weight.data = new_data.to(weight.device)

def mask_element(weight:torch.Tensor, elements=[]):
    r"""mask a list of `elements` of weights and set them to zero

    :param elements: a list of elements index. The index is a four-tuple data
    """
    mask = torch.ones(weight.shape)
    for _element in elements:
        mask[_element] = 0.
    old_data = weight.data.cpu()
    new_data = old_data * mask
    weight.data = new_data.to(weight.device)

def mask_value(weight:torch.Tensor, thresh_value:float):
    """mask the values less than `thresh_value` and set them to zero"""
    weight.data = torch.where(weight.abs() < thresh_value,
                              torch.zeros(weight.shape), weight.data)

def count_zeros(weight:torch.Tensor):
    """return the number of zeros"""
    _temp = torch.where(weight.abs() < 1e-6, torch.ones(weight.shape), torch.zeros(weight.shape))
    return _temp.sum().item()

def percentile(weight:torch.Tensor, q:float):
    """return the `q`-th percentile of the abs function of input weight tensor"""
    k = 1 + round(float(q) * (weight.numel() - 1))
    result = weight.view(-1).abs().kthvalue(k).values.item()
    return result

def percentile_nonzeros(weight:torch.Tensor, q:float):
    """return the `q`-th percentile of the abs function of the nonzeros on input weight tensor"""
    weight_np = weight.data.cpu().numpy()
    weight_nonzero_np = weight_np[weight_np.nonzero()]
    new_weight = torch.from_numpy(weight_nonzero_np)
    return percentile(new_weight, q)


################################################################################
#
# skip torch module, whose input is the same with output tensor
#
################################################################################
class SkipModule(nn.Module):
    """It's a skip module, which does nothing."""
    def __init__(self, skips=[]):
        super(SkipModule, self).__init__()
        skip_weights.extend(skips)

    def forward(self, x):
        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def main():
    if args.quantize:
        model = models.quantization.__dict__[args.arch]()
    else:
        model = models.__dict__[args.arch]()

    if args.gpu is not None:
        print('Use GPU: {}'.format(args.gpu))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.arch == 'inception_v3':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data, transforms.Compose([
                transforms.Resize(299), transforms.CenterCrop(299),
                transforms.ToTensor(),
                normalize
            ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
    elif args.arch == 'googlenet':
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data, transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
    else:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.data, transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    device = torch.device('cpu')
    if args.gpu is not None:
        device = torch.device(args.gpu)
    model.to(device)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    state_dict = torch.load(args.checkpoint)
    if args.skip_module:
        for _weight_name in skip_weights:
            state_dict.pop(_weight_name)

    if args.arch.startswith('densenet'):
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
    model.load_state_dict(state_dict)
    validate(val_loader, model, criterion, args)

if __name__ == "__main__":
    main()
