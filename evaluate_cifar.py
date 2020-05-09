import torch
import argparse
import time
import re
import sys
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parameter import Parameter

import models as model_list

parser = argparse.ArgumentParser(description='evaluate models for cifar10/cifar100 dataset')
parser.add_argument('-a', '--arch', default='resnet18', type=str,
                    help='')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size')
parser.add_argument('--checkpoint', type=str, required=True, help='the local model file')
parser.add_argument('--gpu', type=int, default=0, help='the gpu id, where to run the model')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset to be validated')
parser.add_argument('--data-dir', type=str, required=True,
                    help='where to hold the dataset')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
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
    old_data = weight.data.cpu()
    new_data = torch.where(old_data.abs() < thresh_value,
                              torch.zeros(old_data.shape), old_data)
    weight.data = new_data.to(weight.device)

def global_mask_value(tensors:list, thresh_value:float):
    """mask the values on tensors less than `thresh_value` and set them to zero"""
    for tensor in tensors:
        mask_value(tensor, thresh_value)

def count_zeros(weight:torch.Tensor):
    """return the number of zeros"""
    cpu_data = weight.cpu()
    _temp = torch.where(cpu_data.abs() < 1e-6, torch.ones(cpu_data.shape), torch.zeros(cpu_data.shape))
    return _temp.sum().item()

def global_count_zeros(tensors:list):
    """return the number of zeros for input list"""
    res = 0
    for tensor in tensors:
        res += count_zeros(tensor)
    return res

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

def global_percentile(tensors:list, q:float):
    """return the `q`-th percentile of the tensors in the list"""
    cat_res = torch.cat([tensor.cpu().view(-1) for tensor in tensors])
    return percentile(cat_res, q)

def global_percentile_nonzeros(tensors:list, q:float):
    """return the `q`-th percentile of the abs function of the nonzeros on input list of tensor"""
    cat_res = torch.cat([tensor.cpu().view(-1) for tensor in tensors])
    return percentile_nonzeros(cat_res, q)

def count_sparsity(model:nn.Module):
    """return the sparsity of the input model. Here, only the `weight` is computed"""
    zeros = 0
    totals = 0
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        zeros += count_zeros(param)
        totals += param.numel()
    return (zeros * 1. / totals)

def count_sparsity_bn(model:nn.Module):
    """return the sparsity of the input model on all batch norm operations"""
    zeros = 0
    totals = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 1:
            zeros += count_zeros(param)
            totals += param.numel()
    return (zeros * 1. / totals)

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

def get_target_model(dataset, model_name, device):
    model_func = model_list.__model__[dataset][model_name]
    return model_func(device)


def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0].item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc {top1.avg:.3f} Loss {losses.avg:.3f}'
              .format(top1=top1, losses=losses))


def main():
    # find an model
    if args.arch not in model_list.__model__[args.dataset]:
        print(f'{args.arch} does not implement yet')
        sys.exit(-1)

    device = torch.device('cpu')
    if args.gpu is not None:
        print('Use GPU: {}'.format(args.gpu))
        device = torch.device(args.gpu)
    model = get_target_model(args.dataset, args.arch, device)

    # get and setup data loader
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    if args.dataset == 'cifar100':
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_dir, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize
                             ])),
            batch_size=args.batch_size, shuffle=False
        )
    elif args.dataset == 'cifar10':
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize
                             ])),
            batch_size=args.batch_size, shuffle=False
        )
    elif args.dataset == 'mnist':
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_dir, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.1307], [0.3081])
                           ])),
            batch_size=args.batch_size, shuffle=False
        )

    criterion = nn.CrossEntropyLoss().to(device)

    # load the checkpoint model
    state_dict = torch.load(args.checkpoint)
    print('epoch: {}  acc: {}'.format(state_dict['epoch'], state_dict['acc']))
    if args.skip_module:
        for _weight_name in skip_weights:
            state_dict.pop(_weight_name)

    model_state = state_dict['weights']
    model.load_state_dict(model_state)
    validate(val_loader, model, criterion, device)

if __name__ == "__main__":
    main()
