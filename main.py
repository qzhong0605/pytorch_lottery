import torch
import torchvision
import time
import sys
import os
import argparse
import yaml
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
from torchvision import datasets, transforms

import models as model_list
import utils


# global best accuracy
best_acc = 0

HERE = os.path.abspath(os.path.dirname(__file__))

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


def train(args, model, device, train_loader, optimizer, epoch, file_handler, setup, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    avg_acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, avg_acc],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # measure the data load time
        data, target = data.to(device), target.to(device)
        data_time.update(time.time() - end)

        if 'PRUNING' in setup:
            # perform the network pruning if on the pruning interval
            cur_iter = epoch * len(train_loader) + batch_idx
            pruning_interval = setup['PRUNING']['ITERATION']
            iter_idx = 0
            for interval in pruning_interval:
                if interval == cur_iter:
                    pruning_rate = setup['PRUNING']['COMPRESSION_RATE'][iter_idx]
                    model.pruning_network(pruning_rate)
                    break
                iter_idx += 1

        # compute output and loss
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()

        # compute the accuracy
        top1 = accuracy(output, target)
        avg_acc.update(top1[0].item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.log_interval == 0:
            progress.display(batch_idx)
            file_handler.write(f'{epoch},{batch_idx},{losses.avg},{avg_acc.avg}, {batch_time.avg}\n')

            # update the start time
            end = time.time()
            # show weights sparisity for model
            # utils.show_sparsity_of_model(model)


def test(args, model, device, test_loader, epoch, file_handler, setup, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data:', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    avg_acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, losses, avg_acc],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    global best_acc

    with torch.no_grad():
        end = time.time()

        for i, (data, target) in enumerate(test_loader):
            # measure the elapsed time for data load
            data, target = data.to(device), target.to(device)
            data_time.update(time.time() - end)

            # compute output
            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))

            top1 = accuracy(output, target)
            avg_acc.update(top1[0].item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                progress.display(i)
                # update the start time, ignoring the time for output
                end = time.time()

    file_handler.write(f'Test {epoch}, {losses.avg}, {avg_acc.avg}\n')

    # save checkpoint
    if avg_acc.avg > best_acc:
        dataset = setup['DATASET']['NAME']
        model_name = setup['MODEL']
        if not os.path.exists(f'checkpoint/{dataset}/{model_name}'):
            os.makedirs(f'checkpoint/{dataset}/{model_name}')
        state = {
            'epoch' : epoch,
            'weights' : model.state_dict(),
            'acc' : avg_acc.avg
        }
        torch.save(state, f'checkpoint/{dataset}/{model_name}/ckpt_{epoch}.pt')
        print(f'============ save model as checkpoint/{dataset}/{model_name}/ckpt_{epoch}.pt ======================')
        # update global accuracy
        best_acc = avg_acc.avg


def get_target_model(dataset_name, model_name, device):
    model_class = model_list.__model__[dataset_name][model_name]
    return model_class(device)


def main(args):
    if not os.path.exists(args.config):
        print(f"training configure for network doesn't exist")
        sys.exit(1)

    with open(args.config, 'r') as config_stream:
        setup = yaml.safe_load(config_stream)
    use_cuda = False
    if setup['DEVICE']['TYPE'] == 'cuda' and torch.cuda.is_available():
        use_cuda = True
    kwargs = {'num_workers' : 1, 'pin_memory': True} if use_cuda else {}

    dataset = setup['DATASET']['NAME']
    model_name = setup['MODEL']
    if dataset not in model_list.__model__:
        print('dataset {} not supported yet and exit program'.format(dataset))
        sys.exit(1)
    target_models = model_list.__model__[dataset]
    if model_name not in target_models:
        print('model {} not supported yet and exit program'.format(model_name))
        sys.exit(1)

    train_loader = None
    test_loader = None

    if dataset == 'mnist':
      train_loader = torch.utils.data.DataLoader(
          datasets.MNIST(setup['DATASET']['DIR'], train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(setup['DATASET']['MEAN'], setup['DATASET']['DEVIATION'])
                         ])),
          batch_size=setup['DATASET']['TRAIN_BATCHSIZE'], shuffle=True, **kwargs
      )
      test_loader = torch.utils.data.DataLoader(
          datasets.MNIST(setup['DATASET']['DIR'], train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(setup['DATASET']['MEAN'], setup['DATASET']['DEVIATION'])
                         ])),
          batch_size=setup['DATASET']['EVAL_BATCHSIZE'], shuffle=True, **kwargs
      )
    elif dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(setup['DATASET']['DIR'], train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(setup['DATASET']['MEAN'], setup['DATASET']['DEVIATION'])
                             ])),
            batch_size=setup['DATASET']['TRAIN_BATCHSIZE'], shuffle=True, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(setup['DATASET']['DIR'], train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(setup['DATASET']['MEAN'], setup['DATASET']['DEVIATION'])
                             ])),
            batch_size=setup['DATASET']['EVAL_BATCHSIZE'], shuffle=False, **kwargs
        )
    elif dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(setup['DATASET']['DIR'], train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(setup['DATASET']['MEAN'], setup['DATASET']['DEVIATION'])
                             ])),
            batch_size=setup['DATASET']['TRAIN_BATCHSIZE'], shuffle=True, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(setup['DATASET']['DIR'], train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(setup['DATASET']['MEAN'], setup['DATASET']['DEVIATION'])
                             ])),
            batch_size=setup['DATASET']['EVAL_BATCHSIZE'], shuffle=False, **kwargs
        )

    device = torch.device('cuda:{}'.format(setup['DEVICE']['ID']) if use_cuda else 'cpu')
    model = get_target_model(dataset, model_name, device)

    # define loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # show the weight details for model
    if args.verbose:
        utils.show_details_of_module(model)

    # show the approciate details of network features
    if args.features:
        # the input format is NCHW
        model.dump_tensor_shape((1, args.channel, args.height, args.width))

    if args.model_debug:
        model.set_trace()
    elif args.module_debug:
        model.trace_module(args.module)

    start_epoch = 0
    # resume from the previous saved checkpoint
    if args.resume:
        assert os.path.exists(f"{setup['SOLVER']['CHECKPOINT']}"), f"Error: {setup['SOLVER']['CHECKPOINT']} not found"
        print("=============== restoring from checkpoint ======================")
        checkpoint = torch.load(f"{setup['SOLVER']['CHECKPOINT']}")
        model.load_state_dict(checkpoint['weights'])
        start_epoch = checkpoint['epoch']
        print(f"restart epoch {start_epoch} and last accuracy {checkpoint['acc']}%")

    if setup['SOLVER']['OPTIMIZER'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=setup['SOLVER']['LR'],
                               weight_decay=setup['SOLVER']['WEIGHT_DECAY'])
    elif setup['SOLVER']['OPTIMIZER'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=setup['SOLVER']['LR'],
                              weight_decay=setup['SOLVER']['WEIGHT_DECAY'],
                              momentum=setup['SOLVER']['MOMENTUM'])
    elif setup['SOLVER']['OPTIMIZER'] == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=setup['SOLVER']['LR'],
                                   weight_decay=setup['SOLVER']['WEIGHT_DECAY'])
    else:
        print(f"optimizer {setup['SOLVER']['OPTIMIZER']} is not supported yet")
        sys.exit(1)

    if type(setup['SOLVER']['STEP_SIZE']) == int:
        scheduler = StepLR(optimizer, step_size=setup['SOLVER']['STEP_SIZE'],
                           gamma=setup['SOLVER']['GAMMA'])
    else:
        scheduler = MultiStepLR(optimizer, milestones=setup['SOLVER']['STEP_SIZE'],
                                gamma=setup['SOLVER']['GAMMA'])

    # write experiments to log
    if not os.path.exists(f'experiments/{dataset}/{model_name}'):
        os.makedirs(f'experiments/{dataset}/{model_name}')
    log_handler = open(f'experiments/{dataset}/{model_name}/{time.time()}.log', 'w')

    # do the initialization for network pruning
    if 'PRUNING' in setup:
        model.init_pruning_context(init=setup['PRUNING']['INIT_TYPE'],
                                   init_kind=setup['PRUNING']['INIT_KIND'],
                                   op=setup['PRUNING']['OPERATION'],
                                   check_point='{}/{}/{}.pruning'.format(HERE, setup['PRUNING']['DIR'], setup['MODEL']))
    if 'PRUNING' in setup and not os.path.exists('{}/{}'.format(HERE, setup['PRUNING']['DIR'])):
        os.makedirs('{}/{}'.format(HERE, setup['PRUNING']['DIR']))

    for epoch in range(start_epoch, start_epoch + setup['SOLVER']['TOTAL_EPOCHES']):
        train(args, model, device, train_loader, optimizer, epoch, log_handler, setup, criterion)
        test(args, model, device, test_loader, epoch, log_handler, setup, criterion)
        log_handler.flush()
        scheduler.step()
    log_handler.close()


def init_args():
    parser = argparse.ArgumentParser(description='lottery network training')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before print training status')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print the details of the model, including weights and shapes')
    parser.add_argument('--features', action='store_true', default=False,
                        help='dump the details for network feature')
    parser.add_argument('--model-debug', action='store_true', default=False,
                        help='used to trace all the module')
    parser.add_argument('--module-debug', action='store_true', default=False,
                        help='used to trace a specified type of module')
    parser.add_argument('--module', type=str,
                        help='the target module type name')
    parser.add_argument('--config', type=str, required=True,
                        help='model-related hyper-parameters, including dataset, model architecture and schedule')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether to resume training from the last checkpoint state')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()
    main(args)
