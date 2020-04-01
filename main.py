import torch
import torchvision
import time
import sys
import os
import argparse
import yaml
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
from torchvision import datasets, transforms

import models as model_list
import utils


# global best accuracy
best_acc = 0

HERE = os.path.abspath(os.path.dirname(__file__))

def train(args, model, device, train_loader, optimizer, epoch, file_handler, setup):
    model.train()

    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if 'PRUNING' in setup:
            # perform the network pruning if on the pruning interval
            cur_iter = epoch * len(train_loader) + batch_idx
            pruning_interval = setup['PRUNING']['ITERATION']
            iter_idx = 0
            for interval in pruning_interval:
                if interval == cur_iter:
                    pruning_rate = setup['PRUNING']['COMPRESSION_RATE'][iter_idx]
                    model.pruning_with_percentile(pruning_rate)
                    break
                iter_idx += 1

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # track the time overhead
            end = time.time()

            # track the training accuracy
            pred = output.argmax(dim=1, keepdim=True)    # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Train Accuracy: {:.3f}%\t Time: {:.6f} s/iter'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * correct / len(data),
                (end - start) * 1. / args.log_interval
            ))
            file_handler.write(f'{epoch},{batch_idx},{loss.item()},{correct* 1./len(data)}, {(end-start) * 1./args.log_interval}\n')

            # show weights sparisity for model
            # utils.show_sparsity_of_model(model)
            start = time.time()


def test(args, model, device, test_loader, epoch, file_handler, setup):
    model.eval()
    global best_acc

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    file_handler.write(f'Test {epoch}, {test_loss}, {correct* 1./len(test_loader.dataset)}\n')

    # save checkpoint
    acc = 100. * correct / len(test_loader.dataset)
    if acc > best_acc:
        dataset = setup['DATASET']['NAME']
        if not os.path.exists(f'checkpoint/{dataset}'):
            os.makedirs(f'checkpoint/{dataset}')
        state = {
            'epoch' : epoch,
            'weights' : model.state_dict(),
            'acc' : acc
        }
        torch.save(state, f'checkpoint/{dataset}/ckpt_{epoch}.pt')
        print(f'============ save model as checkpoint/{dataset}/ckpt_{epoch}.pt ======================')
        # update global accuracy
        best_acc = acc


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
            batch_size=setup['DATASET']['EVAL_BATCHSIZE'], shuffle=True, **kwargs
        )

    device = torch.device('cuda' if use_cuda else 'cpu')
    model = get_target_model(dataset, model_name, device)
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
    model.apply_weight_mask()
    model.init_weight_mask()
    if 'PRUNING' in setup and not os.path.exists('{}/{}'.format(HERE, setup['PRUNING']['DIR'])):
        os.makedirs('{}/{}'.format(HERE, setup['PRUNING']['DIR']))

    for epoch in range(start_epoch, start_epoch + setup['SOLVER']['TOTAL_EPOCHES']):
        train(args, model, device, train_loader, optimizer, epoch, log_handler, setup)
        test(args, model, device, test_loader, epoch, log_handler, setup)
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
