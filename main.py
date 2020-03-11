import torch
import torchvision
import time
import sys
import os
import argparse
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import datasets, transforms

import models as model_list
import utils


# global best accuracy
best_acc = 0

def train(args, model, device, train_loader, optimizer, epoch, file_handler):
    model.train()

    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
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
            start = time.time()


def test(args, model, device, test_loader, epoch, file_handler):
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
        dataset = args.model_type.split('_')[0]
        if not os.path.exists(f'checkpoint/{dataset}'):
            os.makedirs(f'checkpoint/{dataset}')
        state = {
            'epoch' : args.epochs,
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
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers' : 1, 'pin_memory': True} if use_cuda else {}
    dataset, model_name = args.model_type.split('_')
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
          datasets.MNIST('data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307, ), (0.3081, ))
                         ])),
          batch_size=args.batch_size, shuffle=True, **kwargs
      )
      test_loader = torch.utils.data.DataLoader(
          datasets.MNIST('data', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307, ), (0.3081, ))
                         ])),
          batch_size=args.test_batch_size, shuffle=True, **kwargs
      )
    elif dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])),
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
        assert os.path.exists(f'{args.checkpoint_model}'), f'Error: {args.checkpoint_model} not found'
        print("=============== restoring from checkpoint ======================")
        checkpoint = torch.load(f'{args.checkpoint_model}')
        model.load_state_dict(checkpoint['weights'])
        start_epoch = checkpoint['epoch']
        print(f"restart epoch {start_epoch} and last accuracy {checkpoint['acc']}%")

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # write experiments to log
    dataset_name, model_type = args.model_type.split('_')
    if not os.path.exists(f'experiments/{dataset_name}/{model_type}'):
        os.makedirs(f'experiments/{dataset_name}/{model_type}')

    log_handler = open(f'experiments/{dataset_name}/{model_type}/{time.time()}.log', 'w')
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, log_handler)
        test(args, model, device, test_loader, epoch, log_handler)
        log_handler.flush()
        scheduler.step()
    log_handler.close()


def init_args():
    parser = argparse.ArgumentParser(description='lottery network training')
    parser.add_argument('--init-type', type=str, default='random',
                        help='the initialization type after pruning, including random and lottery')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training(default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='input batch size for evaluating(default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate(default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='learning rate step gamma(default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--epochs', type=int, default=14,
                        help='number of epochs to train(default: 14)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before print training status')
    parser.add_argument('--model-type', type=str, required=True,
                        help='which model to be trained, whose name components include dataset and real model name'
                        'such as mnist_convnet')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print the details of the model, including weights and shapes')
    parser.add_argument('--features', action='store_true', default=False,
                        help='dump the details for network feature')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume from a last saved checkpoint')
    parser.add_argument('--channel', type=int, default=1,
                        help='the channel for input data')
    parser.add_argument('--height', type=int, default=28,
                        help='the height for input data(default mnist)')
    parser.add_argument('--width', type=int, default=28,
                        help='the width for input data(default mnist)')
    parser.add_argument('--checkpoint-model', type=str,
                        help='a checkpoint model saved on disk')
    parser.add_argument('--model-debug', action='store_true', default=False,
                        help='used to trace all the module')
    parser.add_argument('--module-debug', action='store_true', default=False,
                        help='used to trace a specified type of module')
    parser.add_argument('--module', type=str,
                        help='the target module type name')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()
    main(args)
