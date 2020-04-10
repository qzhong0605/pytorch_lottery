from models.cifar100 import mobilenet as mobilenet_cifar100

# a list of models for cifar100 dataset
__model__ = {
  'mobilenet' : mobilenet_cifar100.build_mobilenet,
}
