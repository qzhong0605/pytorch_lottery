import models.mnist as mnist_model
import models.cifar10  as cifar10_model
import models.cifar100 as cifar100_model
import models.imagenet as imagenet_model

# a list of all the predefined models
__model__ = {
    'mnist' : mnist_model.__model__,
    'cifar10' : cifar10_model.__model__,
    'cifar100' : cifar100_model.__model__,
    'imagenet' : imagenet_model.__model__,
}
