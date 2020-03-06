import models.mnist as mnist_model
import models.cifar10  as cifar10_model

# a list of all the predefined models
__model__ = {
    'mnist' : mnist_model.__model__,
    'cifar10' : cifar10_model.__model__,
}
