from models.mnist import lenet
from models.mnist import fc

# a list of models and factory
__model__ = {
    'lenet' : lenet.build_lenet,
    'fc': fc.build_fc,
}
