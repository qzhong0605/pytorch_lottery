from models.mnist import lenet
from models.mnist import fc
from models.mnist import lenetbn
from models.mnist import lenetcaffe

# a list of models and factory
__model__ = {
    'lenet' : lenet.build_lenet,
    'lenetbn' : lenetbn.build_lenetbn,
    'lenetcaffe' : lenetcaffe.build_lenetcaffe,
    'fc': fc.build_fc,
}
