import torch

################################################################################
#
# debug-related functions
#
################################################################################
class DebugSessions(object):
    r"""
    It's a global object to list all the traced sessions
    """
    __sessions__ = []
    __current_session__ = 0   # the current session

    @classmethod
    def register_session(cls, session):
        cls.__sessions__.append(session)

    @classmethod
    def retrieve_session(cls, idx):
        return cls.__sessions__[idx]

    @classmethod
    def number_session(cls):
        return len(cls.__sessions__)


def show_details_of_module(module):
    """ It's used to display the details of the module, including the weights and
    shapes,
    Here, the module includes
    """
    module_name = type(module).__name__
    print(r'====================== weight of {} module =========================='.format(module_name))
    for name, param in module.named_parameters():
        print(f'{name}: \t {param.shape}')
    print('===============================================================================')


def get_weight_of_module(module, key):
    r"""
    get the specified weight with `key` name
    """
    return module._parameters[key]


def min_weight_of_module(module, key):
    r"""
    retrieve the minimum value of the weight
    """
    return module._parameters[key].min()


def max_weight_of_module(module, key):
    r"""
    retrieve the maximum value of the weight
    """
    return module._parameters[key].max()


def mean_weight_of_module(module, key):
    r"""
    retrieve the mean value of the weight
    """
    return module._parameters[key].mean()


def median_weight_of_module(module, key):
    r"""
    retrieve the median value of the weight
    """
    return module._parameters[key].median()


################################################################################
#
# debug-related functions
#
################################################################################
def show_sparsity_of_model(model):
    """ It's used to display the sparsity of the network, including the weight shape
    and number of non-zero weights
    """
    pass


def setup_dataloader(batch_size):
    """ It's used to setup the data loader for training and evaluation
    """
    pass


def resume_from_checkpoint(model, checkpoint):
    """ restore the weights of network from the last saved state

    args:
        model: an model module
        checkpoint: a saved model checkpoint file
    """
    import os
    if not os.path.exists(checkpoint):
        return
