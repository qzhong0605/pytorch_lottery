import torch
from models import manager

################################################################################
#
# debug-related functions
#
################################################################################
def show_details_of_module(module):
    """ It's used to display the details of the module, including the weights and
    shapes,
    Here, the module includes
    """
    module_name = type(module).__name__
    print(f'====================== weight of {type(module).__name__} module ==========================')
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


def list_sessions():
    for session_id, session in enumerate(manager.DebugSessions.list_sessions()):
        print(f"""{session_id}: {session}""")


def backtrace_modules(session_id):
    session = manager.DebugSessions.retrieve_session(session_id)
    print(f"""=================== Running {session.number_of_running_modules()} module =======================""")
    for idx in range(session.number_of_running_modules()):
        print(f"""module {idx}: {session.index_module(idx)}""")


################################################################################
#
# model-related functions
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
