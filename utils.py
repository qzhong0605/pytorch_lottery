import torch

def show_details_of_module(module):
    """ It's used to display the details of the module, including the weights and
    shapes,
    Here, the module includes
    """
    print('=========================== weight details of module ===============================')
    for name, param in module.named_parameters():
        print(f'{name}: \t {param.shape}')
    print('=====================================================================================')


def get_weight_of_module(module, key):
    r"""
    get the specified weight with `key` name
    """
    return module._parameters[key]


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

