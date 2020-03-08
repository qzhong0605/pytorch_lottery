""" Hook functions, which are used to perform on the forward and backward on network
"""

import torch
import torch.nn as nn

_USAGE = r"""
useful commands:
    show_details_of_module: display the weights information for current module
    get_weight_of_module: return weight tensor with name for current module
    min_weight_of_module: return min value for weight on current module
    max_weight_of_module: return max value for weight on current module
    mean_weight_of_module: return mean value for weight on current module
    median_weight_of_module: return median value for weight on current module
"""

def module_help():
    print(f'{_USAGE}')

def module_features(module, input, output):
    """ This hook is performed after the forward on the network
    It's used to dump the feature shape information

    Args:
        module: an type of torch modul. It should be an atomic module other than
                container module
        input/output: an torch tensor
    """
    assert isinstance(module, nn.Module), f'module must be type of nn.Module'

    print(f"==================== Module({type(module).__name__}) ==========================")
    for __input__ in input:
        print(f'\tinput tensor: {__input__.shape}')

    for __output__ in output:
        print(f'\toutput tensor: {__output__.shape}')
    print(f"===============================================================\n")


def module_debug(module, input, output):
    r""" It's used to trace and debug all the atomic module of the network
    """
    from utils import show_details_of_module
    from utils import get_weight_of_module
    from utils import min_weight_of_module
    from utils import max_weight_of_module
    from utils import mean_weight_of_module
    from utils import median_weight_of_module

    import module_ipdb; module_ipdb.set_trace()
