""" Hook functions, which are used to perform on the forward and backward on network
"""

import torch
import torch.nn as nn

_USAGE = r"""
useful commands:
    show_details_of_module: display the weights information for current module
    get_weight_of_module: return weight tensor with name for a submodule
    min_weight_of_module: return min value of weight for a submodule
    max_weight_of_module: return max value of weight for a submodule
    mean_weight_of_module: return mean value of weight for a submodule
    median_weight_of_module: return median value of weight for a submodule
    list_sessions: display all the running network
    backtrace_modules: display the modules has performed
    retrieve_module: retrieve a submodule with session id and submodule id
"""
def module_help():
    print(_USAGE)


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
    from utils import list_sessions
    from utils import backtrace_modules
    from utils import retrieve_module
    from utils import clear_breakpoint
    from utils import breakpoint_on_module
    from utils import count_zeros
    from utils import percentile

    import module_ipdb; module_ipdb.set_trace()
