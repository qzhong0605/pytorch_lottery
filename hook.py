""" Hook functions, which are used to perform on the forward and backward on network
"""

import torch
import torch.nn as nn

def module_features(module, input, output):
    """ This hook is performed after the forward on the network
    It's used to dump the feature shape information

    Args:
        module: an type of torch modul. It should be an atomic module other than
                container module
        input/output: an torch tensor
    """
    assert isinstance(module, nn.Module), f'module must be type of nn.Module'

    print(f"======================= Module ================================")
    for __input__ in input:
        print(f'\tinput tensor: {__input__.shape}')

    for __output__ in output:
        print(f'\toutput tensor: {__output__.shape}')
    print(f"===============================================================\n")
