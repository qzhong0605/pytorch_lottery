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
    get the specified weight with `key` name for a submodule
    """
    return module._parameters[key]


def min_weight_of_module(module, key):
    r"""
    retrieve the minimum value of the weight for a submodule
    """
    return module._parameters[key].min()


def max_weight_of_module(module, key):
    r"""
    retrieve the maximum value of the weight for a submodule
    """
    return module._parameters[key].max()


def mean_weight_of_module(module, key):
    r"""
    retrieve the mean value of the weight for a submodule
    """
    return module._parameters[key].mean()


def median_weight_of_module(module, key):
    r"""
    retrieve the median value of the weight for a submodule
    """
    return module._parameters[key].median()


def list_sessions():
    for session_id, _ in enumerate(manager.DebugSessions.list_sessions()):
        session = manager.DebugSessions.retrieve_session(session_id)
        last_module = session.last_module()
        if last_module == None:
            print(f"""{session_id}: {session.get_session_name()}""")
        else:
            last_idx = session.number_of_running_modules()
            print(f"""{session_id}: {session.get_session_name()} [module {last_idx-1}: {last_module[0]}]""")


def backtrace_modules(session_id):
    session = manager.DebugSessions.retrieve_session(session_id)
    print(f"""===================== Running {session.number_of_running_modules()} module =========================""")
    for idx in range(session.number_of_running_modules()):
        running_module = session.index_module(idx)
        print(f"""module {idx}: {running_module[0]}""")
        if type(running_module[1][0]) == tuple:
            input_tensor = '\tInput:'
            for __input__ in running_module[1][0]:
                input_tensor += f'\t{__input__.shape}, {__input__.device.type}'
            print(f'{input_tensor}')
        else:
            print(f"""\tInput: {running_module[1][0].shape}, {running_module[1][0].device.type}""")

        print(f"""\tOutput:\t{running_module[2][0].shape}, {running_module[2][0].device.type}""")


def retrieve_module(session_id, module_id):
    r"""
    Retrieve an module with session id and module id. If not exist, return None
    """
    session = manager.DebugSessions.retrieve_session(session_id)
    if session is None:
        print(f"""session ${session_id} doesn't exist""")
        return None
    running_module = session.index_module(module_id)
    return running_module


def breakpoint_on_module(session_id, module_type):
    r"""
    set breakpoint on all the submodule with type of `module_type` for current network
    """
    session = manager.DebugSessions.retrieve_session(session_id)
    if session is None:
        print(f"""session ${session_id} doesn't exist""")
        return
    hook_module = session.get_hook_module()
    hook_module.trace_module(module_type)


def clear_breakpoint(session_id):
    r"""
    Cleanup the existing breakpoints on submodule of network
    """
    session = manager.DebugSessions.retrieve_session(session_id)
    if session is None:
        print(f"""session ${session_id} doesn't exist""")
        return
    hook_module = session_id.get_hook_module()
    hook_module.clear_trace()


################################################################################
#
# model-related functions
#
################################################################################
def show_sparsity_of_model(model):
    """ It's used to display the sparsity of the network, including the weight shape
    and number of non-zero weights
    """
    eps = 1e-5
    print(f"\n\n============================ sparsity =========================================")
    for name, param in model.named_parameters():
        nonzero = param.abs() > eps
        num_nonzero = len(nonzero.nonzero())
        total = param.shape.numel()
        print(f'{name}:  {num_nonzero}  |  {total}  | sparsity: {1. * num_nonzero / total}')
    print(f"================================================================================")


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
