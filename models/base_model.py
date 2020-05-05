""" This is the base model for tracing weight information and performing hook
functions on model
"""

import torch
import os
import math
import torch.nn as nn
from collections import OrderedDict
from typing import Union

from models import session, manager, checker
import hook
import numpy as np

def tensor_nonzero(t:torch.Tensor):
    """return an flattened tensor, which filled with nonzero elements"""
    np_t = t.data.numpy()
    nonzero = np_t[np_t.nonzero()]
    return torch.from_numpy(nonzero)

def count_zeros(t:torch.Tensor):
    """return the number of zeros for the input tensor"""
    cpu_t = t.cpu()
    num_zeros = torch.where(cpu_t.abs() < 1e-6, torch.ones(cpu_t.shape), torch.zeros(cpu_t.shape))
    return num_zeros.sum().item()

def percentile(t: torch.Tensor, q: float) -> Union[int, float]:
    r""" Return the ``q``-th percentile of the flattened input tensor's data
    It's based on https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
    """
    k = 1 + round(float(q) * (t.numel() - 1))
    result = t.abs().kthvalue(k).values.item()
    return result


class HookModule(nn.Module):
    def __init__(self, device, name):
        super(HookModule, self).__init__()
        self._device = device
        self._forward_trace_ids = OrderedDict()   # track the forward  pass
        self._backward_trace_ids = OrderedDict()  # track the backward pass
        self._session = session.Session(manager.DebugSessions.new_session_id(),
                                        name, self)
        manager.DebugSessions.register_session(self._session)

        # track the mask of weights. The key is the weight tensor id and the value
        # is a tuple, including mask tensor and weight name
        self._weight_mask = OrderedDict()
        # a map from the weight name to weight id
        self._weight_nameids = OrderedDict()

        # pruing-related configure
        self._pruning_init = None   # how to initialize the weights
        self._pruning_op = None  # how to pruning network
        self._check_point = None   # where to hold the state of network

    def get_weight_mask(self):
        """return a dict mapping weight name to mask"""
        ret_weight_mask = OrderedDict()
        for _, mask_info in self._weight_mask.items():
            ret_weight_mask.update({mask_info[1] : mask_info[0]})
        return ret_weight_mask

    def init_pruning_configure(self, **kwargs):
        if 'init' in kwargs:
            self._pruning_init = kwargs['init']
        if 'init_kind' in kwargs:
            self._pruning_init_kind = kwargs['init_kind']
        if 'op' in kwargs:
            self._pruning_op = kwargs['op']
        if 'check_point' in kwargs:
            self._check_point = kwargs['check_point']
            if os.path.exists(self._check_point):
                print('remove the existing file: {}'.format(self._check_point))
                os.remove(self._check_point)

    def init_pruning_context(self, **kwargs):
        self.init_pruning_configure(**kwargs)
        self.init_weight_mask()
        # save the initialized state of network
        torch.save(self.state_dict(), self._check_point)
        # register an hook to update the weight with mask before network
        # forward
        def model_update_hook(module, input):
            self.update_weight_with_mask()
        self.register_forward_pre_hook(model_update_hook)

    def update_weight_with_mask(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param.data = param.data * self._weight_mask[id(param)][0].to(self._device)

    def reinitialize(self):
        """re-initialize the weights of networks after pruning"""
        if self._pruning_init == 'random':
            self._random_init()
        elif self._pruning_init == 'last':
            self._last_state_init()
        elif self._pruning_init == 'same':
            # keep the remaining weights to be same with the last
            pass
        else:
            raise NotImplementedError(f'{self._pruning_init} initialization not support yet')

    def _random_init(self, method='kaiming'):
        r""" after pruning network, there is a need to initialize the weight of
        the network. Here, the weights are initialized by the normal method for
        neural networks

        :param method: string, including kaiming and xavier
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if self._pruning_init_kind == 'kaiming':
                    nn.init.kaiming_uniform_(param.data, a=math.sqrt(5))
                elif self._pruning_init_kind == 'xavier':
                    nn.init.xavier_normal_(param.data)
                else:
                    raise NotImplementedError('not support {} initialization'.format(self._pruning_init_kind))
                # update the parameter data with mask
                weight_mask = self._weight_mask[id(param)][0]
                param.data = param.data * weight_mask.to(self._device)

    def _last_state_init(self):
        r""" after pruning network, filling the weights from the last saved
        checkpoint

        :param filename: string, a disk state file
        """
        self.restore_state(self._check_point)
        for name, param in self.named_parameters():
            if 'weight' in name:
                # update the parameter data with mask
                weight_mask = self._weight_mask[id(param)][0]
                param.data = param.data * weight_mask.to(self._device)

    def init_weight_mask(self):
        for name, param in self.named_parameters():
            mask = torch.ones(param.shape)
            self._weight_nameids.update({name : id(param)})
            self._weight_mask.update({id(param) : (mask, name)})

    def checkpoint(self, filename):
        r""" when performing traing operation, it's used to save the weights and
        mask for the current state of network. It's useful to restore the last
        state for resuming training

        :param filename: string, representing the checkpoint file name
        """
        torch.save(self.state_dict(), filename)

    def restore_state(self, filename):
        r""" For pruning algorithm, it's useful to restore the weights of network
        to the last pruning operation. Therefore, it restores the weights from a
        disk file

        :param filename: string, representing a disk file containing the state of network
        """
        assert os.path.exists(filename), f'model file {filename} must exist'
        state = torch.load(filename)
        self.load_state_dict(state)

    def pruning_network(self, q:float):
        r""" choose one pruning method to compact the network, including `layer`
        and global.
        """
        if self._pruning_op == 'layer':
            self.pruning_with_percentile(q)
        elif self._pruning_op == 'global':
            self.global_pruning(q)
        else:
            raise NotImplementedError("pruning {} method doesn't be supported")

        # now reinitialize the weights of network
        self.reinitialize()

    def pruning_with_percentile(self, q: float):
        r""" pruning weights with specified percent. If the values are less than
        percentile value, the mask would be set to 0; or else to 1

        :param q: float, a percent float. It must be between 0 and 1 inclusive
        """
        if len(self._weight_mask) == 0:
            return
        for name, param in self.named_parameters():
            if 'weight' not in name:
                # skip the other type weights
                continue
            cpu_param = param.cpu()
            old_mask = self._weight_mask[id(param)][0]
            percentile_value = percentile(tensor_nonzero(cpu_param), q)
            new_mask = torch.where(cpu_param.abs() < percentile_value,
                                   torch.zeros_like(old_mask), old_mask)
            param.data = param.data * new_mask.to(self._device)
            self._weight_mask.update({id(param) : (new_mask, name)})

    def global_pruning(self, q:float):
        r""" For deep network, there are including different number of weights for
        different module. To keep the less weights, there is a need to do pruning
        on a global way.
        """
        if len(self._weight_mask) == 0:
            return
        # caculate the kth value for all weights of network
        global_weights = torch.cat([param.abs().view(-1)
                                    for name, param in self.named_parameters()
                                    if 'weight' in name])
        percentile_value = percentile(tensor_nonzero(global_weights), q)

        for name, param in self.named_parameters():
            if 'weight' not in name:
                # skip the other type weights
                continue
            cpu_param = param.cpu()
            old_mask = self._weight_mask[id(param)][0]
            new_mask = torch.where(cpu_param.abs() < percentile_value,
                                   torch.zeros_like(old_mask),
                                   old_mask)
            param.data = param.data * new_mask.to(self._device)
            self._weight_mask.update({id(param) : (new_mask, name)})

    def _register_forward_hook(self, global_forward_fn=None):
        """ register an forward hook, which would be performed on all the
        children modules of the current module after forward

        It should not be called directly

        Return:
            a directory: map the module to all the handler id
        """
        _forward_hooks = OrderedDict()
        for module in self.modules():
            if not checker.is_atomic_module(module):
                continue
            # current module is an atomic module
            handler = module.register_forward_hook(global_forward_fn)
            _forward_hooks.update({module : handler.id})
        return _forward_hooks

    def _register_preforward_hook(self, global_forward_fn=None):
        """ register an global_backward_fn on all the basic module of the network

        Return:
            a directory: map the module to all the handler id
        """
        _pre_forward_hooks = OrderedDict()
        for module in self.modules():
            if not checker.is_atomic_module(module):
                continue
            # current module is an atomic module
            handler = module.register_forward_pre_hook(global_forward_fn)
            _pre_forward_hooks.update({module : handler.id})
        return _pre_forward_hooks

    def _unregister_forward_hook(self, forward_hook):
        """ unregister forward hooks for current module

        Args:
            forward_hook: a map from module to hook handler id
        """
        for module, handler_id in forward_hook.items():
            module._forward_hooks.pop(handler_id)

    def _unregister_preforward_hook(self, pre_forward_hooks):
        """ unregister preforward hooks for current module

        Args:
            pre_forward_hooks: a map from module to preforward hook handler id
        """
        for module, handler_id in pre_forward_hooks.items():
            module._pre_forward_hooks.pip(handler_id)

    def _register_backward_hook(self, global_backward_fn=None):
        """ register an backward hook, which would be performed on all the
        children modules of the current module after backward

        It should not be called directly
        """
        _backward_hooks = OrderedDict()
        for module in self.modules():
            if not checker.is_atomic_module(module):
                continue
            handler = module.register_backward_hook(global_backward_fn)
            _backward_hooks.update({module : handler.id})
        return _backward_hooks

    def _unregister_backward_hook(self, backward_hook):
        """ unregister backward hooks for all the children modules of current module

        Args:
            backward_hook: an ordered dict, mapping from module to handler id
        """
        for module, handler_id in backward_hook.items():
            module._backward_hooks.pop(handler_id)

    def dump_tensor_shape(self, input):
        """ It output the approciate tensor information for current model

        Args:
            input: a tuple or list, indicating the input shape
        """
        input_tensor = torch.rand(input)
        input_tensor = input_tensor.to(self._device)
        forward_hooks = self._register_forward_hook(hook.module_features)
        self.__call__(input_tensor)
        self._unregister_forward_hook(forward_hooks)

    def set_trace(self):
        """ set trace on all atomic module. The traces are set after performing
        forward pass on module
        """
        self.register_cleanup_running_modules()
        def register_active_module(module, input, output):
            r"""
            register an active module into current session
            """
            self._session.add_module(module, input, output)
        for module in self.modules():
            if not checker.is_atomic_module(module):
                continue
            # register active module
            module.register_forward_hook(register_active_module)

    def clear_trace(self):
        self._unregister_forward_hook(self._forward_trace_ids)
        self._unregister_backward_hook(self._backward_trace_ids)

    def register_cleanup_running_modules(self):
        r"""
        register an hook to cleanup all running modules. It's performed after
        forward of network
        """
        def cleanup_running_modules(module, input):
            self._session.clear_running_modules()
        self.register_forward_pre_hook(cleanup_running_modules)

    def trace_module(self, module_type, trace_bp=False):
        """ set trace on the specified module

        For torch framworks, its network is constructed dynamicly. Therefore,
        the traced running modules are cleaned up after a forward pass

        If we want to trace the bp procedure of network training, the traced
        running modules alse include a fake module, which represents an bp module
        At this time, the running modules would be cleaned up after a backward pass

        Args:
            module_type: a string representing module type
            trace_bp: a bool. The default beheavior is to ignore the backward pass
        """
        self.register_cleanup_running_modules()
        def register_active_fp_module(module, input, output):
            r"""
            register an active module into current session
            """
            self._session.add_fp_module(module, input, output)
        def register_active_bp_module(module, grad_input, grad_output):
            self._session.add_bp_module(module, grad_input, grad_output)

        for module in self.modules():
            if not checker.is_atomic_module(module):
                continue

            # register atomic module
            module.register_forward_hook(register_active_fp_module)
            if trace_bp:
                module.register_backward_hook(register_active_bp_module)
            if type(module).__name__ == module_type:
                handler = module.register_forward_hook(hook.module_debug)
                self._forward_trace_ids.update({module : handler.id})

                if trace_bp:
                    bp_handler = module.register_backward_hook(hook.module_debug)
                    self._backward_trace_ids.update({module : bp_handler.id})
