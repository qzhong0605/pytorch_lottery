""" This is the base model for tracing weight information and performing hook
functions on model
"""

import torch
import os
import torch.nn as nn
from collections import OrderedDict
from typing import Union

from models import session, manager, checker
import hook
import numpy as np


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

    def apply_weight_mask(self):
        r"""
        It's used for model pruning. The pruing weights are expressed by mask.
        For a network forwarding, the weight must be multiplied with mask
        """
        def update_weights_before_forward(module, input):
            # ignore bias parameter
            if len(module._parameters) == 0:
                return
            weight = module._parameters['weight']
            weight_mask = self._weight_mask[id(weight)][0]
            weight.data = weight * weight_mask.to(self._device)

        self._register_preforward_hook(update_weights_before_forward)

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
        weights_mask = OrderedDict()
        for _, mask in self._weight_mask.items():
            weights_mask.update({mask[1] : mask[0]})

        state = {
            'mask' : weights_mask,
            'weight' : self.state_dict()
        }
        torch.save(state, filename)

    def restore_state(self, filename):
        r""" For pruning algorithm, it's useful to restore the weights of network
        to the last pruning operation. Therefore, it restores the weights from a
        disk file

        :param filename: string, representing a disk file containing the state of network
        """
        assert os.path.exists(filename), f'model file {filename} must exist'
        state = torch.load(filename)
        self.load_state_dict(state['weight'])

        for name, mask in state['mask'].items():
            weight_id = self._weight_nameids[name]
            self._weight_mask.update({weight_id : (mask, name)})

    def pruning_with_percentile(self, q: float):
        r""" pruning weights with specified percent. If the values are less than
        percentile value, the mask would be set to 0; or else to 1

        :param q: float, a percent float. It must be between 0 and 1 inclusive
        """
        def percentile(t: torch.Tensor, q: float) -> Union[int, float]:
            r""" Return the ``q``-th percentile of the flattened input tensor's data
            It's based on https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
            """
            k = 1 + round(float(q) * (t.numel() - 1))
            result = t.view(-1).abs().kthvalue(k).values.item()
            return result

        if len(self._weight_mask) == 0:
            return
        for name, param in self.named_parameters():
            cpu_param = param.cpu()
            old_mask = self._weight_mask[id(param)][0]
            real_param = cpu_param * old_mask
            percentile_value = percentile(real_param, q)
            new_mask = torch.where(real_param.abs() < percentile_value,
                                   torch.zeros_like(old_mask), old_mask)
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

    def register_cleanup_running_modules(self):
        r"""
        register an hook to cleanup all running modules. It's performed after
        forward of network
        """
        def cleanup_running_modules(module, input, output):
            self._session.clear_running_modules()
        self.register_forward_hook(cleanup_running_modules)

    def trace_module(self, module_type):
        """ set trace on the specified module

        For torch framworks, its network is constructed dynamicly. Therefore,
        the traced running modules are cleaned up after an forward pass

        Args:
            module_type: a string representing module type
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

            # register atomic module
            module.register_forward_hook(register_active_module)
            if type(module).__name__ == module_type:
                handler = module.register_forward_hook(hook.module_debug)
                self._forward_trace_ids.update({module : handler.id})
