""" This is the base model for tracing weight information and performing hook
functions on model
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import hook

__atomic_model__ = [
    nn.Conv1d, nn.Conv2d, nn.Conv3d,   # convolution module
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, # transpose convolution module
    nn.ReLU, nn.RReLU, nn.ReLU6, nn.LeakyReLU, nn.Sigmoid, # activation module
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, # average pool module
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, # max pool module
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, # adaptive average pool module
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d, # adaptive max pool module
    nn.Linear, # fully-connected module
    nn.LogSoftmax, # classifier module
    nn.Flatten,  # flatten module
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, # batch normalization module
]

def is_atomic_module(module):
    for __module__ in __atomic_model__:
        if isinstance(module, __module__):
            return True
    return False


class HookModule(nn.Module):
    def __init__(self, device):
        super(HookModule, self).__init__()
        self._device = device

    def _register_forward_hook(self, global_forward_fn=None):
        """ register an forward hook, which would be performed on all the
        children modules of the current module after forward

        It should not be called directly

        Return:
            a directory: map the module to all the handler id
        """
        _forward_hooks = OrderedDict()
        for module in self.modules():
            if not is_atomic_module(module):
                continue
            # current module is an atomic module
            handler = module.register_forward_hook(global_forward_fn)
            _forward_hooks.update({module : handler.id})
        return _forward_hooks


    def _unregister_forward_hook(self, forward_hook):
        """ unregister forward hooks for current module

        Args:
            forward_hook: a map from module to hook hander id
        """
        for module, handler_id in forward_hook.items():
            module._forward_hooks.pop(handler_id)


    def _register_backward_book(self, global_backward_fn=None):
        """ register an backward hook, which would be performed on all the
        children modules of the current module after backward

        It should not be called directly
        """
        _backward_hooks = OrderedDict()
        for module in self.modules():
            if not is_atomic_module(module):
                continue
            handler = module.register_backward_hook(global_backward_fn)
            _backward_hooks.update({module : handler.id})
        return _backward_hooks


    def _unregister_backward_book(self, backward_hook):
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
