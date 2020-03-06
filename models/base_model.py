""" This is the base model for tracing weight information and performing hook
functions on model
"""

import torch
import torch.nn as nn
from collections import OrderedDict

class HookModule(nn.Module):
    def __init__(self):
        super(HookModule, self).__init__()

    def _register_forward_hook(self, global_forward_fn=None):
        """ register an forward hook, which would be performed on all the
        children modules of the current module after forward

        It should not be called directly

        Return:
            a directory: map the module to all the handler id
        """
        _forward_hooks = OrderedDict()
        for module in self.modules():
            if isinstance(module, nn.Sequential):
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
        for module in self.modules():
            if isinstance(module, nn.Sequential):
                continue
            handler_id = forward_hook[module]
            module._forward_hooks.pop(handler_id)


    def _register_backward_book(self, global_backward_fn=None):
        """ register an backward hook, which would be performed on all the
        children modules of the current module after backward

        It should not be called directly
        """
        _backward_hooks = OrderedDict()
        for module in self.modules():
            if isinstance(module, nn.Sequential):
                continue
            handler = module.register_backward_hook(global_backward_fn)
            _backward_hooks.update({module : handler.id})
        return _backward_hooks


    def _unregister_backward_book(self, backward_hook):
        """ unregister backward hooks for all the children modules of current module

        Args:
            backward_hook: an ordered dict, mapping from module to handler id
        """
        for module in self.modules():
            if isinstance(module, nn.Sequential):
                continue
            handler_id = backward_hook[module]
            module._backward_hooks.pop(handler_id)


    def dump_tensor_shape(self, input):
        """ It output the approciate tensor information for current model

        Args:
            input: a tuple or list, indicating the input shape
        """
        input_tensor = torch.rand(input)
        self.__call__(input_tensor)
