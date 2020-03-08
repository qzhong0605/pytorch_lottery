import torch

from models import checker

class Session(object):
    r"""
    A session represent a network to be debugged
    """
    def __init__(self):
        self._running_modules = []

    def add_module(self, module):
        r"""
        add a new module to be traced

        Args:
            module: an instance of atomic module, such as Conv2d, ReLU etc
        """
        if not checker.is_atomic_module(module):
            print(f'module {type(module).__name__} is not an atomic module')
            return
        self._running_modules.append(module)

    def index_module(self, idx):
        r"""
        Retrieve a module and return it

        Return:
            a module or None
        """
        if idx >= len(self._running_modules):
            print(f'idx {idx} out of range running modules')
            return None
        return self._running_modules[idx]

    def number_of_running_modules(self):
        return len(self._running_modules)

    def clear_running_modules(self):
        r"""
        clean up all the existing running modules
        """
        self._running_modules.clear()
