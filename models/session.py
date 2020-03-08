import torch

from models import checker

class Session(object):
    r"""
    A session represent a network to be debugged
    """
    def __init__(self, session_id, name):
        self._running_modules = []
        self._session_id = session_id
        self._name = name

    def get_session_id(self):
        return self._session_id

    def get_session_name(self):
        return self._name

    def add_module(self, module, input, output):
        r"""
        add a new module to be traced, including input and output

        Args:
            module: an instance of atomic module, such as Conv2d, ReLU etc
            input: a tuple of tensor
            output: a tuple of tensor
        """
        if not checker.is_atomic_module(module):
            print(f'module {type(module).__name__} is not an atomic module')
            return
        self._running_modules.append((module, input, output))

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

    def last_module(self):
        r"""
        Return the lastest running module
        """
        return self.index_module(len(self._running_modules) - 1)

    def number_of_running_modules(self):
        return len(self._running_modules)

    def clear_running_modules(self):
        r"""
        clean up all the existing running modules
        """
        self._running_modules.clear()
