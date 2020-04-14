import torch

from models import checker

class Session(object):
    r"""
    A session represent a network to be debugged
    """
    def __init__(self, session_id, name, hook_module):
        self._running_fp_modules = []
        self._running_bp_modules = []

        self._session_id = session_id
        self._name = name
        self._hook_module = hook_module    # the debugged network module

    def get_hook_module(self):
        return self._hook_module

    def get_session_id(self):
        return self._session_id

    def get_session_name(self):
        return self._name

    def add_fp_module(self, module, input, output):
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
        self._running_fp_modules.append((module, input, output))

    def add_bp_module(self, module, grad_input, grad_output):
        """add a bp module to be traced, including grad_input and grad_output"""
        if not checker.is_atomic_module(module):
            print(f'module {type(module).__name__} is not an atomic module')
            return
        self._running_bp_modules.append((module, grad_input, grad_output))

    def index_module(self, idx):
        r"""
        Retrieve a module and return it

        Return:
            a module or None
        """
        if idx < len(self._running_fp_modules):
            return self._running_fp_modules[idx]
        elif idx < self.number_of_running_modules():
            return self._running_bp_modules[idx - len(self._running_fp_modules)]
        else:
            print(f'idx {idx} out of range running modules')
            return None

    def last_module(self):
        r"""
        Return the lastest running module
        """
        if self.number_of_running_modules() == 0:
            return None
        return self.index_module(self.number_of_running_modules() - 1)

    def number_of_running_modules(self):
        return len(self._running_fp_modules) + len(self._running_bp_modules)

    def number_of_fp_modules(self):
        """return the number of forward running modules"""
        return len(self._running_fp_modules)

    def number_of_bp_modules(self):
        """return the number of backward running modules"""
        return len(self._running_bp_modules)

    def clear_running_modules(self):
        r"""
        clean up all the existing running modules, including bp running modules
        """
        self._running_fp_modules.clear()
        self._running_bp_modules.clear()
