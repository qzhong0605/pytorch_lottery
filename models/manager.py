""" It's a manager, which is used to manage all the running session
"""

from collections import OrderedDict

class DebugSessions(object):
    r"""
    It's a global object to list all the traced sessions
    """
    __sessions__ = OrderedDict()
    __current_session__ = 0   # the current session
    session_idx = 0

    @classmethod
    def register_session(cls, session):
        r"""
        add a new session with session_id
        """
        cls.__sessions__.update({session.get_session_id(): session})

    @classmethod
    def disable_session(cls, session_id):
        r"""
        remove a session from current session list
        """
        cls.__sessions__.pop(session_id)

    @classmethod
    def retrieve_session(cls, idx):
        if idx not in __sessions__:
            return None
        return cls.__sessions__[idx]

    @classmethod
    def number_session(cls):
        return len(cls.__sessions__)

    @classmethod
    def new_session_id(cls):
        return_id = cls.session_idx
        cls.session_idx += 1
        return return_id

    @classmethod
    def list_sessions(cls):
        return cls.__sessions__

    @classmethod
    def current_session(cls):
        return cls.__current_session__
