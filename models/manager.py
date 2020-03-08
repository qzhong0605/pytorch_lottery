""" It's a manager, which is used to manage all the running session
"""

class DebugSessions(object):
    r"""
    It's a global object to list all the traced sessions
    """
    __sessions__ = []
    __current_session__ = 0   # the current session
    session_idx = 0

    @classmethod
    def register_session(cls, session):
        cls.__sessions__.append(session)

    @classmethod
    def retrieve_session(cls, idx):
        return cls.__sessions__[idx]

    @classmethod
    def number_session(cls):
        return len(cls.__sessions__)

    @classmethod
    def new_session_id(cls):
        return_id = session_idx
        session_idx += 1
        return return_id
