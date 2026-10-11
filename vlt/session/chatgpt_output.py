"""Internal approval scope; legacy providers have no approval closure."""
from contextlib import contextmanager
from contextvars import ContextVar

_current = ContextVar('chatgpt_output_approval', default=None)


def current():
    return _current.get()


def allowed(approval, direction=None):
    if approval is None:
        return True
    try:
        return (approval() if direction is None else approval(direction)) is True
    except Exception:
        return False


@contextmanager
def scope(approval):
    token = _current.set(approval)
    try:
        yield
    finally:
        _current.reset(token)
