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


def combined(approvals):
    """A delayed mixed-direction frame retains each row's own authorization."""
    tagged = tuple((approval, who) for approval, who in approvals if approval is not None)
    if not tagged:
        return None
    return lambda: all(allowed(approval, who) for approval, who in tagged)


@contextmanager
def scope(approval):
    token = _current.set(approval)
    try:
        yield
    finally:
        _current.reset(token)
