"""Internal approval scope; legacy providers have no approval closure."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

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


@dataclass(frozen=True, eq=False, slots=True)
class _Combined:
    rows: tuple

    def __call__(self):
        return all(allowed(approval, who) for approval, who in self.rows)


def same(first, second):
    if first is second:
        return True
    if not isinstance(first, _Combined) or not isinstance(second, _Combined):
        return False
    return len(first.rows) == len(second.rows) and all(
        a is b and who == other for (a, who), (b, other) in zip(first.rows, second.rows))


def combined(approvals):
    """A delayed mixed-direction frame retains each row's own authorization."""
    tagged = tuple((approval, who) for approval, who in approvals if approval is not None)
    if not tagged:
        return None
    return _Combined(tagged)


@contextmanager
def scope(approval):
    token = _current.set(approval)
    try:
        yield
    finally:
        _current.reset(token)
