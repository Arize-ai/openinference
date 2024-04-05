import contextlib
from typing import Iterator, Union
from warnings import warn

from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    attach,
    detach,
    get_value,
    set_value,
)


@contextlib.contextmanager
def suppress_tracing() -> Iterator[None]:
    """
    Context manager to pause OpenTelemetry instrumentation.

    Examples:
        with suppress_tracing():
            # No tracing will occur within this block
            ...
    """
    token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    yield
    detach(token)


def start_session(session_id: Union[str, int]) -> object:
    """
    Starts a new session. The session created is not bound to the scope in which
    `start_session` is invoked; to end a session, you must explicitly call
    `end_session`. If a session is already active, this function has no effect.

    Returns a token that can be used to end the session.
    """
    if get_value("session.id") is not None:
        warn(
            "You are attempting to start a new OpenInference session, but one already exists. "
            "The existing session will be preserved and no new session will be created. "
            "In order to start a new session, you must first end the existing one.",
            UserWarning,
        )
        return
    return attach(set_value("session.id", session_id))


def end_session(token: object) -> None:
    """
    Ends a session. The input token must match the token returned by the
    corresponding call to `start_session`. If no session is active or if the
    token does not match, `end_sessions` has no effect.
    """
    detach(token)
