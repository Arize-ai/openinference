import contextlib
from typing import Iterator


@contextlib.contextmanager
def suppress_tracing() -> Iterator[None]:
    """
    Context manager to pause OpenTelemetry instrumentation.

    Examples:
        with suppress_tracing():
            # No tracing will occur within this block
            ...
    """
    try:
        from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, attach, detach, set_value
    except ImportError:
        yield
        return
    token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    yield
    detach(token)
