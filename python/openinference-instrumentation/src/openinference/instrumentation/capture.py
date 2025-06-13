from contextvars import ContextVar, Token
from types import TracebackType
from typing import Iterable, Optional, Type

from opentelemetry.trace import SpanContext

_current_capture_span_context = ContextVar["capture_span_context"]("current_capture_span_context")


def _capture_span_context(span_context: SpanContext) -> None:
    capture = _current_capture_span_context.get(None)
    if capture is not None:
        capture._contexts.append(span_context)


class capture_span_context:
    """
    Context manager for capturing OpenInference span context.

    Examples:
        with capture_span_context() as capture:
            response = openai_client.chat.completions.create(...)
            span_context = capture.get_last_span_context()
            if span_context:
                phoenix_client.annotations.add_span_annotation(
                    span_id=span_context.span_id,
                    annotation_name="feedback",
                    ...
                )
    """

    _contexts: list[SpanContext]
    _token: Token["capture_span_context"]

    def __init__(self) -> None:
        self._contexts = []

    def __enter__(self) -> "capture_span_context":
        self._token = _current_capture_span_context.set(self)
        return self

    def __exit__(
        self,
        _exc_type: Optional[Type[BaseException]],
        _exc_value: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> None:
        _current_capture_span_context.reset(self._token)
        self._contexts.clear()

    def get_last_span_context(self) -> Optional[SpanContext]:
        """
        Returns the last captured span context, or None if no spans were captured.
        """
        if self._contexts:
            return self._contexts[-1]
        return None

    def get_span_contexts(self) -> Iterable[SpanContext]:
        """
        Returns a list of all captured span contexts.
        """
        return self._contexts.copy()
