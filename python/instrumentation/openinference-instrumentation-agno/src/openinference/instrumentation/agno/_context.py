from contextlib import contextmanager
from contextvars import Token
from typing import Iterator, Optional, Protocol, Sequence

from openinference.instrumentation._spans import OpenInferenceSpan
from openinference.instrumentation._types import OpenInferenceSpanKind
from opentelemetry import context as _global_context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import Context, set_value
from opentelemetry.trace import Link, Span, SpanKind, Status, StatusCode
from opentelemetry.util.types import Attributes

from openinference.instrumentation import OITracer


class RuntimeContext(Protocol):
    def attach(self, context: Context) -> Token[Context]: ...
    def get_current(self) -> Context: ...
    def detach(self, token: Token[Context]) -> None: ...


class SpanActivation:
    def __init__(self, runtime_context: Optional[RuntimeContext] = None) -> None:
        self._ctx = runtime_context if runtime_context is not None else _global_context_api

    def attach(self, context: Context) -> Token[Context]:
        return self._ctx.attach(context)

    def get_current(self) -> Context:
        return self._ctx.get_current()

    def detach(self, token: Token[Context]) -> None:
        self._ctx.detach(token)

    def get_value(self, key: str) -> object:
        return self.get_current().get(key)

    def attach_value(self, key: str, value: object) -> Token[Context]:
        return self.attach(set_value(key, value, self.get_current()))

    def attach_span(self, span: Span) -> Token[Context]:
        return self.attach(trace_api.set_span_in_context(span, self.get_current()))

    def get_current_span(self) -> Span:
        return trace_api.get_current_span(self.get_current())

    @contextmanager
    def use_span(self, span: Span, end_on_exit: bool = False) -> Iterator[Span]:
        # Mirrors trace_api.use_span, which cannot be used here because it always
        # activates in the process-global context.
        token = self.attach_span(span)
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
            span.record_exception(exc)
            raise
        finally:
            self.detach(token)
            if end_on_exit:
                span.end()


class ActivationTracer(OITracer):
    def start_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Optional[Sequence[Link]] = (),
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        *,
        openinference_span_kind: Optional[OpenInferenceSpanKind] = None,
    ) -> OpenInferenceSpan:
        return super().start_span(
            name,
            get_activation().get_current() if context is None else context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception,
            openinference_span_kind=openinference_span_kind,
        )


_ACTIVATION = SpanActivation()


def get_activation() -> SpanActivation:
    return _ACTIVATION


def set_activation(activation: SpanActivation) -> None:
    global _ACTIVATION
    _ACTIVATION = activation
