"""Deprecated compatibility layer over the AG2 instrumentation."""

from __future__ import annotations

from typing import Any

from openinference.instrumentation.ag2 import AG2Instrumentor


class SpanAttributes:
    """Deprecated: legacy attribute keys kept for backward compatibility."""

    OPENINFERENCE_SPAN_KIND: str = "openinference.span.kind"
    INPUT_VALUE: str = "input.value"
    INPUT_MIME_TYPE: str = "input.mime_type"
    OUTPUT_VALUE: str = "output.value"
    OUTPUT_MIME_TYPE: str = "output.mime_type"
    TOOL_NAME: str = "tool.name"
    TOOL_ARGS: str = "tool.args"
    TOOL_KWARGS: str = "tool.kwargs"
    TOOL_PARAMETERS: str = "tool.parameters"
    TOOL_CALL_FUNCTION_ARGUMENTS: str = "tool_call.function.arguments"
    TOOL_CALL_FUNCTION_NAME: str = "tool_call.function.name"


class AutogenInstrumentor:
    """Compatibility facade for the legacy ``autogen`` distribution."""

    def __init__(self) -> None:
        self._instrumentor = AG2Instrumentor()
        self._instrument_requested = False

    @property
    def is_instrumented_by_opentelemetry(self) -> bool:
        return bool(self._instrumentor.is_instrumented_by_opentelemetry)

    def instrument(self, **kwargs: Any) -> AutogenInstrumentor:
        self._instrumentor.instrument(**kwargs)
        self._instrument_requested = True
        return self

    def uninstrument(self, **kwargs: Any) -> AutogenInstrumentor:
        # Uninstrument only after this facade was asked to instrument; the delegate
        # is a singleton, so an untouched facade must not tear down instrumentation
        # installed through AG2Instrumentor directly.
        if self._instrument_requested:
            self._instrumentor.uninstrument(**kwargs)
            self._instrument_requested = False
        return self


__all__ = ["AutogenInstrumentor", "SpanAttributes"]
