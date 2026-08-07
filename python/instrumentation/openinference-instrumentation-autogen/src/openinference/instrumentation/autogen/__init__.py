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
        self._owns_instrumentation = False

    @property
    def is_instrumented_by_opentelemetry(self) -> bool:
        return bool(self._instrumentor.is_instrumented_by_opentelemetry)

    def instrument(self, **kwargs: Any) -> AutogenInstrumentor:
        already_instrumented = self._instrumentor.is_instrumented_by_opentelemetry
        self._instrumentor.instrument(**kwargs)
        self._owns_instrumentation = (
            not already_instrumented and self._instrumentor.is_instrumented_by_opentelemetry
        )
        return self

    def uninstrument(self, **kwargs: Any) -> AutogenInstrumentor:
        # Only tear down instrumentation this facade installed; the delegate is
        # a singleton shared with any independently used AG2Instrumentor.
        if self._owns_instrumentation:
            self._instrumentor.uninstrument(**kwargs)
            self._owns_instrumentation = False
        return self


__all__ = ["AutogenInstrumentor", "SpanAttributes"]
