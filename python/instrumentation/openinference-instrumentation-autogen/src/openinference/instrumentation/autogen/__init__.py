from __future__ import annotations

from collections.abc import Collection
from typing import Any

from openinference.instrumentation.ag2 import AG2Instrumentor

_instruments = ("autogen >= 0.5.0",)


class AutogenInstrumentor:
    """Compatibility facade for the legacy ``autogen`` distribution."""

    def __init__(self) -> None:
        self._instrumentor = AG2Instrumentor()

    @property
    def is_instrumented_by_opentelemetry(self) -> bool:
        return bool(self._instrumentor.is_instrumented_by_opentelemetry)

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrument(self, **kwargs: Any) -> AutogenInstrumentor:
        # This facade validates the legacy distribution; the delegate checks for renamed AG2.
        kwargs["skip_dep_check"] = True
        self._instrumentor.instrument(**kwargs)
        return self

    def uninstrument(self, **kwargs: Any) -> AutogenInstrumentor:
        self._instrumentor.uninstrument(**kwargs)
        return self


__all__ = ["AutogenInstrumentor"]
