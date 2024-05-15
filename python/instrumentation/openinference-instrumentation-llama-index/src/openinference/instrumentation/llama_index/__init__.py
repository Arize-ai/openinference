import importlib
import logging
from typing import Any, Collection, Optional

from openinference.instrumentation.llama_index.package import _instruments
from openinference.instrumentation.llama_index.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

_MODULE = "llama_index.core"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LlamaIndexInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for LlamaIndex
    """

    __slots__ = (
        "_event_handler",
        "_original_global_handler",  # deprecated; to be removed
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        from openinference.instrumentation.llama_index._callback import (
            OpenInferenceTraceCallbackHandler,
        )

        if _legacy_llama_index():
            import llama_index.core

            self._original_global_handler = llama_index.core.global_handler
            llama_index.core.global_handler = OpenInferenceTraceCallbackHandler(tracer=tracer)
        else:
            from llama_index.core.instrumentation import get_dispatcher

            from ._handler import EventHandler

            self._event_handler: Optional[EventHandler] = EventHandler(tracer=tracer)
            dispatcher = get_dispatcher()
            dispatcher.add_event_handler(self._event_handler)
            dispatcher.add_span_handler(self._event_handler.span_handler)

    def _uninstrument(self, **kwargs: Any) -> None:
        if _legacy_llama_index():
            import llama_index.core

            llama_index.core.global_handler = self._original_global_handler
            self._original_global_handler = None
        else:
            from llama_index.core.instrumentation import get_dispatcher

            dispatcher = get_dispatcher()
            dispatcher.span_handlers[:] = filter(
                lambda h: h is not self._event_handler.span_handler,  # type: ignore
                dispatcher.span_handlers,
            )
            dispatcher.event_handlers[:] = filter(
                lambda h: h is not self._event_handler,
                dispatcher.event_handlers,
            )
            self._event_handler = None


def _legacy_llama_index() -> bool:
    v = importlib.metadata.version("llama-index-core")
    return tuple(map(int, v.split(".")[:3])) < (0, 10, 37)
