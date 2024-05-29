import logging
from typing import Any, Collection

from openinference.instrumentation.llama_index.package import _instruments
from openinference.instrumentation.llama_index.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LlamaIndexInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for LlamaIndex
    """

    __slots__ = (
        "_event_handler",
        "_use_legacy_callback_handler",  # deprecated
        "_original_global_handler",  # deprecated
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        self._use_legacy_callback_handler = kwargs.get("use_legacy_callback_handler")
        if self._use_legacy_callback_handler:
            import llama_index.core

            if hasattr(llama_index.core, "global_handler"):
                print("Using legacy callback handler.")
            else:
                print("Legacy callback handler is not available.")
                self._use_legacy_callback_handler = False
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        self._event_handler = None

        if self._use_legacy_callback_handler:
            from openinference.instrumentation.llama_index._callback import (
                OpenInferenceTraceCallbackHandler,
            )

            import llama_index.core

            self._original_global_handler = llama_index.core.global_handler
            llama_index.core.global_handler = OpenInferenceTraceCallbackHandler(tracer=tracer)
        else:
            from llama_index.core.instrumentation import get_dispatcher

            from ._handler import EventHandler

            self._event_handler = EventHandler(tracer=tracer)
            dispatcher = get_dispatcher()
            dispatcher.add_event_handler(self._event_handler)
            dispatcher.add_span_handler(self._event_handler.span_handler)

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._use_legacy_callback_handler:
            import llama_index.core

            llama_index.core.global_handler = self._original_global_handler
            self._original_global_handler = None
        else:
            if self._event_handler is None:
                return
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
