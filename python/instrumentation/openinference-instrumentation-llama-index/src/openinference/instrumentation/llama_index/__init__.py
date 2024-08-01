import logging
from typing import Any, Collection

from openinference.instrumentation import OITracer, TraceConfig
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
        "_config",
        "_span_handler",
        "_event_handler",
        "_use_legacy_callback_handler",  # deprecated
        "_original_global_handler",  # deprecated
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._use_legacy_callback_handler = bool(kwargs.get("use_legacy_callback_handler"))
        if self._use_legacy_callback_handler:
            import llama_index.core

            if hasattr(llama_index.core, "global_handler"):
                print(
                    "Using legacy callback handler. "
                    "TraceConfig not supported for callback handlers."
                )
            else:
                self._use_legacy_callback_handler = False
                print(
                    "Legacy callback handler is not available. "
                    "Using new instrumentation event/span handler instead."
                )
        tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )
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
            self._span_handler = self._event_handler.span_handler
            dispatcher = get_dispatcher()
            for span_handler in dispatcher.span_handlers:
                if isinstance(span_handler, type(self._span_handler)):
                    break
            else:
                dispatcher.add_span_handler(self._span_handler)
            for event_handler in dispatcher.event_handlers:
                if isinstance(event_handler, type(self._event_handler)):
                    break
            else:
                dispatcher.add_event_handler(self._event_handler)

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
                lambda h: not isinstance(h, type(self._span_handler)),
                dispatcher.span_handlers,
            )
            dispatcher.event_handlers[:] = filter(
                lambda h: not isinstance(h, type(self._event_handler)),
                dispatcher.event_handlers,
            )
            self._event_handler = None
