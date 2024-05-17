import importlib
import logging
from typing import Any, Collection

from openinference.instrumentation.llama_index.package import _instruments
from openinference.instrumentation.llama_index.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_ELIGIBLE_VERSION_FOR_NEW_INSTRUMENTATION = (0, 10, 37)


class LlamaIndexInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for LlamaIndex
    """

    __slots__ = (
        "_original_global_handler",
        "_use_experimental_instrumentation",  # feature flag
        "_event_handler",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        self._use_experimental_instrumentation = kwargs.get("use_experimental_instrumentation")
        if self._use_experimental_instrumentation:
            if not _legacy_llama_index():
                print(
                    "`use_experimental_instrumentation` feature flag is set. Spans "
                    "will be generated using the new instrumentation system "
                    "For more information about the new instrumentation system, visit "
                    "https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation/"  # noqa E501
                )
            else:
                print(
                    f"`use_experimental_instrumentation` feature flag is set. But "
                    f"the version of `llama-index-core` is not "
                    f">={'.'.join(map(str, _ELIGIBLE_VERSION_FOR_NEW_INSTRUMENTATION))}, "
                    f"so the flag is ignored."
                )
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        from openinference.instrumentation.llama_index._callback import (
            OpenInferenceTraceCallbackHandler,
        )

        if (
            _legacy_llama_index()
            or not self._use_experimental_instrumentation
            or not _legacy_llama_index()
            and self._use_experimental_instrumentation == "both"
        ):
            import llama_index.core

            self._original_global_handler = llama_index.core.global_handler
            llama_index.core.global_handler = OpenInferenceTraceCallbackHandler(tracer=tracer)

        self._event_handler = None
        if not _legacy_llama_index() and self._use_experimental_instrumentation:
            from llama_index.core.instrumentation import get_dispatcher

            from ._handler import EventHandler

            self._event_handler = EventHandler(tracer=tracer)
            dispatcher = get_dispatcher()
            dispatcher.add_event_handler(self._event_handler)
            dispatcher.add_span_handler(self._event_handler.span_handler)

    def _uninstrument(self, **kwargs: Any) -> None:
        if (
            _legacy_llama_index()
            or not self._use_experimental_instrumentation
            or not _legacy_llama_index()
            and self._use_experimental_instrumentation == "both"
        ):
            import llama_index.core

            llama_index.core.global_handler = self._original_global_handler
            self._original_global_handler = None

        if not _legacy_llama_index() and self._use_experimental_instrumentation:
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


def _legacy_llama_index() -> bool:
    v = importlib.metadata.version("llama-index-core")
    return tuple(map(int, v.split(".")[:3])) < _ELIGIBLE_VERSION_FOR_NEW_INSTRUMENTATION
