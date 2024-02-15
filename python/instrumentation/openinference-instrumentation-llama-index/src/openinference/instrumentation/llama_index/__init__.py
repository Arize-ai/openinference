import logging
from typing import Any, Collection

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

    __slots__ = ("_original_global_handler",)

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        from openinference.instrumentation.llama_index._callback import (
            OpenInferenceTraceCallbackHandler,
        )

        import llama_index.core

        self._original_global_handler = llama_index.core.global_handler
        llama_index.core.global_handler = OpenInferenceTraceCallbackHandler(tracer=tracer)

    def _uninstrument(self, **kwargs: Any) -> None:
        import llama_index.core

        llama_index.core.global_handler = self._original_global_handler
        self._original_global_handler = None
