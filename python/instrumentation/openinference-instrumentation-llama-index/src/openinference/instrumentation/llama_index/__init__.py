import logging
from importlib import import_module
from typing import TYPE_CHECKING, Any, Collection

from openinference.instrumentation.llama_index._callback import OpenInferenceTraceCallbackHandler
from openinference.instrumentation.llama_index.package import _instruments
from openinference.instrumentation.llama_index.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

_MODULE = "llama_index"

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
        if TYPE_CHECKING:
            import llama_index
        else:
            llama_index = import_module(_MODULE)
        self._original_global_handler = llama_index.global_handler
        llama_index.global_handler = OpenInferenceTraceCallbackHandler(tracer=tracer)

    def _uninstrument(self, **kwargs: Any) -> None:
        if TYPE_CHECKING:
            import llama_index
        else:
            llama_index = import_module(_MODULE)
        llama_index.global_handler = self._original_global_handler
        self._original_global_handler = None
