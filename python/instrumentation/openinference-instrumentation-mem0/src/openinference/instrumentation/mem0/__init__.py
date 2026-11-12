import logging
from importlib import import_module
from inspect import iscoroutinefunction
from typing import Any, Collection, Dict, List, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.mem0.version import __version__
from openinference.semconv.trace import OpenInferenceSpanKindValues

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("mem0ai >= 2.0.12",)

# The classes are patched through mem0's public re-exports rather than their defining
# modules (`mem0.memory.main`, `mem0.client.main`); wrapt sets the attribute on the same
# class object either way.
_MODULE = "mem0"

# Open-source memory (`Memory`, `AsyncMemory`) and the hosted platform client
# (`MemoryClient`, `AsyncMemoryClient`) expose the same memory operations.
_CLASSES = ("Memory", "AsyncMemory", "MemoryClient", "AsyncMemoryClient")

# `search` and `get_all` return a ranked or filtered list of memories, so they are
# RETRIEVER spans whose memories are recorded as retrieval documents. The remaining
# operations read or mutate memory state and are CHAIN spans.
_METHODS = {
    "add": OpenInferenceSpanKindValues.CHAIN,
    "search": OpenInferenceSpanKindValues.RETRIEVER,
    "get": OpenInferenceSpanKindValues.CHAIN,
    "get_all": OpenInferenceSpanKindValues.RETRIEVER,
    "update": OpenInferenceSpanKindValues.CHAIN,
    "delete": OpenInferenceSpanKindValues.CHAIN,
    "delete_all": OpenInferenceSpanKindValues.CHAIN,
    "history": OpenInferenceSpanKindValues.CHAIN,
}

__all__ = ("Mem0Instrumentor",)


class Mem0Instrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Traces mem0 memory operations as OpenInference spans.

    Wraps ``add``, ``search``, ``get``, ``get_all``, ``update``, ``delete``,
    ``delete_all`` and ``history`` on ``Memory``, ``AsyncMemory``, ``MemoryClient`` and
    ``AsyncMemoryClient``. ``instrument()`` accepts a ``tracer_provider`` and a
    ``TraceConfig``; ``uninstrument()`` restores the original methods.
    """

    __slots__ = ("_tracer", "_wrapped")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from openinference.instrumentation.mem0._wrappers import (
            _AsyncMemoryOperationWrapper,
            _MemoryOperationWrapper,
        )

        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        module = import_module(_MODULE)
        self._wrapped: List[Tuple[Any, str]] = []
        for class_name in _CLASSES:
            if (cls := getattr(module, class_name, None)) is None:
                continue
            for method_name, span_kind in _METHODS.items():
                if (method := getattr(cls, method_name, None)) is None:
                    continue
                wrapper_kwargs: Dict[str, Any] = dict(
                    tracer=self._tracer,
                    span_name=f"{class_name}.{method_name}",
                    method_name=method_name,
                    span_kind=span_kind,
                )
                if iscoroutinefunction(method):
                    wrap_function_wrapper(
                        cls, method_name, _AsyncMemoryOperationWrapper(**wrapper_kwargs)
                    )
                else:
                    wrap_function_wrapper(
                        cls, method_name, _MemoryOperationWrapper(**wrapper_kwargs)
                    )
                self._wrapped.append((cls, method_name))

    def _uninstrument(self, **kwargs: Any) -> None:
        for cls, method_name in getattr(self, "_wrapped", ()):
            unwrap(cls, method_name)
        self._wrapped = []
