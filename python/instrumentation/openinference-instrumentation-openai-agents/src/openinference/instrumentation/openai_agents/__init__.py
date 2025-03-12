import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.openai._request import (
    _AsyncRequest,
    _Request,
)
from openinference.instrumentation.openai.package import _instruments
from openinference.instrumentation.openai.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MODULE = "openai"


class OpenAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for openai
    """

    __slots__ = (
        "_original_request",
        "_original_async_request",
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
        tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )
        openai = import_module(_MODULE)
        self._original_request = openai.OpenAI.request
        self._original_async_request = openai.AsyncOpenAI.request
        wrap_function_wrapper(
            module=_MODULE,
            name="OpenAI.request",
            wrapper=_Request(tracer=tracer, openai=openai),
        )
        wrap_function_wrapper(
            module=_MODULE,
            name="AsyncOpenAI.request",
            wrapper=_AsyncRequest(tracer=tracer, openai=openai),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        openai = import_module(_MODULE)
        openai.OpenAI.request = self._original_request
        openai.AsyncOpenAI.request = self._original_async_request
