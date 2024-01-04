import logging
from typing import Any, Collection

from openinference.instrumentation.openai._request import (
    _AsyncRequest,
    _Request,
)
from openinference.instrumentation.openai.package import _instruments
from openinference.instrumentation.openai.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MODULE = "openai"


class OpenAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for openai.OpenAI.request and openai.AsyncOpenAI.request
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if (include_extra_attributes := kwargs.get("include_extra_attributes")) is None:
            include_extra_attributes = True
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        wrap_function_wrapper(
            module=_MODULE,
            name="OpenAI.request",
            wrapper=_Request(
                tracer=tracer,
                include_extra_attributes=include_extra_attributes,
            ),
        )
        wrap_function_wrapper(
            module=_MODULE,
            name="AsyncOpenAI.request",
            wrapper=_AsyncRequest(
                tracer=tracer,
                include_extra_attributes=include_extra_attributes,
            ),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        pass
