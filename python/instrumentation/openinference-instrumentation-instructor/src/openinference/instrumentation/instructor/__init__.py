import logging
from importlib import import_module
from typing import Any, Collection

from openinference.instrumentation.instructor._wrappers import (
    _HandleResponseWrapper,
)
from openinference.instrumentation.instructor.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from functools import wraps
from typing import Union, Callable, TypeVar, ParamSpec

from openai import OpenAI, AsyncOpenAI

# (TODO)fix this
_instruments = ("instructor >= 0.0.1",)

logger = logging.getLogger(__name__)


class InstructorInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_handle_response_model",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        self._original_handle_response_model = getattr(import_module("instructor.patch"), "handle_response_model", None)
        process_resp_wrapper = _HandleResponseWrapper(tracer=tracer)
        wrap_function_wrapper(
            "instructor.patch",
            "handle_response_model",
            process_resp_wrapper
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        pass
