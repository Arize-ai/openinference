import logging
from importlib import import_module
from typing import Any, Collection

from openinference.instrumentation.instructor._wrappers import (
    _HandleResponseWrapper,
    _PatchWrapper,
)
from openinference.instrumentation.instructor.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

_instruments = ("instructor >= 0.0.1",)


class InstructorInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_handle_response_model",
        "_original_patch",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        self._original_patch = getattr(import_module("instructor"), "patch", None)
        patch_wrapper = _PatchWrapper(tracer=tracer)
        wrap_function_wrapper(
            "instructor",
            "patch",
            patch_wrapper
        )

        self._original_handle_response_model = getattr(import_module("instructor.patch"), "handle_response_model", None)
        process_resp_wrapper = _HandleResponseWrapper(tracer=tracer)
        wrap_function_wrapper(
            "instructor.patch",
            "handle_response_model",
            process_resp_wrapper
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._original_patch is not None:
            instructor_module = import_module("instructor")
            instructor_module.patch = self._original_patch
            self._original_patch = None

        if self._original_handle_response_model is not None:
            patch_module = import_module("instructor.patch")
            patch_module.handle_response_model = self._original_handle_response_model
            self._original_handle_response_model = None
