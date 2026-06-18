import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.instructor.version import __version__

_instruments = ("instructor >= 0.0.1",)

logger = logging.getLogger(__name__)


class InstructorInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_tracer",
        "_original_handle_response_model",
        "_original_patch",
        "_patch_module",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from openinference.instrumentation.instructor._wrappers import (
            _HandleResponseWrapper,
            _PatchWrapper,
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

        self._original_patch = getattr(import_module("instructor"), "patch", None)
        patch_wrapper = _PatchWrapper(tracer=self._tracer)  # type: ignore[arg-type]
        wrap_function_wrapper("instructor", "patch", patch_wrapper)

        # ``handle_response_model`` has moved across instructor releases:
        #   <= 1.10        -> instructor.patch
        #   1.11 - 1.15.1  -> instructor.core.patch (instructor.patch still present)
        #   >= 1.15.3      -> instructor.processing.response (instructor.patch removed,
        #                     instructor.core.patch no longer re-exports it)
        # Resolve it from the first module that actually defines it. We probe with
        # ``import_module`` + an explicit ``is not None`` check rather than relying on
        # an exception, because ``getattr(module, name, None)`` does NOT raise when the
        # module exists but lacks the attribute — which previously left ``_patch_module``
        # pointing at a module without ``handle_response_model`` and made the subsequent
        # ``wrap_function_wrapper`` raise ``AttributeError`` (see #3253).
        self._patch_module = None
        self._original_handle_response_model = None
        for module_name in (
            "instructor.core.patch",
            "instructor.patch",
            "instructor.processing.response",
            "instructor.processing",
        ):
            try:
                module = import_module(module_name)
            except ModuleNotFoundError:
                continue
            original_handle_response_model = getattr(module, "handle_response_model", None)
            if original_handle_response_model is not None:
                self._patch_module = module_name
                self._original_handle_response_model = original_handle_response_model
                break

        if self._patch_module is None:
            # Don't crash instrument() if the symbol can't be found — degrade gracefully.
            logger.warning(
                "Could not locate `handle_response_model` in any known instructor module "
                "(instructor.core.patch, instructor.patch, instructor.processing.response, "
                "instructor.processing); skipping that wrapper. Instructor response-handling "
                "spans will not be emitted for this version of instructor."
            )
        else:
            process_resp_wrapper = _HandleResponseWrapper(tracer=self._tracer)  # type: ignore[arg-type]
            wrap_function_wrapper(self._patch_module, "handle_response_model", process_resp_wrapper)

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._original_patch is not None:
            instructor_module = import_module("instructor")
            instructor_module.patch = self._original_patch  # type: ignore[attr-defined]
            self._original_patch = None

        if self._patch_module is not None and self._original_handle_response_model is not None:
            patch_module = import_module(self._patch_module)
            patch_module.handle_response_model = self._original_handle_response_model  # type: ignore[attr-defined]
            self._original_handle_response_model = None
