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
        "_original_v2_create_factories",
        "_v2_create_factory_wrappers",
        "_patch_module",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from openinference.instrumentation.instructor._wrappers import (
            _HandleResponseWrapper,
            _PatchWrapper,
            _V2CreateFactoryWrapper,
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

        # The generated v2 create callables contain cache lookup and retry execution.
        # Wrapping their factories produces one span around every public create call,
        # including cache hits, without double-counting retry calls.
        self._original_v2_create_factories = []
        self._v2_create_factory_wrappers = []
        try:
            v2_patch_module = import_module("instructor.v2.core.patch")
        except ModuleNotFoundError:
            v2_patch_module = None
        if v2_patch_module is not None:
            for function_name, is_async in (
                ("_create_sync_wrapper", False),
                ("_create_async_wrapper", True),
            ):
                original = getattr(v2_patch_module, function_name, None)
                if original is None:
                    continue
                self._original_v2_create_factories.append((function_name, original))
                factory_wrapper = _V2CreateFactoryWrapper(
                    tracer=self._tracer,  # type: ignore[arg-type]
                    is_async=is_async,
                )
                self._v2_create_factory_wrappers.append(factory_wrapper)
                wrap_function_wrapper("instructor.v2.core.patch", function_name, factory_wrapper)

        self._original_patch = getattr(import_module("instructor"), "patch", None)
        patch_wrapper = _PatchWrapper(tracer=self._tracer)  # type: ignore[arg-type]
        wrap_function_wrapper("instructor", "patch", patch_wrapper)

        # Resolve from the first known instructor module that actually defines it.
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
        # Clients patched while instrumented keep their generated create callables,
        # so those callables must stop tracing too.
        for factory_wrapper in getattr(self, "_v2_create_factory_wrappers", []):
            factory_wrapper.disable()
        self._v2_create_factory_wrappers = []

        for function_name, original in getattr(self, "_original_v2_create_factories", []):
            module = import_module("instructor.v2.core.patch")
            setattr(module, function_name, original)
        self._original_v2_create_factories = []

        if self._original_patch is not None:
            instructor_module = import_module("instructor")
            instructor_module.patch = self._original_patch  # type: ignore[attr-defined]
            self._original_patch = None

        if self._patch_module is not None and self._original_handle_response_model is not None:
            patch_module = import_module(self._patch_module)
            patch_module.handle_response_model = self._original_handle_response_model  # type: ignore[attr-defined]
            self._original_handle_response_model = None
