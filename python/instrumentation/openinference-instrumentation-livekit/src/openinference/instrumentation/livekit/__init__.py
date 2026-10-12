import logging
from importlib import import_module
from typing import Any, Callable, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.dependencies import get_dependency_conflicts
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.livekit._targets import LIVEKIT_WRAP_TARGETS
from openinference.instrumentation.livekit._wrappers import _make_async_wrapper
from openinference.instrumentation.livekit.package import _instruments_any
from openinference.instrumentation.livekit.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _resolve_method(module_name: str, method_qualname: str) -> tuple[Any, str]:
    module = import_module(module_name)
    class_name, _, method_name = method_qualname.rpartition(".")
    owner = getattr(module, class_name)
    return owner, method_name


class LiveKitInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    __slots__ = ("_originals", "_tracer")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments_any

    def _check_dependency_conflicts(self) -> Any:
        conflicts = [get_dependency_conflicts([dependency]) for dependency in _instruments_any]
        if any(conflict is None for conflict in conflicts):
            return None
        return conflicts[0]

    def _instrument(self, **kwargs: Any) -> None:
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
        self._originals: dict[tuple[str, str], Callable[..., Any]] = {}
        for target in LIVEKIT_WRAP_TARGETS:
            try:
                owner, method_name = _resolve_method(target.module, target.method)
            except (ImportError, AttributeError):
                logger.debug("skipping %s.%s", target.module, target.method)
                continue
            key = (target.module, target.method)
            self._originals[key] = getattr(owner, method_name)
            wrap_function_wrapper(
                target.module,
                target.method,
                _make_async_wrapper(self._tracer, target),
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        for (module_name, method_qualname), original in getattr(self, "_originals", {}).items():
            owner, method_name = _resolve_method(module_name, method_qualname)
            setattr(owner, method_name, original)


__all__ = ("LiveKitInstrumentor", "__version__")
