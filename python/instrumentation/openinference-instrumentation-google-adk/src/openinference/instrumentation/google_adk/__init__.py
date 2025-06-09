import logging
from typing import Any, Collection, Iterator, List, Tuple, cast

import wrapt
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from opentelemetry.trace import Span, Tracer, get_current_span
from opentelemetry.util._decorator import _agnosticcontextmanager
from wrapt import resolve_path, wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.google_adk.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("google-adk >= 1.2.1",)


class GoogleADKInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for google-adk
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)

        self._tracer = cast(
            Tracer,
            OITracer(
                trace_api.get_tracer(__name__, __version__, tracer_provider),
                config=config,
            ),
        )

        from google.adk.agents import BaseAgent
        from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
        from google.adk.runners import Runner

        from openinference.instrumentation.google_adk._wrappers import (
            _BaseAgentRunAsync,
            _BaseLlmFlowCallLlmAsync,
            _RunnerRunAsync,
            _TraceToolCall,
        )

        self._originals: List[Tuple[Any, Any, Any]] = []
        method_wrappers: dict[Any, Any] = {
            Runner.run_async: _RunnerRunAsync(self._tracer),
            BaseAgent.run_async: _BaseAgentRunAsync(self._tracer),
            BaseLlmFlow._call_llm_async: _BaseLlmFlowCallLlmAsync(self._tracer),
        }
        for method, wrapper in method_wrappers.items():
            module, name = method.__module__, method.__qualname__
            self._originals.append(resolve_path(module, name))
            wrap_function_wrapper(module, name, wrapper)

        from google.adk.flows.llm_flows.base_llm_flow import functions  # type: ignore[attr-defined]

        setattr(functions, "tracer", self._tracer)
        setattr(
            functions,
            "trace_tool_call",
            _TraceToolCall(self._tracer)(functions.trace_tool_call),  # type: ignore[attr-defined]
        )
        _disable_existing_tracers()

    def _uninstrument(self, **kwargs: Any) -> None:
        _restore_existing_tracers()

        from google.adk.flows.llm_flows.base_llm_flow import functions  # type: ignore[attr-defined]

        if callable(
            original := getattr(functions.trace_tool_call, "__wrapped__"),  # type: ignore[attr-defined]
        ):
            from google.adk.flows.llm_flows.base_llm_flow import (  # type: ignore[attr-defined]
                functions,
            )  # type ignore[attr-defined]

            setattr(functions, "trace_tool_call", original)

        from google.adk.telemetry import tracer

        setattr(functions, "tracer", tracer)

        for parent, attribute, original in getattr(self, "_originals", ()):
            setattr(parent, attribute, original)


class _PassthroughTracer(wrapt.ObjectProxy):  # type: ignore[misc]
    @_agnosticcontextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        yield get_current_span()


def _disable_existing_tracers() -> None:
    from google.adk.runners import (  # type: ignore[attr-defined]
        tracer,  # pyright: ignore[reportPrivateImportUsage]
    )

    if isinstance(tracer, Tracer):
        from google.adk import runners

        setattr(runners, "tracer", _PassthroughTracer(tracer))

    from google.adk.agents.base_agent import (
        tracer,  # pyright: ignore[reportPrivateImportUsage]
    )

    if isinstance(tracer, Tracer):
        from google.adk.agents import base_agent

        setattr(base_agent, "tracer", _PassthroughTracer(tracer))

    from google.adk.flows.llm_flows.base_llm_flow import (  # type: ignore[attr-defined]
        tracer,  # pyright: ignore[reportPrivateImportUsage]
    )

    if isinstance(tracer, Tracer):
        from google.adk.flows.llm_flows import base_llm_flow

        setattr(base_llm_flow, "tracer", _PassthroughTracer(tracer))


def _restore_existing_tracers() -> None:
    from google.adk.runners import (  # type: ignore[attr-defined]
        tracer,  # pyright: ignore[reportPrivateImportUsage]
    )

    if isinstance(original := getattr(tracer, "__wrapped__"), Tracer):
        from google.adk import runners

        setattr(runners, "tracer", original)

    from google.adk.agents.base_agent import (
        tracer,  # pyright: ignore[reportPrivateImportUsage]
    )

    if isinstance(original := getattr(tracer, "__wrapped__"), Tracer):
        from google.adk.agents import base_agent

        setattr(base_agent, "tracer", original)

    from google.adk.flows.llm_flows.base_llm_flow import (  # type: ignore[attr-defined]
        tracer,  # pyright: ignore[reportPrivateImportUsage]
    )

    if isinstance(original := getattr(tracer, "__wrapped__"), Tracer):
        from google.adk.flows.llm_flows import base_llm_flow

        setattr(base_llm_flow, "tracer", original)
