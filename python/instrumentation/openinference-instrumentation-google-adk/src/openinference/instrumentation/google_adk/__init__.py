import contextvars
import logging
import sys
from typing import Any, Collection, Dict, Iterator, List, Optional, Tuple, cast

import wrapt
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from opentelemetry.trace import Span, Tracer, get_current_span
from opentelemetry.util._decorator import _agnosticcontextmanager
from wrapt import resolve_path, wrap_function_wrapper

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_input_attributes,
    get_output_attributes,
    safe_json_dumps,
)
from openinference.instrumentation.google_adk.version import __version__
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("google-adk >= 1.2.1",)

_COMPACTION_MODULE = "google.adk.apps.compaction"

_compaction_input_var: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar(
    "_openinference_compaction_input", default=None
)


class GoogleADKInstrumentor(BaseInstrumentor):  # type: ignore
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
        from google.adk.runners import Runner

        from openinference.instrumentation.google_adk._wrappers import (
            _BaseAgentRunAsync,
            _RunnerRunAsync,
        )

        # Store original methods for cleanup during uninstrumentation
        self._originals: List[Tuple[Any, Any, Any]] = []
        self._tracer_patches: List[Tuple[Any, str, Any, Any]] = []
        method_wrappers: Dict[Any, Any] = {
            Runner.run_async: _RunnerRunAsync(self._tracer),
            BaseAgent.run_async: _BaseAgentRunAsync(self._tracer),
        }

        # Wrap each method with its corresponding tracer
        for method, wrapper in method_wrappers.items():
            module, name = method.__module__, method.__qualname__
            self._originals.append(resolve_path(module, name))
            wrap_function_wrapper(module, name, wrapper)

        self._patch_trace_call_llm()
        self._patch_trace_tool_call()
        self._disable_existing_tracers()

    def _uninstrument(self, **kwargs: Any) -> None:
        self._unpatch_trace_call_llm()
        self._unpatch_trace_tool_call()
        self._restore_existing_tracers()

        # Restore all wrapped methods to their original state
        for parent, attribute, original in getattr(self, "_originals", ()):
            setattr(parent, attribute, original)

    def _patch_trace_call_llm(self) -> None:
        """Patch the LLM call tracing functionality to use our tracer."""
        from google.adk.flows.llm_flows import base_llm_flow

        from openinference.instrumentation.google_adk._wrappers import _TraceCallLlm

        setattr(base_llm_flow, "tracer", self._tracer)
        setattr(
            base_llm_flow,
            "trace_call_llm",
            _TraceCallLlm(self._tracer)(base_llm_flow.trace_call_llm),  # type: ignore[attr-defined]
        )

    def _unpatch_trace_call_llm(self) -> None:
        """Restore the original LLM call tracing functionality."""
        from google.adk.flows.llm_flows import base_llm_flow

        if callable(
            original := getattr(base_llm_flow.trace_call_llm, "__wrapped__"),  # type: ignore[attr-defined]
        ):
            from google.adk.flows.llm_flows import (
                base_llm_flow,
            )

            setattr(base_llm_flow, "trace_call_llm", original)

        from google.adk.telemetry import tracer

        setattr(base_llm_flow, "tracer", tracer)

    def _patch_trace_tool_call(self) -> None:
        """Patch the tool call tracing functionality to use our tracer."""
        from openinference.instrumentation.google_adk._wrappers import _TraceToolCall

        target = _resolve_trace_tool_call_module()
        # On ADK < 1.32 the target is google.adk.flows.llm_flows.functions, whose
        # local `tracer` attr is used for ad-hoc tool spans we want to convert to OI
        # spans. On ADK >= 1.32 the target is google.adk.telemetry.tracing — its
        # `tracer` attr is the global ADK tracer that `_disable_existing_tracers`
        # swaps with a _SelectiveExecuteToolTracer to convert `execute_tool *` spans
        # to OI spans while passing through other operations, so we leave it alone here.
        if _adk_version() < (1, 32, 0):
            setattr(target, "tracer", self._tracer)
        setattr(
            target,
            "trace_tool_call",
            _TraceToolCall(self._tracer)(target.trace_tool_call),
        )

    def _unpatch_trace_tool_call(self) -> None:
        """Restore the original tool call tracing functionality."""
        target = _resolve_trace_tool_call_module()

        if callable(
            original := getattr(target.trace_tool_call, "__wrapped__", None),
        ):
            setattr(target, "trace_tool_call", original)

        if _adk_version() < (1, 32, 0):
            from google.adk.telemetry import tracer

            setattr(target, "tracer", tracer)

    def _disable_existing_tracers(self) -> None:
        """Disable existing tracers to prevent double-instrumentation."""
        from google.adk.runners import (  # type: ignore[attr-defined]
            tracer,  # pyright: ignore[reportPrivateImportUsage]
        )

        if isinstance(tracer, Tracer):
            from google.adk import runners

            setattr(runners, "tracer", _PassthroughTracer(tracer))

        # ADK 1.32 removed `tracer` from google.adk.agents.base_agent.
        # Skip patching it on newer ADK; telemetry.tracing.tracer below covers the path.
        if _adk_version() < (1, 32, 0):
            from google.adk.agents.base_agent import (  # type: ignore[attr-defined,unused-ignore]
                tracer as base_agent_tracer,  # pyright: ignore[reportPrivateImportUsage]
            )

            if isinstance(base_agent_tracer, Tracer):
                from google.adk.agents import base_agent

                setattr(base_agent, "tracer", _PassthroughTracer(base_agent_tracer))

        if _adk_version() >= (1, 32, 0):
            # ADK 1.32 consolidated tool + agent telemetry: a single shared
            # `tracing.tracer` now drives `execute_tool {name}` (via
            # `_instrumentation.record_tool_execution`), `invoke_agent {name}` (via
            # `record_agent_invocation`), and the experimental
            # `generate_content {model}` path. We want OI spans for the tool family
            # but suppression for the others — so wrap with a name-dispatching
            # proxy. See `_SelectiveExecuteToolTracer` for the full rationale.
            from google.adk.flows.llm_flows import functions
            from google.adk.telemetry import (  # type: ignore[attr-defined,import-not-found,unused-ignore]
                tracing as adk_tracing,  # type: ignore[attr-defined,unused-ignore]
            )

            adk_proxy: Optional[Tracer] = None
            if isinstance(adk_tracing.tracer, Tracer):
                original_adk_tracer = adk_tracing.tracer
                adk_proxy = cast(
                    Tracer, _SelectiveExecuteToolTracer(original_adk_tracer, self._tracer)
                )
                self._tracer_patches.append((adk_tracing, "tracer", original_adk_tracer, adk_proxy))
                setattr(adk_tracing, "tracer", adk_proxy)
            # `functions.tracer` is a *separate* binding: `functions.py` does
            # `from ...telemetry.tracing import tracer` at import time, capturing
            # the original tracer locally. Reassigning `tracing.tracer` above
            # won't reach it, and it's still used for parallel-call
            # `execute_tool (merged)` spans, so wrap it independently.
            functions_tracer = getattr(functions, "tracer", None)
            if isinstance(functions_tracer, Tracer):
                setattr(
                    functions,
                    "tracer",
                    _SelectiveExecuteToolTracer(functions_tracer, self._tracer),
                )
            self._patch_compaction_helpers(adk_tracing, adk_proxy)
        elif _adk_version() >= (1, 15, 0):
            from google.adk.telemetry import (  # type: ignore[attr-defined,import-not-found,unused-ignore]
                tracing as adk_tracing,  # type: ignore[attr-defined,unused-ignore]
            )

            if isinstance(adk_tracing.tracer, Tracer):
                setattr(adk_tracing, "tracer", _PassthroughTracer(adk_tracing.tracer))

    def _patch_compaction_helpers(self, adk_tracing: Any, adk_proxy: Optional[Tracer]) -> None:
        """Ensure apps.compaction resolves our patched `tracer`,
        `_build_compaction_attributes`, and `_build_compaction_result_attributes`,
        regardless of whether it's already loaded.
        """
        wrapped_input_builder = _wrap_build_compaction_attributes(
            adk_tracing._build_compaction_attributes
        )
        wrapped_result_builder = _wrap_build_compaction_result_attributes(
            adk_tracing._build_compaction_result_attributes
        )
        self._tracer_patches.append(
            (
                adk_tracing,
                "_build_compaction_attributes",
                adk_tracing._build_compaction_attributes,
                wrapped_input_builder,
            )
        )
        setattr(adk_tracing, "_build_compaction_attributes", wrapped_input_builder)
        self._tracer_patches.append(
            (
                adk_tracing,
                "_build_compaction_result_attributes",
                adk_tracing._build_compaction_result_attributes,
                wrapped_result_builder,
            )
        )
        setattr(adk_tracing, "_build_compaction_result_attributes", wrapped_result_builder)

        compaction = sys.modules.get(_COMPACTION_MODULE)
        if compaction is None:
            return
        if adk_proxy is not None:
            compaction_tracer = getattr(compaction, "tracer", None)
            if compaction_tracer is not adk_proxy and isinstance(compaction_tracer, Tracer):
                self._tracer_patches.append((compaction, "tracer", compaction_tracer, adk_proxy))
                setattr(compaction, "tracer", adk_proxy)
        if getattr(compaction, "_build_compaction_attributes", None) is not wrapped_input_builder:
            original_input_builder = compaction._build_compaction_attributes
            self._tracer_patches.append(
                (
                    compaction,
                    "_build_compaction_attributes",
                    original_input_builder,
                    wrapped_input_builder,
                )
            )
            setattr(compaction, "_build_compaction_attributes", wrapped_input_builder)
        if (
            getattr(compaction, "_build_compaction_result_attributes", None)
            is not wrapped_result_builder
        ):
            original_result_builder = compaction._build_compaction_result_attributes
            self._tracer_patches.append(
                (
                    compaction,
                    "_build_compaction_result_attributes",
                    original_result_builder,
                    wrapped_result_builder,
                )
            )
            setattr(compaction, "_build_compaction_result_attributes", wrapped_result_builder)

    def _restore_compaction_helpers(self) -> None:
        """Undo `_patch_compaction_helpers`, including the case where
        apps.compaction loaded *during* the instrumented session
        """
        compaction = sys.modules.get(_COMPACTION_MODULE)
        if compaction is not None:
            explicitly_patched_attrs = set()
            for module, attr, original, replacement in self._tracer_patches:
                if module is compaction and getattr(compaction, attr, None) is replacement:
                    setattr(compaction, attr, original)
                    explicitly_patched_attrs.add(attr)

            for module, attr, original, replacement in self._tracer_patches:
                if (
                    module is not compaction
                    and attr not in explicitly_patched_attrs
                    and getattr(compaction, attr, None) is replacement
                ):
                    setattr(compaction, attr, original)

        for module, attr, original, replacement in reversed(self._tracer_patches):
            if getattr(module, attr, None) is replacement:
                setattr(module, attr, original)

        self._tracer_patches = []

    def _restore_existing_tracers(self) -> None:
        """Restore original tracers that were disabled during instrumentation."""
        from google.adk.runners import (  # type: ignore[attr-defined]
            tracer,  # pyright: ignore[reportPrivateImportUsage]
        )

        if isinstance(original := getattr(tracer, "__wrapped__"), Tracer):
            from google.adk import runners

            setattr(runners, "tracer", original)

        if _adk_version() < (1, 32, 0):
            from google.adk.agents.base_agent import (  # type: ignore[attr-defined,unused-ignore]
                tracer as base_agent_tracer,  # pyright: ignore[reportPrivateImportUsage]
            )

            if isinstance(original := getattr(base_agent_tracer, "__wrapped__"), Tracer):
                from google.adk.agents import base_agent

                setattr(base_agent, "tracer", original)

        if _adk_version() >= (1, 15, 0):
            from google.adk.telemetry import (  # type: ignore[attr-defined,import-not-found,unused-ignore]
                tracing as adk_tracing,  # type: ignore[attr-defined,unused-ignore]
            )

            if isinstance(original := getattr(adk_tracing.tracer, "__wrapped__", None), Tracer):
                setattr(adk_tracing, "tracer", original)

        if _adk_version() >= (1, 32, 0):
            from google.adk.flows.llm_flows import functions

            functions_tracer = getattr(functions, "tracer", None)
            if isinstance(original := getattr(functions_tracer, "__wrapped__", None), Tracer):
                setattr(functions, "tracer", original)

            self._restore_compaction_helpers()


class _PassthroughTracer(wrapt.ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    """Tracer proxy that suppresses span creation by yielding the current span.

    Used to neutralize an ADK-internal tracer whose spans would duplicate work that
    one of our outer wrappers (e.g. ``_RunnerRunAsync``, ``_BaseAgentRunAsync``,
    ``_TraceCallLlm``) is already producing as an OpenInference span. ADK callers
    still get back a span object from ``start_as_current_span`` — they just keep
    writing into the OI span we opened upstream.

    Use this when *every* span the wrapped tracer produces is unwanted. If the
    tracer is shared across multiple span types and only some are unwanted, use
    :class:`_SelectiveExecuteToolTracer` instead.
    """

    @_agnosticcontextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        yield get_current_span()


def _wrap_build_compaction_attributes(original: Any) -> Any:
    """Wraps ``telemetry.tracing._build_compaction_attributes``: captures its
    return value (the compaction request -- trigger, summarizer type, event
    count, thresholds; never raw event content) into ``_compaction_input_var``
    for ``_SelectiveExecuteToolTracer`` to pick up when it opens the
    ``compact_events`` span a few lines later. Never alters the return value
    ADK itself uses."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        try:
            input_attributes = {
                key: value
                for key, value in result.items()
                if isinstance(key, str) and key.startswith("gen_ai.compaction.")
            }
            _compaction_input_var.set(input_attributes or None)
        except Exception:
            logger.exception("Failed to capture compaction span input.")
        return result

    return wrapper


def _wrap_build_compaction_result_attributes(original: Any) -> Any:
    """Wraps ``telemetry.tracing._build_compaction_result_attributes``: sets
    ``output.value`` directly on ``get_current_span()``, since this function
    runs *while* the ``compact_events`` span is current (inside
    ``_summarize_events_with_trace``'s ``with`` block) -- no need to read the
    span back afterward. Only runs if the summarizer actually returned, so a
    raised exception never produces a fabricated result."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        try:
            get_current_span().set_attributes(
                get_output_attributes(
                    safe_json_dumps(dict(result) if result else {"compacted": False}),
                    mime_type=OpenInferenceMimeTypeValues.JSON,
                )
            )
        except Exception:
            logger.exception("Failed to set compaction span output.")
        return result

    return wrapper


class _SelectiveExecuteToolTracer(wrapt.ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    """Tracer proxy that emits OI spans for tool/compaction spans and suppresses the rest.

    Why this exists
    ---------------
    Pre-1.32, ADK created spans through several module-level ``tracer`` attributes,
    one per span family — ``base_agent.tracer`` for ``invoke_agent {name}``,
    ``functions.tracer`` for ``execute_tool {name}`` / ``execute_tool (merged)``,
    ``tracing.tracer`` for the experimental ``generate_content {model}`` path. The
    instrumentor patched each one independently: ``OITracer`` where we wanted OI
    spans (the tool path), :class:`_PassthroughTracer` where our outer wrappers
    already covered it (agent + LLM paths).

    ADK 1.32 consolidated tool and agent telemetry into ``telemetry/_instrumentation.py``,
    which calls ``tracing.tracer.start_as_current_span(...)`` for *both* ``invoke_agent``
    and ``execute_tool`` spans. A single shared object now drives three families:

    - ``invoke_agent {name}``   → suppress (``_BaseAgentRunAsync`` produces ``agent_run``)
    - ``generate_content ...``  → suppress (``_TraceCallLlm`` produces ``call_llm``)
    - ``execute_tool {name}``   → emit as OI span (``_TraceToolCall`` enriches it)
    - ``execute_tool (merged)`` → emit as OI span (parallel-call summary)

    A blanket :class:`_PassthroughTracer` swallows the tool spans — leaving
    ``_TraceToolCall`` to write TOOL attributes onto the parent ``call_llm`` span
    and producing no tool span at all. A blanket ``OITracer`` swap goes the other
    way, emitting duplicate ``invoke_agent`` and ``generate_content`` spans
    alongside the OI ``agent_run`` / ``call_llm`` spans we already create.

    This proxy routes by span name: forward to the OI tracer for
    ``execute_tool *`` and ``compact_events *`` (so ``_TraceToolCall`` and ADK
    compaction each get a real span), passthrough for everything else.

    Why ``functions.tracer`` is patched separately
    ----------------------------------------------
    ``flows/llm_flows/functions.py`` does ``from ...telemetry.tracing import tracer``
    at import time, capturing the original tracer in a *local* name. Later
    reassignments of ``tracing.tracer`` don't reach it, so the parallel-call
    ``execute_tool (merged)`` span (still created in ``functions.py`` on 1.32)
    would emit through the original ADK tracer unless we patch ``functions.tracer``
    too. ``_disable_existing_tracers`` wraps both attributes with this proxy.

    Implementation note
    -------------------
    The ``_self_oi_tracer`` prefix follows the ``wrapt.ObjectProxy`` convention:
    attributes named ``_self_*`` live on the proxy itself rather than being
    delegated to the wrapped object, which lets us hold the OI tracer reference
    without colliding with the underlying ADK tracer's namespace.
    """

    def __init__(self, wrapped: Tracer, oi_tracer: Tracer) -> None:
        super().__init__(wrapped)
        self._self_oi_tracer = oi_tracer

    @_agnosticcontextmanager
    def start_as_current_span(self, name: str, *args: Any, **kwargs: Any) -> Iterator[Span]:
        is_compaction = isinstance(name, str) and name.startswith("compact_events ")
        if is_compaction or (isinstance(name, str) and name.startswith("execute_tool")):
            # Tool/compaction path — produce a real OI span; _TraceToolCall
            # enriches tool spans via `get_current_span()` once
            # `tracing.trace_tool_call(...)` runs inside.
            if is_compaction:
                captured_input = _compaction_input_var.get()
                _compaction_input_var.set(None)
                compaction_attributes: Dict[str, Any] = {
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                }
                if captured_input:
                    try:
                        compaction_attributes.update(
                            get_input_attributes(
                                safe_json_dumps(captured_input),
                                mime_type=OpenInferenceMimeTypeValues.JSON,
                            )
                        )
                    except Exception:
                        logger.exception("Failed to set compaction span input.")
                kwargs = dict(kwargs)
                kwargs["attributes"] = {**kwargs.get("attributes", {}), **compaction_attributes}
            with self._self_oi_tracer.start_as_current_span(name, *args, **kwargs) as span:
                yield span
            return
        # Agent / experimental-LLM paths — already covered by our outer wrappers,
        # so suppress to avoid duplicate spans.
        yield get_current_span()


def _adk_version() -> Tuple[int, int, int]:
    """Return the installed google-adk version as a (major, minor, patch) tuple."""
    from google.adk import __version__

    return cast(Tuple[int, int, int], tuple(int(x) for x in __version__.split(".")[:3]))


def _resolve_trace_tool_call_module() -> Any:
    """Return the module that exposes ``trace_tool_call`` for the installed ADK.

    ADK 1.32 moved ``trace_tool_call`` from ``google.adk.flows.llm_flows.functions``
    to ``google.adk.telemetry.tracing``. Both modules also carry a ``tracer``
    attribute that the instrumentor swaps in.
    """
    if _adk_version() >= (1, 32, 0):
        from google.adk.telemetry import (  # type: ignore[attr-defined,import-not-found,unused-ignore]
            tracing as adk_tracing,  # type: ignore[attr-defined,unused-ignore]
        )

        return adk_tracing

    from google.adk.flows.llm_flows import functions

    return functions
