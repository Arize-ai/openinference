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

_instruments = ("google-adk >= 2.10.0",)

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
        from google.adk.flows.llm_flows.core import _model_call

        from openinference.instrumentation.google_adk._wrappers import _TraceCallLlm

        setattr(_model_call, "tracer", self._tracer)
        setattr(
            _model_call,
            "trace_call_llm",
            _TraceCallLlm(self._tracer)(
                _model_call.trace_call_llm  # type: ignore[attr-defined]
            ),
        )

    def _unpatch_trace_call_llm(self) -> None:
        """Restore the original LLM call tracing functionality."""
        from google.adk.flows.llm_flows.core import _model_call
        from google.adk.telemetry import tracer

        trace_call_llm = _model_call.trace_call_llm  # type: ignore[attr-defined]
        if callable(original := getattr(trace_call_llm, "__wrapped__", None)):
            setattr(_model_call, "trace_call_llm", original)

        setattr(_model_call, "tracer", tracer)

    def _patch_trace_tool_call(self) -> None:
        """Patch the tool call tracing functionality to use our tracer."""
        from google.adk.telemetry import tracing

        from openinference.instrumentation.google_adk._wrappers import _TraceToolCall

        # tracing.tracer is the shared ADK tracer. _disable_existing_tracers wraps
        # it, so this method only wraps trace_tool_call.
        setattr(
            tracing,
            "trace_tool_call",
            _TraceToolCall(self._tracer)(tracing.trace_tool_call),
        )

    def _unpatch_trace_tool_call(self) -> None:
        """Restore the original tool call tracing functionality."""
        from google.adk.telemetry import tracing

        if callable(
            original := getattr(tracing.trace_tool_call, "__wrapped__", None),
        ):
            setattr(tracing, "trace_tool_call", original)

    def _disable_existing_tracers(self) -> None:
        """Disable existing tracers to prevent double-instrumentation."""
        from google.adk.runners import (  # type: ignore[attr-defined]
            tracer,  # pyright: ignore[reportPrivateImportUsage]
        )

        if isinstance(tracer, Tracer):
            from google.adk import runners

            setattr(runners, "tracer", _PassthroughTracer(tracer))

        # tracing.tracer drives execute_tool, invoke_agent, and generate_content.
        # Emit OI spans for the tool family and suppress the others.
        from google.adk.telemetry import tracing as adk_tracing

        adk_proxy: Optional[Tracer] = None
        if isinstance(adk_tracing.tracer, Tracer):
            original_adk_tracer = adk_tracing.tracer
            adk_proxy = cast(Tracer, _SelectiveExecuteToolTracer(original_adk_tracer, self._tracer))
            self._tracer_patches.append((adk_tracing, "tracer", original_adk_tracer, adk_proxy))
            setattr(adk_tracing, "tracer", adk_proxy)
        # execute_tool (merged) uses a module-local tracer captured at import
        # time. Reassigning tracing.tracer does not reach it.
        for merged_module in _merged_tool_span_modules():
            merged_tracer = getattr(merged_module, "tracer", None)
            if isinstance(merged_tracer, Tracer):
                merged_proxy = _SelectiveExecuteToolTracer(merged_tracer, self._tracer)
                self._tracer_patches.append((merged_module, "tracer", merged_tracer, merged_proxy))
                setattr(merged_module, "tracer", merged_proxy)
        self._patch_compaction_helpers(adk_tracing, adk_proxy)

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

        from google.adk.telemetry import tracing as adk_tracing

        if isinstance(original := getattr(adk_tracing.tracer, "__wrapped__", None), Tracer):
            setattr(adk_tracing, "tracer", original)

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

    Why the merged-span module's ``tracer`` is patched separately
    -------------------------------------------------------------
    The parallel-call ``execute_tool (merged)`` span is created in a module that
    did ``from ...telemetry.tracing import tracer`` at import time, capturing the
    original tracer in a *local* name. Later reassignments of ``tracing.tracer``
    don't reach it, so the merged span would emit through the original ADK tracer
    unless we patch that binding too. The binding lives on
    ``flows/llm_flows/tools/_batch_executor.py``. ``_disable_existing_tracers``
    wraps it (see ``_merged_tool_span_modules``) with this proxy.

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


def _merged_tool_span_modules() -> List[Any]:
    """Return the module whose local ``tracer`` creates ``execute_tool (merged)``.

    ``flows/llm_flows/tools/_batch_executor.py`` does
    ``from ...telemetry.tracing import tracer`` at import time, so a later
    reassignment of ``tracing.tracer`` does not reach that span.
    """
    from google.adk.flows.llm_flows.tools import _batch_executor

    return [_batch_executor]
