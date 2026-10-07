# pyright: reportPrivateImportUsage=false
# mypy: disable-error-code="attr-defined"

"""Tests for Google ADK instrumentation patching and unpatching."""

import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Iterator, Optional, cast

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer, get_current_span

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.google_adk import (
    _COMPACTION_MODULE,
    GoogleADKInstrumentor,
    _compaction_input_var,
    _merged_tool_span_modules,
    _PassthroughTracer,
    _SelectiveExecuteToolTracer,
    _workflow_span_modules,
)
from openinference.semconv.trace import SpanAttributes


def test_instrumentation_patching() -> None:
    """Test that all instrumentation patching and unpatching works correctly."""
    from google.adk import runners
    from google.adk.agents import BaseAgent
    from google.adk.flows.llm_flows.core import _model_call
    from google.adk.runners import Runner
    from google.adk.telemetry import tracing

    compaction = sys.modules.get(_COMPACTION_MODULE)
    original_merged_tracers = [
        (module, module.tracer)
        for module in _merged_tool_span_modules() + _workflow_span_modules()
        if hasattr(module, "tracer")
    ]

    original_runner_run_async = Runner.run_async
    original_agent_run_async = BaseAgent.run_async
    original_runners_tracer = runners.tracer
    original_llm_flow_tracer = _model_call.tracer
    original_trace_call_llm = _model_call.trace_call_llm
    original_trace_tool_module_tracer = tracing.tracer
    original_trace_tool_call = tracing.trace_tool_call
    original_build_attrs = getattr(tracing, "_build_compaction_attributes", None)
    original_build_result_attrs = getattr(tracing, "_build_compaction_result_attributes", None)
    if compaction is not None:
        original_compaction_tracer = compaction.tracer
        original_compaction_build_attrs = compaction._build_compaction_attributes
        original_compaction_build_result_attrs = compaction._build_compaction_result_attributes

    GoogleADKInstrumentor().instrument()

    assert Runner.run_async is not original_runner_run_async
    assert BaseAgent.run_async is not original_agent_run_async
    assert runners.tracer is not original_runners_tracer
    assert _model_call.tracer is not original_llm_flow_tracer
    assert _model_call.trace_call_llm is not original_trace_call_llm
    assert tracing.tracer is not original_trace_tool_module_tracer
    assert tracing.trace_tool_call is not original_trace_tool_call
    assert tracing._build_compaction_attributes is not original_build_attrs
    assert tracing._build_compaction_result_attributes is not original_build_result_attrs
    if compaction is not None:
        assert compaction.tracer is not original_compaction_tracer
        assert compaction._build_compaction_attributes is not original_compaction_build_attrs
        assert (
            compaction._build_compaction_result_attributes
            is not original_compaction_build_result_attrs
        )
        assert compaction.tracer is tracing.tracer
        assert compaction._build_compaction_attributes is tracing._build_compaction_attributes
        assert (
            compaction._build_compaction_result_attributes
            is tracing._build_compaction_result_attributes
        )

    assert isinstance(runners.tracer, _PassthroughTracer)
    assert isinstance(_model_call.tracer, OITracer)
    assert isinstance(tracing.tracer, _SelectiveExecuteToolTracer)
    assert original_merged_tracers
    for merged_module, _ in original_merged_tracers:
        assert isinstance(merged_module.tracer, _SelectiveExecuteToolTracer)
    if compaction is not None:
        assert isinstance(compaction.tracer, _SelectiveExecuteToolTracer)

    GoogleADKInstrumentor().uninstrument()

    assert Runner.run_async is original_runner_run_async
    assert BaseAgent.run_async is original_agent_run_async
    assert runners.tracer is original_runners_tracer
    assert _model_call.tracer is original_llm_flow_tracer
    assert _model_call.trace_call_llm is original_trace_call_llm
    assert tracing.tracer is original_trace_tool_module_tracer
    assert tracing.trace_tool_call is original_trace_tool_call
    for merged_module, original_tracer in original_merged_tracers:
        assert merged_module.tracer is original_tracer
    assert tracing._build_compaction_attributes is original_build_attrs
    assert tracing._build_compaction_result_attributes is original_build_result_attrs
    if compaction is not None:
        assert compaction.tracer is original_compaction_tracer
        assert compaction._build_compaction_attributes is original_compaction_build_attrs
        assert (
            compaction._build_compaction_result_attributes is original_compaction_build_result_attrs
        )


def test_uninstrument_preserves_later_merged_tracer(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    original_tracers = [
        (module, module.tracer)
        for module in _merged_tool_span_modules()
        if hasattr(module, "tracer")
    ]
    assert original_tracers
    replacement = OITracer(
        tracer_provider.get_tracer("application"), config=TraceConfig(hide_inputs=True)
    )
    instrumentor = GoogleADKInstrumentor()
    try:
        instrumentor.instrument(tracer_provider=tracer_provider)
        try:
            for module, _ in original_tracers:
                module.tracer = replacement
        finally:
            instrumentor.uninstrument()

        for module, _ in original_tracers:
            with module.tracer.start_as_current_span(
                "application-span", attributes={SpanAttributes.INPUT_VALUE: "synthetic-input"}
            ):
                pass
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == len(original_tracers)
        for span in spans:
            assert span.attributes is not None
            assert span.attributes[SpanAttributes.INPUT_VALUE] == "__REDACTED__"
        for module, _ in original_tracers:
            assert module.tracer is replacement
    finally:
        for module, original_tracer in original_tracers:
            module.tracer = original_tracer


class _DummySpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_attributes(self, attributes: "dict[str, object]") -> None:
        self.attributes.update(attributes)


class _DummyTracer:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.span = _DummySpan()

    @contextmanager
    def start_as_current_span(
        self, name: str, *_: object, attributes: "Optional[dict[str, object]]" = None, **__: object
    ) -> Iterator[object]:
        self.names.append(name)
        # A fresh span per call, like a real tracer -- attributes from one
        # `compact_events` call must never bleed into the next.
        self.span = _DummySpan()
        if attributes:
            self.span.set_attributes(attributes)
        yield self.span


def test_selective_tracer_routes_compaction_to_oi_tracer() -> None:
    wrapped = _DummyTracer()
    oi = _DummyTracer()
    tracer = _SelectiveExecuteToolTracer(cast(Tracer, wrapped), cast(Tracer, oi))

    with tracer.start_as_current_span("execute_tool weather") as span:
        assert cast(object, span) is oi.span

    with tracer.start_as_current_span("compact_events sliding_window") as span:
        assert cast(object, span) is oi.span
        assert cast(_DummySpan, span).attributes.get("openinference.span.kind") == "CHAIN"
        # Nothing populated _compaction_input_var in this test.
        assert "input.value" not in cast(_DummySpan, span).attributes

    # Near-miss: no trailing space after "compact_events" must NOT match -- ADK
    # always emits `f'compact_events {trigger}'`, so tightening the prefix loses
    # nothing but excludes lookalikes like "compact_eventside".
    with tracer.start_as_current_span("compact_eventside") as span:
        assert span is get_current_span()

    with tracer.start_as_current_span("invoke_agent planner") as span:
        assert span is get_current_span()

    assert oi.names == ["execute_tool weather", "compact_events sliding_window"]


def test_selective_tracer_applies_captured_compaction_input() -> None:
    """`_compaction_input_var` is read (and consumed) exactly once, at span
    creation -- this is the bridge `_wrap_build_compaction_attributes` uses
    to get the compaction request onto the span without ever reading
    `span.attributes` back afterward."""
    wrapped = _DummyTracer()
    oi = _DummyTracer()
    tracer = _SelectiveExecuteToolTracer(cast(Tracer, wrapped), cast(Tracer, oi))

    token = _compaction_input_var.set({"gen_ai.compaction.trigger": "sliding_window"})
    try:
        with tracer.start_as_current_span("compact_events sliding_window") as span:
            attributes = cast(_DummySpan, span).attributes
            assert attributes.get("input.mime_type") == "application/json"
            assert (
                attributes.get("input.value") == '{"gen_ai.compaction.trigger": "sliding_window"}'
            )
    finally:
        _compaction_input_var.reset(token)

    # Consumed -- a second span without repopulating the var gets no input.value.
    with tracer.start_as_current_span("compact_events sliding_window") as span:
        assert "input.value" not in cast(_DummySpan, span).attributes


def _fake_compaction_module(
    *, tracer: object, build_attrs: object, build_result_attrs: object
) -> ModuleType:
    module = ModuleType(_COMPACTION_MODULE)
    module.tracer = tracer
    module._build_compaction_attributes = build_attrs
    module._build_compaction_result_attributes = build_result_attrs
    return module


@contextmanager
def _compaction_module_absent() -> Iterator[None]:
    """Temporarily remove ``google.adk.apps.compaction`` from ``sys.modules``.

    Restores the *exact* original module object on exit -- never re-imports --
    so tests using this cannot create a second module object while something
    else (e.g. ``runners.py`` on ADK 1.32) still holds a reference to the first.
    This is what makes the lifecycle tests below hermetic and order-independent
    regardless of whether an earlier test already imported the real module.
    """
    saved = sys.modules.pop(_COMPACTION_MODULE, None)
    try:
        yield
    finally:
        if saved is not None:
            sys.modules[_COMPACTION_MODULE] = saved
        else:
            sys.modules.pop(_COMPACTION_MODULE, None)


def test_compaction_module_preloaded_is_explicitly_patched_and_restored() -> None:
    """If apps.compaction is already imported when we instrument, its local
    `tracer`/`_build_compaction_attributes`/`_build_compaction_result_attributes`
    names were captured before we patched anything -- each must be rebound
    explicitly (not just inherited via alias) and restored to its *own*
    original.

    Deliberately distinct from telemetry.tracing's own pre-patch values (not
    just "whatever apps.compaction would realistically start with") --
    restoring apps.compaction to the *source's* original instead of its own
    is exactly the bug this test catches: both a source-patch record and a
    compaction-module record share the same replacement object, and scanning
    the source record first (wrong order) silently restores the wrong value.
    """
    from google.adk.telemetry import tracing as adk_tracing

    def _dummy_build_attrs(*args: object, **kwargs: object) -> "dict[str, object]":
        return {}

    def _dummy_build_result_attrs(*args: object, **kwargs: object) -> "dict[str, object]":
        return {}

    compaction_original_tracer = TracerProvider().get_tracer("compaction-dummy")
    compaction_original_build_attrs = _dummy_build_attrs
    compaction_original_build_result_attrs = _dummy_build_result_attrs

    fake = _fake_compaction_module(
        tracer=compaction_original_tracer,
        build_attrs=compaction_original_build_attrs,
        build_result_attrs=compaction_original_build_result_attrs,
    )
    with _compaction_module_absent():
        sys.modules[_COMPACTION_MODULE] = fake

        GoogleADKInstrumentor().instrument()
        try:
            assert isinstance(fake.tracer, _SelectiveExecuteToolTracer)
            assert cast(object, fake.tracer) is adk_tracing.tracer
            assert fake._build_compaction_attributes is adk_tracing._build_compaction_attributes
            assert (
                fake._build_compaction_result_attributes
                is adk_tracing._build_compaction_result_attributes
            )
        finally:
            GoogleADKInstrumentor().uninstrument()

        assert cast(object, fake.tracer) is compaction_original_tracer
        assert fake._build_compaction_attributes is compaction_original_build_attrs
        assert fake._build_compaction_result_attributes is compaction_original_build_result_attrs


def test_compaction_module_not_preloaded_stays_untouched() -> None:
    """If apps.compaction is not loaded when we instrument, we must never
    force the import ourselves (that's the circular-dependency ADK is itself
    avoiding by deferring it) -- nothing should reference it at all."""
    with _compaction_module_absent():
        GoogleADKInstrumentor().instrument()
        try:
            assert _COMPACTION_MODULE not in sys.modules
        finally:
            GoogleADKInstrumentor().uninstrument()
        assert _COMPACTION_MODULE not in sys.modules


def test_compaction_module_inherits_alias_when_imported_during_session() -> None:
    """If apps.compaction imports *during* the instrumented session (the
    common case on ADK >= 2.x, where it's deferred until the first actual
    compaction call), `from ..telemetry.tracing import X` picks up whatever
    telemetry.tracing.X is at that moment -- our already-patched source. This
    must be detected (not re-wrapped) and correctly unwound on uninstrument."""
    from google.adk.telemetry import tracing as adk_tracing

    with _compaction_module_absent():
        GoogleADKInstrumentor().instrument()
        try:
            assert isinstance(adk_tracing.tracer, _SelectiveExecuteToolTracer)
            patched_tracer = adk_tracing.tracer
            patched_build_attrs = adk_tracing._build_compaction_attributes
            patched_build_result_attrs = adk_tracing._build_compaction_result_attributes

            # Simulate apps.compaction importing now, mid-session -- it binds
            # its own local names to whatever telemetry.tracing currently has.
            fake = _fake_compaction_module(
                tracer=adk_tracing.tracer,
                build_attrs=adk_tracing._build_compaction_attributes,
                build_result_attrs=adk_tracing._build_compaction_result_attributes,
            )
            sys.modules[_COMPACTION_MODULE] = fake
            assert fake.tracer is patched_tracer
            assert fake._build_compaction_attributes is patched_build_attrs
            assert fake._build_compaction_result_attributes is patched_build_result_attrs
        finally:
            GoogleADKInstrumentor().uninstrument()

        # Restoring telemetry.tracing must also restore the module that only
        # ever inherited the alias -- nothing was explicitly tracked for it.
        assert not isinstance(fake.tracer, _SelectiveExecuteToolTracer)
        assert fake.tracer is adk_tracing.tracer
        assert fake._build_compaction_attributes is adk_tracing._build_compaction_attributes
        assert (
            fake._build_compaction_result_attributes
            is adk_tracing._build_compaction_result_attributes
        )


def test_compaction_module_two_instrument_cycles_stay_pristine() -> None:
    """Instrument/uninstrument twice in a row with apps.compaction preloaded
    the whole time -- the second cycle must patch and restore exactly like
    the first, with no leftover double-wrapping from the first cycle.

    Uses originals distinct from telemetry.tracing's own pre-patch values
    (see test_compaction_module_preloaded_is_explicitly_patched_and_restored)
    so a wrong-original restore after cycle 1 would surface as a failure
    in cycle 2 too, not just silently persist."""
    from google.adk.telemetry import tracing as adk_tracing

    def _dummy_build_attrs(*args: object, **kwargs: object) -> "dict[str, object]":
        return {}

    def _dummy_build_result_attrs(*args: object, **kwargs: object) -> "dict[str, object]":
        return {}

    # Stable across both cycles once correctly restored -- apps.compaction's
    # `tracer` always gets rebound to a proxy wrapping *this*, regardless of
    # apps.compaction's own prior tracer, so the no-double-wrap check below
    # must compare against it, not against `compaction_original_tracer`.
    adk_tracing_original_tracer = adk_tracing.tracer
    compaction_original_tracer = TracerProvider().get_tracer("compaction-dummy")
    compaction_original_build_attrs = _dummy_build_attrs
    compaction_original_build_result_attrs = _dummy_build_result_attrs

    fake = _fake_compaction_module(
        tracer=compaction_original_tracer,
        build_attrs=compaction_original_build_attrs,
        build_result_attrs=compaction_original_build_result_attrs,
    )
    with _compaction_module_absent():
        sys.modules[_COMPACTION_MODULE] = fake

        for _ in range(2):
            GoogleADKInstrumentor().instrument()
            try:
                assert isinstance(fake.tracer, _SelectiveExecuteToolTracer)
                # Not a proxy-of-a-proxy: unwrapping once reaches ADK's real tracer.
                assert fake.tracer.__wrapped__ is adk_tracing_original_tracer
            finally:
                GoogleADKInstrumentor().uninstrument()

            assert cast(object, fake.tracer) is compaction_original_tracer
            assert fake._build_compaction_attributes is compaction_original_build_attrs
            assert (
                fake._build_compaction_result_attributes is compaction_original_build_result_attrs
            )
