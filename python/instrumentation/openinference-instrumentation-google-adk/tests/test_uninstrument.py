# pyright: reportPrivateImportUsage=false
# mypy: disable-error-code="attr-defined"

"""Tests for Google ADK instrumentation patching and unpatching functionality.

This test verifies that the GoogleADKInstrumentor correctly patches and unpatchs:
- Runner.run_async method
- BaseAgent.run_async method
- All tracers (runners, agents, llm_flows, functions/telemetry.tracing, apps.compaction)
- trace_call_llm and trace_tool_call methods
- apps.compaction's compaction-attribute builder functions (input/output source)
"""

import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Iterator, Optional, cast

import pytest
from google.adk import __version__ as _ADK_VERSION_STR
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer, get_current_span

from openinference.instrumentation import OITracer
from openinference.instrumentation.google_adk import (
    _COMPACTION_MODULE,
    GoogleADKInstrumentor,
    _compaction_input_var,
    _PassthroughTracer,
    _SelectiveExecuteToolTracer,
)

_ADK_VERSION = cast(tuple[int, int, int], tuple(int(x) for x in _ADK_VERSION_STR.split(".")[:3]))


def test_instrumentation_patching() -> None:
    """Test that all instrumentation patching and unpatching works correctly.

    This test verifies that:
    1. All methods and tracers are properly wrapped during instrumentation
    2. All tracers are replaced with appropriate types (_PassthroughTracer or OITracer)
    3. All methods and tracers are restored to their original state after uninstrumentation
    """
    # Import all necessary modules
    from google.adk import runners
    from google.adk.agents import BaseAgent
    from google.adk.flows.llm_flows import base_llm_flow
    from google.adk.runners import Runner

    # ADK 1.32 moved trace_tool_call from flows.llm_flows.functions to telemetry.tracing
    # and removed the re-export of `tracer` from agents.base_agent.
    trace_tool_module: ModuleType
    compaction: Optional[ModuleType] = None
    if _ADK_VERSION >= (1, 32, 0):
        from google.adk.telemetry import tracing

        compaction = sys.modules.get(_COMPACTION_MODULE)
        trace_tool_module = tracing
    else:
        from google.adk.flows.llm_flows import functions

        trace_tool_module = functions

    # Store original state of all methods and tracers
    original_runner_run_async = Runner.run_async
    original_agent_run_async = BaseAgent.run_async
    original_runners_tracer = runners.tracer
    original_base_llm_flow_tracer = base_llm_flow.tracer
    original_trace_call_llm = base_llm_flow.trace_call_llm
    original_trace_tool_module_tracer = trace_tool_module.tracer
    original_trace_tool_call = trace_tool_module.trace_tool_call
    original_build_attrs = getattr(trace_tool_module, "_build_compaction_attributes", None)
    original_build_result_attrs = getattr(
        trace_tool_module, "_build_compaction_result_attributes", None
    )
    if compaction is not None:
        original_compaction_tracer = compaction.tracer
        original_compaction_build_attrs = compaction._build_compaction_attributes
        original_compaction_build_result_attrs = compaction._build_compaction_result_attributes

    if _ADK_VERSION < (1, 32, 0):
        from google.adk.agents import base_agent

        original_agents_tracer = base_agent.tracer

    # Apply instrumentation
    GoogleADKInstrumentor().instrument()

    # Verify all methods and tracers are wrapped with our implementations
    assert Runner.run_async is not original_runner_run_async
    assert BaseAgent.run_async is not original_agent_run_async
    assert runners.tracer is not original_runners_tracer
    assert base_llm_flow.tracer is not original_base_llm_flow_tracer
    assert base_llm_flow.trace_call_llm is not original_trace_call_llm
    assert trace_tool_module.tracer is not original_trace_tool_module_tracer
    assert trace_tool_module.trace_tool_call is not original_trace_tool_call
    if _ADK_VERSION >= (1, 32, 0):
        assert trace_tool_module._build_compaction_attributes is not original_build_attrs
        assert (
            trace_tool_module._build_compaction_result_attributes is not original_build_result_attrs
        )
    if compaction is not None:
        assert compaction.tracer is not original_compaction_tracer
        assert compaction._build_compaction_attributes is not original_compaction_build_attrs
        assert (
            compaction._build_compaction_result_attributes
            is not original_compaction_build_result_attrs
        )
        # Preloaded case: apps.compaction ends up bound to the *same* wrapped
        # callables telemetry.tracing was source-patched with -- not separate,
        # independently-wrapped copies (that would be the double-wrap bug).
        assert compaction.tracer is trace_tool_module.tracer
        assert (
            compaction._build_compaction_attributes
            is trace_tool_module._build_compaction_attributes
        )
        assert (
            compaction._build_compaction_result_attributes
            is trace_tool_module._build_compaction_result_attributes
        )

    # Verify all tracers are patched with correct types
    assert isinstance(runners.tracer, _PassthroughTracer)
    assert isinstance(base_llm_flow.tracer, OITracer)
    if _ADK_VERSION >= (1, 32, 0):
        # tracing.tracer is the global ADK tracer; on >= 1.32 we wrap it with a
        # selective tracer that emits OI spans for `execute_tool *` and
        # `compact_events *`, and passes through everything else.
        assert isinstance(trace_tool_module.tracer, _SelectiveExecuteToolTracer)
        # functions.tracer is also wrapped to catch `execute_tool (merged)` spans.
        from google.adk.flows.llm_flows import functions as _functions

        assert isinstance(_functions.tracer, _SelectiveExecuteToolTracer)
        if compaction is not None:
            assert isinstance(compaction.tracer, _SelectiveExecuteToolTracer)
    else:
        # functions.tracer is module-local; we substitute our OITracer directly
        assert isinstance(trace_tool_module.tracer, OITracer)

    if _ADK_VERSION < (1, 32, 0):
        assert base_agent.tracer is not original_agents_tracer  # noqa: F821
        assert isinstance(base_agent.tracer, _PassthroughTracer)  # noqa: F821

    # Remove instrumentation
    GoogleADKInstrumentor().uninstrument()

    # Verify all methods and tracers are restored to their original state
    assert Runner.run_async is original_runner_run_async
    assert BaseAgent.run_async is original_agent_run_async
    assert runners.tracer is original_runners_tracer
    assert base_llm_flow.tracer is original_base_llm_flow_tracer
    assert base_llm_flow.trace_call_llm is original_trace_call_llm
    assert trace_tool_module.tracer is original_trace_tool_module_tracer
    assert trace_tool_module.trace_tool_call is original_trace_tool_call
    if _ADK_VERSION >= (1, 32, 0):
        assert trace_tool_module._build_compaction_attributes is original_build_attrs
        assert trace_tool_module._build_compaction_result_attributes is original_build_result_attrs
    if compaction is not None:
        assert compaction.tracer is original_compaction_tracer
        assert compaction._build_compaction_attributes is original_compaction_build_attrs
        assert (
            compaction._build_compaction_result_attributes is original_compaction_build_result_attrs
        )

    if _ADK_VERSION < (1, 32, 0):
        assert base_agent.tracer is original_agents_tracer  # noqa: F821


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


@pytest.mark.skipif(
    _ADK_VERSION < (1, 32, 0),
    reason="apps.compaction tracer/attribute-builder rebinding only runs on google-adk >= 1.32.0",
)
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


@pytest.mark.skipif(
    _ADK_VERSION < (1, 32, 0),
    reason="apps.compaction tracer/attribute-builder rebinding only runs on google-adk >= 1.32.0",
)
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


@pytest.mark.skipif(
    _ADK_VERSION < (1, 32, 0),
    reason="apps.compaction tracer/attribute-builder rebinding only runs on google-adk >= 1.32.0",
)
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


@pytest.mark.skipif(
    _ADK_VERSION < (1, 32, 0),
    reason="apps.compaction tracer/attribute-builder rebinding only runs on google-adk >= 1.32.0",
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
