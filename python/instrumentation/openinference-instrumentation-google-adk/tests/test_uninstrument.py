# pyright: reportPrivateImportUsage=false
# mypy: disable-error-code="attr-defined"

"""Tests for Google ADK instrumentation patching and unpatching functionality.

This test verifies that the GoogleADKInstrumentor correctly patches and unpatchs:
- Runner.run_async method
- BaseAgent.run_async method
- All tracers (runners, agents, llm_flows, functions/telemetry.tracing)
- trace_call_llm and trace_tool_call methods
"""

from typing import cast

from google.adk import __version__ as _ADK_VERSION_STR

from openinference.instrumentation import OITracer
from openinference.instrumentation.google_adk import GoogleADKInstrumentor, _PassthroughTracer

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
    if _ADK_VERSION >= (1, 32, 0):
        from google.adk.telemetry import tracing as trace_tool_module
    else:
        from google.adk.flows.llm_flows import (
            functions as trace_tool_module,  # type: ignore[no-redef]
        )

    # Store original state of all methods and tracers
    original_runner_run_async = Runner.run_async
    original_agent_run_async = BaseAgent.run_async
    original_runners_tracer = runners.tracer
    original_base_llm_flow_tracer = base_llm_flow.tracer
    original_trace_call_llm = base_llm_flow.trace_call_llm
    original_trace_tool_module_tracer = trace_tool_module.tracer
    original_trace_tool_call = trace_tool_module.trace_tool_call

    if _ADK_VERSION < (1, 32, 0):
        from google.adk.agents import base_agent  # type: ignore[attr-defined]

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

    # Verify all tracers are patched with correct types
    assert isinstance(runners.tracer, _PassthroughTracer)
    assert isinstance(base_llm_flow.tracer, OITracer)
    if _ADK_VERSION >= (1, 32, 0):
        # tracing.tracer is the global ADK tracer suppressed via _PassthroughTracer
        assert isinstance(trace_tool_module.tracer, _PassthroughTracer)
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

    if _ADK_VERSION < (1, 32, 0):
        assert base_agent.tracer is original_agents_tracer  # noqa: F821
