# pyright: reportPrivateImportUsage=false
# mypy: disable-error-code="attr-defined"

"""Tests for Google ADK instrumentation patching and unpatching functionality.

This test verifies that the GoogleADKInstrumentor correctly patches and unpatchs:
- Runner.run_async method
- BaseAgent.run_async method
- All tracers (runners, agents, llm_flows, functions)
- trace_call_llm and trace_tool_call methods
"""

from openinference.instrumentation import OITracer
from openinference.instrumentation.google_adk import GoogleADKInstrumentor, _PassthroughTracer


def test_instrumentation_patching() -> None:
    """Test that all instrumentation patching and unpatching works correctly.

    This test verifies that:
    1. All methods and tracers are properly wrapped during instrumentation
    2. All tracers are replaced with appropriate types (_PassthroughTracer or OITracer)
    3. All methods and tracers are restored to their original state after uninstrumentation
    """
    # Import all necessary modules
    from google.adk import runners
    from google.adk.agents import BaseAgent, base_agent
    from google.adk.flows.llm_flows import base_llm_flow, functions
    from google.adk.runners import Runner

    # Store original state of all methods and tracers
    original_runner_run_async = Runner.run_async
    original_agent_run_async = BaseAgent.run_async
    original_runners_tracer = runners.tracer
    original_agents_tracer = base_agent.tracer
    original_base_llm_flow_tracer = base_llm_flow.tracer
    original_trace_call_llm = base_llm_flow.trace_call_llm
    original_functions_tracer = functions.tracer
    original_trace_tool_call = functions.trace_tool_call

    # Apply instrumentation
    GoogleADKInstrumentor().instrument()

    # Verify all methods and tracers are wrapped with our implementations
    assert Runner.run_async is not original_runner_run_async
    assert BaseAgent.run_async is not original_agent_run_async
    assert runners.tracer is not original_runners_tracer
    assert base_agent.tracer is not original_agents_tracer
    assert base_llm_flow.tracer is not original_base_llm_flow_tracer
    assert base_llm_flow.trace_call_llm is not original_trace_call_llm
    assert functions.tracer is not original_functions_tracer
    assert functions.trace_tool_call is not original_trace_tool_call

    # Verify all tracers are patched with correct types
    assert isinstance(runners.tracer, _PassthroughTracer)
    assert isinstance(base_agent.tracer, _PassthroughTracer)
    assert isinstance(base_llm_flow.tracer, OITracer)
    assert isinstance(functions.tracer, OITracer)

    # Remove instrumentation
    GoogleADKInstrumentor().uninstrument()

    # Verify all methods and tracers are restored to their original state
    assert Runner.run_async is original_runner_run_async
    assert BaseAgent.run_async is original_agent_run_async
    assert runners.tracer is original_runners_tracer
    assert base_agent.tracer is original_agents_tracer
    assert base_llm_flow.tracer is original_base_llm_flow_tracer
    assert base_llm_flow.trace_call_llm is original_trace_call_llm
    assert functions.tracer is original_functions_tracer
    assert functions.trace_tool_call is original_trace_tool_call
