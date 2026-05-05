# ruff: noqa: E501
import base64
from collections import defaultdict
from secrets import token_hex
from typing import Any, cast

import pytest
from google.adk import Agent, __version__
from google.adk.runners import InMemoryRunner
from google.adk.tools.load_artifacts_tool import load_artifacts_tool as load_artifacts
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_VERSION = cast(tuple[int, int, int], tuple(int(x) for x in __version__.split(".")[:3]))


@pytest.mark.vcr
async def test_google_adk_instrumentor(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def get_weather(city: str) -> dict[str, str]:
        """Retrieves the current weather report for a specified city.

        Args:
            city (str): The name of the city for which to retrieve the weather report.

        Returns:
            dict: status and result or error msg.
        """
        return {
            "status": "success",
            "report": (
                f"The weather in {city} is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }

    agent_name = f"_{token_hex(4)}"
    agent = Agent(
        name=agent_name,
        model="gemini-2.0-flash",
        description="Agent to answer questions using tools.",
        instruction="You must use the available tools to find an answer.",
        tools=[get_weather],
    )

    app_name = token_hex(4)
    user_id = token_hex(4)
    session_id = token_hex(4)
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session_service = runner.session_service
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="What is the weather in New York?")]
        ),
    ):
        ...

    spans = sorted(in_memory_span_exporter.get_finished_spans(), key=lambda s: s.start_time or 0)
    spans_by_name: dict[str, list[ReadableSpan]] = defaultdict(list)
    for span in spans:
        spans_by_name[span.name].append(span)
    assert len(spans) == 5

    invocation_span = spans_by_name[f"invocation [{app_name}]"][0]
    assert invocation_span.status.is_ok
    assert not invocation_span.parent
    invocation_attributes = dict(invocation_span.attributes or {})
    assert invocation_attributes.pop("user.id", None) == user_id
    assert invocation_attributes.pop("session.id", None) == session_id
    assert invocation_attributes.pop("openinference.span.kind", None) == "CHAIN"
    assert invocation_attributes.pop("output.mime_type", None) == "application/json"
    assert invocation_attributes.pop("output.value", None)
    assert invocation_attributes.pop("input.mime_type", None) == "application/json"
    assert invocation_attributes.pop("input.value", None)
    assert not invocation_attributes

    agent_run_span = spans_by_name[f"agent_run [{agent_name}]"][0]
    assert agent_run_span.status.is_ok
    assert agent_run_span.parent
    assert agent_run_span.parent is invocation_span.get_span_context()
    agent_run_attributes = dict(agent_run_span.attributes or {})
    assert agent_run_attributes.pop("user.id", None) == user_id
    assert agent_run_attributes.pop("session.id", None) == session_id
    assert agent_run_attributes.pop("openinference.span.kind", None) == "AGENT"
    assert agent_run_attributes.pop("agent.name", None) == agent_name
    assert agent_run_attributes.pop("output.mime_type", None) == "application/json"
    assert agent_run_attributes.pop("output.value", None)
    # GenAI attributes set by google-adk library
    agent_run_attributes.pop("gen_ai.agent.description", None)
    agent_run_attributes.pop("gen_ai.agent.name", None)
    agent_run_attributes.pop("gen_ai.conversation.id", None)
    agent_run_attributes.pop("gen_ai.operation.name", None)
    agent_run_attributes.pop("gen_ai.agent.version", None)
    assert not agent_run_attributes

    call_llm_span0 = spans_by_name["call_llm"][0]
    assert call_llm_span0.status.is_ok
    assert call_llm_span0.parent
    assert call_llm_span0.parent is agent_run_span.get_span_context()
    call_llm_attributes0 = dict(call_llm_span0.attributes or {})
    assert call_llm_attributes0.pop("user.id", None) == user_id
    assert call_llm_attributes0.pop("session.id", None) == session_id
    assert call_llm_attributes0.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes0.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes0.pop("output.value", None)
    assert call_llm_attributes0.pop("input.mime_type", None) == "application/json"
    assert call_llm_attributes0.pop("input.value", None)
    assert call_llm_attributes0.pop("llm.input_messages.0.message.content", None)
    assert call_llm_attributes0.pop("llm.input_messages.0.message.role", None) == "system"
    assert (
        call_llm_attributes0.pop(
            "llm.input_messages.1.message.contents.0.message_content.text", None
        )
        == "What is the weather in New York?"
    )
    assert (
        call_llm_attributes0.pop(
            "llm.input_messages.1.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes0.pop("llm.input_messages.1.message.role", None) == "user"
    assert call_llm_attributes0.pop("llm.invocation_parameters", None)
    assert call_llm_attributes0.pop("llm.model_name", None) == "gemini-2.0-flash"
    assert call_llm_attributes0.pop("llm.output_messages.0.message.role", None) == "model"
    assert (
        call_llm_attributes0.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"city": "New York"}'
    )
    assert (
        call_llm_attributes0.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name", None
        )
        == "get_weather"
    )
    assert call_llm_attributes0.pop("llm.token_count.completion", None) == 6
    assert call_llm_attributes0.pop("llm.token_count.prompt", None) == 107
    assert call_llm_attributes0.pop("llm.token_count.total", None) == 113
    assert call_llm_attributes0.pop("llm.tools.0.tool.json_schema", None)
    assert call_llm_attributes0.pop("llm.provider", None) == "google"
    assert call_llm_attributes0.pop("gcp.vertex.agent.event_id", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.invocation_id", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.llm_request", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.llm_response", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.session_id", None)
    assert call_llm_attributes0.pop("gen_ai.request.model", None) == "gemini-2.0-flash"
    assert call_llm_attributes0.pop("gen_ai.system", None) == "gcp.vertex.agent"
    if _VERSION >= (1, 5, 0):
        assert call_llm_attributes0.pop("gen_ai.usage.input_tokens", None) == 107
        assert call_llm_attributes0.pop("gen_ai.usage.output_tokens", None) == 6
    call_llm_attributes0.pop("gen_ai.response.finish_reasons", None)
    call_llm_attributes0.pop("gen_ai.agent.name", None)
    call_llm_attributes0.pop("gen_ai.conversation.id", None)
    call_llm_attributes0.pop("gen_ai.operation.name", None)
    call_llm_attributes0.pop("gen_ai.agent.version", None)
    assert not call_llm_attributes0

    tool_span = spans_by_name["execute_tool get_weather"][0]
    assert tool_span.status.is_ok
    assert tool_span.parent
    assert tool_span.parent is call_llm_span0.get_span_context()
    tool_attributes = dict(tool_span.attributes or {})
    assert tool_attributes.pop("user.id", None) == user_id
    assert tool_attributes.pop("session.id", None) == session_id
    assert tool_attributes.pop("openinference.span.kind", None) == "TOOL"
    assert tool_attributes.pop("input.mime_type", None) == "application/json"
    assert tool_attributes.pop("input.value", None) == '{"city": "New York"}'
    assert tool_attributes.pop("output.mime_type", None) == "application/json"
    assert tool_attributes.pop("output.value", None)
    assert (
        tool_attributes.pop("tool.description", None)
        == "Retrieves the current weather report for a specified city.\n\nArgs:\n    city (str): The name of the city for which to retrieve the weather report.\n\nReturns:\n    dict: status and result or error msg."
    )
    assert tool_attributes.pop("tool.name", None) == "get_weather"
    assert tool_attributes.pop("tool.parameters", None) == '{"city": "New York"}'
    assert tool_attributes.pop("gcp.vertex.agent.event_id", None)
    assert tool_attributes.pop("gcp.vertex.agent.llm_request", None) == "{}"
    assert tool_attributes.pop("gcp.vertex.agent.llm_response", None) == "{}"
    assert tool_attributes.pop("gcp.vertex.agent.tool_call_args", None) == '{"city": "New York"}'
    assert (
        tool_attributes.pop("gcp.vertex.agent.tool_response", None)
        == '{"status": "success", "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}'
    )
    # GenAI attributes set by google-adk library
    tool_attributes.pop("gen_ai.operation.name", None)
    tool_attributes.pop("gen_ai.system", None)
    tool_attributes.pop("gen_ai.tool.call.id", None)
    tool_attributes.pop("gen_ai.tool.description", None)
    tool_attributes.pop("gen_ai.tool.name", None)
    tool_attributes.pop("gen_ai.tool.type", None)
    tool_attributes.pop("gen_ai.agent.version", None)
    assert not tool_attributes

    call_llm_span1 = spans_by_name["call_llm"][1]
    assert call_llm_span1.status.is_ok
    assert call_llm_span1.parent
    assert call_llm_span1.parent is agent_run_span.get_span_context()
    call_llm_attributes1 = dict(call_llm_span1.attributes or {})
    assert call_llm_attributes1.pop("user.id", None) == user_id
    assert call_llm_attributes1.pop("session.id", None) == session_id
    assert call_llm_attributes1.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes1.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes1.pop("output.value", None)
    assert call_llm_attributes1.pop("input.mime_type", None) == "application/json"
    assert call_llm_attributes1.pop("input.value", None)
    assert call_llm_attributes1.pop("llm.input_messages.0.message.content", None)
    assert call_llm_attributes1.pop("llm.input_messages.0.message.role", None) == "system"
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.1.message.contents.0.message_content.text", None
        )
        == "What is the weather in New York?"
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.1.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes1.pop("llm.input_messages.1.message.role", None) == "user"
    assert call_llm_attributes1.pop("llm.input_messages.2.message.role", None) == "model"
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"city": "New York"}'
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.name", None
        )
        == "get_weather"
    )
    assert (
        call_llm_attributes1.pop("llm.input_messages.3.message.content", None)
        == '{"status": "success", "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}'
    )
    assert call_llm_attributes1.pop("llm.input_messages.3.message.name", None) == "get_weather"
    assert call_llm_attributes1.pop("llm.input_messages.3.message.role", None) == "tool"
    assert call_llm_attributes1.pop("llm.invocation_parameters", None)
    assert call_llm_attributes1.pop("llm.model_name", None) == "gemini-2.0-flash"
    assert (
        call_llm_attributes1.pop(
            "llm.output_messages.0.message.contents.0.message_content.text", None
        )
        == "OK. The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."
    )
    assert (
        call_llm_attributes1.pop(
            "llm.output_messages.0.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes1.pop("llm.output_messages.0.message.role", None) == "model"
    assert call_llm_attributes1.pop("llm.token_count.completion", None) == 24
    assert call_llm_attributes1.pop("llm.token_count.prompt", None) == 141
    assert call_llm_attributes1.pop("llm.token_count.total", None) == 165
    assert call_llm_attributes1.pop("llm.tools.0.tool.json_schema", None)
    assert call_llm_attributes1.pop("llm.provider", None) == "google"
    assert call_llm_attributes1.pop("gcp.vertex.agent.event_id", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.invocation_id", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.llm_request", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.llm_response", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.session_id", None)
    assert call_llm_attributes1.pop("gen_ai.request.model", None) == "gemini-2.0-flash"
    assert call_llm_attributes1.pop("gen_ai.system", None) == "gcp.vertex.agent"
    if _VERSION >= (1, 5, 0):
        assert call_llm_attributes1.pop("gen_ai.usage.input_tokens", None) == 141
        assert call_llm_attributes1.pop("gen_ai.usage.output_tokens", None) == 24
    call_llm_attributes1.pop("gen_ai.response.finish_reasons", None)
    call_llm_attributes1.pop("gen_ai.agent.name", None)
    call_llm_attributes1.pop("gen_ai.conversation.id", None)
    call_llm_attributes1.pop("gen_ai.operation.name", None)
    call_llm_attributes1.pop("gen_ai.agent.version", None)
    assert not call_llm_attributes1


@pytest.mark.vcr
async def test_google_adk_instrumentor_multi_tool_call(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def get_weather(city: str) -> dict[str, str]:
        """Retrieves the current weather report for a specified city.

        Args:
            city (str): The name of the city for which to retrieve the weather report.

        Returns:
            dict: status and result or error msg.
        """
        return {
            "status": "success",
            "report": (
                f"The weather in {city} is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }

    agent_name = f"_{token_hex(4)}"
    agent = Agent(
        name=agent_name,
        model="gemini-2.5-flash",
        description="Agent to answer questions using tools.",
        instruction="You must use the available tools to find an answer.",
        tools=[get_weather],
    )

    app_name = token_hex(4)
    user_id = token_hex(4)
    session_id = token_hex(4)
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session_service = runner.session_service
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="What is the weather in New York and London?")]
        ),
    ):
        ...

    spans = sorted(in_memory_span_exporter.get_finished_spans(), key=lambda s: s.start_time or 0)
    spans_by_name: dict[str, list[ReadableSpan]] = defaultdict(list)
    for span in spans:
        spans_by_name[span.name].append(span)
    assert len(spans) == 7

    invocation_span = spans_by_name[f"invocation [{app_name}]"][0]
    assert invocation_span.status.is_ok
    assert not invocation_span.parent
    invocation_attributes = dict(invocation_span.attributes or {})
    assert invocation_attributes.pop("user.id", None) == user_id
    assert invocation_attributes.pop("session.id", None) == session_id
    assert invocation_attributes.pop("openinference.span.kind", None) == "CHAIN"
    assert invocation_attributes.pop("output.mime_type", None) == "application/json"
    assert invocation_attributes.pop("output.value", None)
    assert invocation_attributes.pop("input.mime_type", None) == "application/json"
    assert invocation_attributes.pop("input.value", None)
    assert not invocation_attributes

    agent_run_span = spans_by_name[f"agent_run [{agent_name}]"][0]
    assert agent_run_span.status.is_ok
    assert agent_run_span.parent
    assert agent_run_span.parent is invocation_span.get_span_context()
    agent_run_attributes = dict(agent_run_span.attributes or {})
    assert agent_run_attributes.pop("user.id", None) == user_id
    assert agent_run_attributes.pop("session.id", None) == session_id
    assert agent_run_attributes.pop("openinference.span.kind", None) == "AGENT"
    assert agent_run_attributes.pop("agent.name", None) == agent_name
    assert agent_run_attributes.pop("output.mime_type", None) == "application/json"
    assert agent_run_attributes.pop("output.value", None)
    # GenAI attributes set by google-adk library
    agent_run_attributes.pop("gen_ai.agent.description", None)
    agent_run_attributes.pop("gen_ai.agent.name", None)
    agent_run_attributes.pop("gen_ai.conversation.id", None)
    agent_run_attributes.pop("gen_ai.operation.name", None)
    agent_run_attributes.pop("gen_ai.agent.version", None)
    assert not agent_run_attributes

    # call_llm[0]: first LLM call returns get_weather for New York only (with thoughts)
    call_llm_span0 = spans_by_name["call_llm"][0]
    assert call_llm_span0.status.is_ok
    assert call_llm_span0.parent
    assert call_llm_span0.parent is agent_run_span.get_span_context()
    call_llm_attributes0 = dict(call_llm_span0.attributes or {})
    assert call_llm_attributes0.pop("user.id", None) == user_id
    assert call_llm_attributes0.pop("session.id", None) == session_id
    assert call_llm_attributes0.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes0.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes0.pop("output.value", None)
    assert call_llm_attributes0.pop("input.mime_type", None) == "application/json"
    assert call_llm_attributes0.pop("input.value", None)
    assert call_llm_attributes0.pop("llm.input_messages.0.message.content", None)
    assert call_llm_attributes0.pop("llm.input_messages.0.message.role", None) == "system"
    assert (
        call_llm_attributes0.pop(
            "llm.input_messages.1.message.contents.0.message_content.text", None
        )
        == "What is the weather in New York and London?"
    )
    assert (
        call_llm_attributes0.pop(
            "llm.input_messages.1.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes0.pop("llm.input_messages.1.message.role", None) == "user"
    assert call_llm_attributes0.pop("llm.invocation_parameters", None)
    assert call_llm_attributes0.pop("llm.model_name", None) == "gemini-2.5-flash"
    assert call_llm_attributes0.pop("llm.output_messages.0.message.role", None) == "model"
    assert (
        call_llm_attributes0.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"city": "New York"}'
    )
    assert (
        call_llm_attributes0.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name", None
        )
        == "get_weather"
    )
    assert call_llm_attributes0.pop("llm.token_count.completion", None) == 92
    assert call_llm_attributes0.pop("llm.token_count.completion_details.reasoning", None) == 76
    assert call_llm_attributes0.pop("llm.token_count.prompt", None) == 136
    assert call_llm_attributes0.pop("llm.token_count.total", None) == 228
    assert call_llm_attributes0.pop("llm.tools.0.tool.json_schema", None)
    assert call_llm_attributes0.pop("llm.provider", None) == "google"
    assert call_llm_attributes0.pop("gcp.vertex.agent.event_id", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.invocation_id", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.llm_request", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.llm_response", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.session_id", None)
    assert call_llm_attributes0.pop("gen_ai.request.model", None) == "gemini-2.5-flash"
    assert call_llm_attributes0.pop("gen_ai.system", None) == "gcp.vertex.agent"
    if _VERSION >= (1, 5, 0):
        assert call_llm_attributes0.pop("gen_ai.usage.input_tokens", None) == 136
        assert call_llm_attributes0.pop("gen_ai.usage.output_tokens", None) == 16
        assert call_llm_attributes0.pop("gen_ai.usage.experimental.reasoning_tokens", None) == 76
    call_llm_attributes0.pop("gen_ai.response.finish_reasons", None)
    call_llm_attributes0.pop("gen_ai.agent.name", None)
    call_llm_attributes0.pop("gen_ai.conversation.id", None)
    call_llm_attributes0.pop("gen_ai.operation.name", None)
    call_llm_attributes0.pop("gen_ai.agent.version", None)
    assert not call_llm_attributes0

    tool_span = spans_by_name["execute_tool get_weather"][0]
    assert tool_span.status.is_ok
    assert tool_span.parent
    assert tool_span.parent is call_llm_span0.get_span_context()
    tool_attributes = dict(tool_span.attributes or {})
    assert tool_attributes.pop("user.id", None) == user_id
    assert tool_attributes.pop("session.id", None) == session_id
    assert tool_attributes.pop("openinference.span.kind", None) == "TOOL"
    assert tool_attributes.pop("input.mime_type", None) == "application/json"
    assert tool_attributes.pop("input.value", None) == '{"city": "New York"}'
    assert tool_attributes.pop("output.mime_type", None) == "application/json"
    assert tool_attributes.pop("output.value", None)
    assert (
        tool_attributes.pop("tool.description", None)
        == "Retrieves the current weather report for a specified city.\n\nArgs:\n    city (str): The name of the city for which to retrieve the weather report.\n\nReturns:\n    dict: status and result or error msg."
    )
    assert tool_attributes.pop("tool.name", None) == "get_weather"
    assert tool_attributes.pop("tool.parameters", None) == '{"city": "New York"}'
    assert tool_attributes.pop("gcp.vertex.agent.event_id", None)
    assert tool_attributes.pop("gcp.vertex.agent.llm_request", None) == "{}"
    assert tool_attributes.pop("gcp.vertex.agent.llm_response", None) == "{}"
    assert tool_attributes.pop("gcp.vertex.agent.tool_call_args", None) == '{"city": "New York"}'
    assert (
        tool_attributes.pop("gcp.vertex.agent.tool_response", None)
        == '{"status": "success", "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}'
    )
    # GenAI attributes set by google-adk library
    tool_attributes.pop("gen_ai.operation.name", None)
    tool_attributes.pop("gen_ai.system", None)
    tool_attributes.pop("gen_ai.tool.call.id", None)
    tool_attributes.pop("gen_ai.tool.description", None)
    tool_attributes.pop("gen_ai.tool.name", None)
    tool_attributes.pop("gen_ai.tool.type", None)
    tool_attributes.pop("gen_ai.agent.version", None)
    assert not tool_attributes

    # call_llm[1]: second LLM call returns get_weather for London only (no thoughts)
    call_llm_span1 = spans_by_name["call_llm"][1]
    assert call_llm_span1.status.is_ok
    assert call_llm_span1.parent
    assert call_llm_span1.parent is agent_run_span.get_span_context()
    call_llm_attributes1 = dict(call_llm_span1.attributes or {})
    assert call_llm_attributes1.pop("user.id", None) == user_id
    assert call_llm_attributes1.pop("session.id", None) == session_id
    assert call_llm_attributes1.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes1.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes1.pop("output.value", None)
    assert call_llm_attributes1.pop("input.mime_type", None) == "application/json"
    assert call_llm_attributes1.pop("input.value", None)
    assert call_llm_attributes1.pop("llm.input_messages.0.message.content", None)
    assert call_llm_attributes1.pop("llm.input_messages.0.message.role", None) == "system"
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.1.message.contents.0.message_content.text", None
        )
        == "What is the weather in New York and London?"
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.1.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes1.pop("llm.input_messages.1.message.role", None) == "user"
    assert call_llm_attributes1.pop("llm.input_messages.2.message.role", None) == "model"
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"city": "New York"}'
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.name", None
        )
        == "get_weather"
    )
    assert (
        call_llm_attributes1.pop("llm.input_messages.3.message.content", None)
        == '{"status": "success", "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}'
    )
    assert call_llm_attributes1.pop("llm.input_messages.3.message.name", None) == "get_weather"
    assert call_llm_attributes1.pop("llm.input_messages.3.message.role", None) == "tool"
    assert call_llm_attributes1.pop("llm.invocation_parameters", None)
    assert call_llm_attributes1.pop("llm.model_name", None) == "gemini-2.5-flash"
    assert call_llm_attributes1.pop("llm.output_messages.0.message.role", None) == "model"
    assert (
        call_llm_attributes1.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"city": "London"}'
    )
    assert (
        call_llm_attributes1.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name", None
        )
        == "get_weather"
    )
    assert call_llm_attributes1.pop("llm.token_count.completion", None) == 15
    assert call_llm_attributes1.pop("llm.token_count.prompt", None) == 194
    assert call_llm_attributes1.pop("llm.token_count.total", None) == 209
    assert call_llm_attributes1.pop("llm.tools.0.tool.json_schema", None)
    assert call_llm_attributes1.pop("llm.provider", None) == "google"
    assert call_llm_attributes1.pop("gcp.vertex.agent.event_id", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.invocation_id", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.llm_request", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.llm_response", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.session_id", None)
    assert call_llm_attributes1.pop("gen_ai.request.model", None) == "gemini-2.5-flash"
    assert call_llm_attributes1.pop("gen_ai.system", None) == "gcp.vertex.agent"
    if _VERSION >= (1, 5, 0):
        assert call_llm_attributes1.pop("gen_ai.usage.input_tokens", None) == 194
        assert call_llm_attributes1.pop("gen_ai.usage.output_tokens", None) == 15
    call_llm_attributes1.pop("gen_ai.response.finish_reasons", None)
    call_llm_attributes1.pop("gen_ai.agent.name", None)
    call_llm_attributes1.pop("gen_ai.conversation.id", None)
    call_llm_attributes1.pop("gen_ai.operation.name", None)
    call_llm_attributes1.pop("gen_ai.agent.version", None)
    assert not call_llm_attributes1

    tool_span1 = spans_by_name["execute_tool get_weather"][1]
    assert tool_span1.status.is_ok
    assert tool_span1.parent
    assert tool_span1.parent is call_llm_span1.get_span_context()
    tool_attributes1 = dict(tool_span1.attributes or {})
    assert tool_attributes1.pop("user.id", None) == user_id
    assert tool_attributes1.pop("session.id", None) == session_id
    assert tool_attributes1.pop("openinference.span.kind", None) == "TOOL"
    assert tool_attributes1.pop("input.mime_type", None) == "application/json"
    assert tool_attributes1.pop("input.value", None) == '{"city": "London"}'
    assert tool_attributes1.pop("output.mime_type", None) == "application/json"
    assert tool_attributes1.pop("output.value", None)
    assert (
        tool_attributes1.pop("tool.description", None)
        == "Retrieves the current weather report for a specified city.\n\nArgs:\n    city (str): The name of the city for which to retrieve the weather report.\n\nReturns:\n    dict: status and result or error msg."
    )
    assert tool_attributes1.pop("tool.name", None) == "get_weather"
    assert tool_attributes1.pop("tool.parameters", None) == '{"city": "London"}'
    assert tool_attributes1.pop("gcp.vertex.agent.event_id", None)
    assert tool_attributes1.pop("gcp.vertex.agent.llm_request", None) == "{}"
    assert tool_attributes1.pop("gcp.vertex.agent.llm_response", None) == "{}"
    assert tool_attributes1.pop("gcp.vertex.agent.tool_call_args", None) == '{"city": "London"}'
    assert (
        tool_attributes1.pop("gcp.vertex.agent.tool_response", None)
        == '{"status": "success", "report": "The weather in London is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}'
    )
    # GenAI attributes set by google-adk library
    tool_attributes1.pop("gen_ai.operation.name", None)
    tool_attributes1.pop("gen_ai.system", None)
    tool_attributes1.pop("gen_ai.tool.call.id", None)
    tool_attributes1.pop("gen_ai.tool.description", None)
    tool_attributes1.pop("gen_ai.tool.name", None)
    tool_attributes1.pop("gen_ai.tool.type", None)
    tool_attributes1.pop("gen_ai.agent.version", None)
    assert not tool_attributes1

    # call_llm[2]: final LLM call returns text response
    call_llm_span2 = spans_by_name["call_llm"][2]
    assert call_llm_span2.status.is_ok
    assert call_llm_span2.parent
    assert call_llm_span2.parent is agent_run_span.get_span_context()
    call_llm_attributes2 = dict(call_llm_span2.attributes or {})
    assert call_llm_attributes2.pop("user.id", None) == user_id
    assert call_llm_attributes2.pop("session.id", None) == session_id
    assert call_llm_attributes2.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes2.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes2.pop("output.value", None)
    assert call_llm_attributes2.pop("input.mime_type", None) == "application/json"
    assert call_llm_attributes2.pop("input.value", None)
    assert call_llm_attributes2.pop("llm.input_messages.0.message.content", None)
    assert call_llm_attributes2.pop("llm.input_messages.0.message.role", None) == "system"
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.1.message.contents.0.message_content.text", None
        )
        == "What is the weather in New York and London?"
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.1.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes2.pop("llm.input_messages.1.message.role", None) == "user"
    assert call_llm_attributes2.pop("llm.input_messages.2.message.role", None) == "model"
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"city": "New York"}'
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.name", None
        )
        == "get_weather"
    )
    assert (
        call_llm_attributes2.pop("llm.input_messages.3.message.content", None)
        == '{"status": "success", "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}'
    )
    assert call_llm_attributes2.pop("llm.input_messages.3.message.name", None) == "get_weather"
    assert call_llm_attributes2.pop("llm.input_messages.3.message.role", None) == "tool"
    assert call_llm_attributes2.pop("llm.input_messages.4.message.role", None) == "model"
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.4.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"city": "London"}'
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.4.message.tool_calls.0.tool_call.function.name", None
        )
        == "get_weather"
    )
    assert (
        call_llm_attributes2.pop("llm.input_messages.5.message.content", None)
        == '{"status": "success", "report": "The weather in London is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}'
    )
    assert call_llm_attributes2.pop("llm.input_messages.5.message.name", None) == "get_weather"
    assert call_llm_attributes2.pop("llm.input_messages.5.message.role", None) == "tool"
    assert call_llm_attributes2.pop("llm.invocation_parameters", None)
    assert call_llm_attributes2.pop("llm.model_name", None) == "gemini-2.5-flash"
    assert (
        call_llm_attributes2.pop(
            "llm.output_messages.0.message.contents.0.message_content.text", None
        )
        == "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit). The weather in London is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."
    )
    assert (
        call_llm_attributes2.pop(
            "llm.output_messages.0.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes2.pop("llm.output_messages.0.message.role", None) == "model"
    assert call_llm_attributes2.pop("llm.token_count.completion", None) == 43
    assert call_llm_attributes2.pop("llm.token_count.prompt", None) == 250
    assert call_llm_attributes2.pop("llm.token_count.total", None) == 293
    assert call_llm_attributes2.pop("llm.tools.0.tool.json_schema", None)
    assert call_llm_attributes2.pop("llm.provider", None) == "google"
    assert call_llm_attributes2.pop("gcp.vertex.agent.event_id", None)
    assert call_llm_attributes2.pop("gcp.vertex.agent.invocation_id", None)
    assert call_llm_attributes2.pop("gcp.vertex.agent.llm_request", None)
    assert call_llm_attributes2.pop("gcp.vertex.agent.llm_response", None)
    assert call_llm_attributes2.pop("gcp.vertex.agent.session_id", None)
    assert call_llm_attributes2.pop("gen_ai.request.model", None) == "gemini-2.5-flash"
    assert call_llm_attributes2.pop("gen_ai.system", None) == "gcp.vertex.agent"
    if _VERSION >= (1, 5, 0):
        assert call_llm_attributes2.pop("gen_ai.usage.input_tokens", None) == 250
        assert call_llm_attributes2.pop("gen_ai.usage.output_tokens", None) == 43
    call_llm_attributes2.pop("gen_ai.response.finish_reasons", None)
    call_llm_attributes2.pop("gen_ai.agent.name", None)
    call_llm_attributes2.pop("gen_ai.conversation.id", None)
    call_llm_attributes2.pop("gen_ai.operation.name", None)
    call_llm_attributes2.pop("gen_ai.agent.version", None)
    assert not call_llm_attributes2


@pytest.mark.vcr
async def test_google_adk_instrumentor_multi_agent(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def get_weather(city: str) -> dict[str, str]:
        """Retrieves the current weather report for a specified city.

        Args:
            city (str): The name of the city for which to retrieve the weather report.

        Returns:
            dict: status and result or error msg.
        """
        return {
            "status": "success",
            "report": (
                f"The weather in {city} is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }

    def add(a: int, b: int) -> int:
        """Adds two numbers together.

        Args:
            a (int): The first number.
            b (int): The second number.

        Returns:
            int: The sum of the two numbers.
        """
        return a + b

    addition_agent_name = "addition_agent"
    addition_agent = Agent(
        name=addition_agent_name,
        model="gemini-2.0-flash",
        description="Agent to add two numbers together.",
        instruction="You must add two numbers together.",
        tools=[add],
    )

    weather_agent_name = "weather_agent"
    weather_agent = Agent(
        name=weather_agent_name,
        model="gemini-2.0-flash",
        description="Agent to answer questions using tools.",
        instruction="You must use the available tools to find an answer.",
        tools=[get_weather],
    )

    root_agent_name = "root_agent"
    root_agent = Agent(
        name=root_agent_name,
        model="gemini-2.0-flash",
        description="Agent that routes the user's request to the appropriate agent.",
        instruction="You must route the user's request to the appropriate agent.",
        sub_agents=[addition_agent, weather_agent],
    )

    app_name = token_hex(4)
    user_id = token_hex(4)
    session_id = token_hex(4)
    runner = InMemoryRunner(agent=root_agent, app_name=app_name)
    session_service = runner.session_service
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="What is the weather in New York?")]
        ),
    ):
        ...

    spans = sorted(in_memory_span_exporter.get_finished_spans(), key=lambda s: s.start_time or 0)
    spans_by_name: dict[str, list[ReadableSpan]] = defaultdict(list)
    for span in spans:
        spans_by_name[span.name].append(span)
    assert len(spans) == 8

    # 1. invocation span
    invocation_span = spans_by_name[f"invocation [{app_name}]"][0]
    assert invocation_span.status.is_ok
    assert not invocation_span.parent
    invocation_attributes = dict(invocation_span.attributes or {})
    assert invocation_attributes.pop("user.id", None) == user_id
    assert invocation_attributes.pop("session.id", None) == session_id
    assert invocation_attributes.pop("openinference.span.kind", None) == "CHAIN"
    assert invocation_attributes.pop("output.mime_type", None) == "application/json"
    assert invocation_attributes.pop("output.value", None)
    assert invocation_attributes.pop("input.mime_type", None) == "application/json"
    assert invocation_attributes.pop("input.value", None)
    assert not invocation_attributes

    # 2. agent_run [root_agent]
    root_agent_run_span = spans_by_name[f"agent_run [{root_agent_name}]"][0]
    assert root_agent_run_span.status.is_ok
    assert root_agent_run_span.parent
    assert root_agent_run_span.parent is invocation_span.get_span_context()
    root_agent_run_attributes = dict(root_agent_run_span.attributes or {})
    assert root_agent_run_attributes.pop("user.id", None) == user_id
    assert root_agent_run_attributes.pop("session.id", None) == session_id
    assert root_agent_run_attributes.pop("openinference.span.kind", None) == "AGENT"
    assert root_agent_run_attributes.pop("agent.name", None) == root_agent_name
    assert root_agent_run_attributes.pop("output.mime_type", None) == "application/json"
    assert root_agent_run_attributes.pop("output.value", None)
    # GenAI attributes set by google-adk library
    root_agent_run_attributes.pop("gen_ai.agent.description", None)
    root_agent_run_attributes.pop("gen_ai.agent.name", None)
    root_agent_run_attributes.pop("gen_ai.conversation.id", None)
    root_agent_run_attributes.pop("gen_ai.operation.name", None)
    root_agent_run_attributes.pop("gen_ai.agent.version", None)
    assert not root_agent_run_attributes

    # 3. call_llm (root agent - transfer_to_agent)
    call_llm_span0 = spans_by_name["call_llm"][0]
    assert call_llm_span0.status.is_ok
    assert call_llm_span0.parent
    assert call_llm_span0.parent is root_agent_run_span.get_span_context()
    call_llm_attributes0 = dict(call_llm_span0.attributes or {})
    assert call_llm_attributes0.pop("user.id", None) == user_id
    assert call_llm_attributes0.pop("session.id", None) == session_id
    assert call_llm_attributes0.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes0.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes0.pop("output.value", None)
    assert call_llm_attributes0.pop("input.mime_type", None) == "application/json"
    assert call_llm_attributes0.pop("input.value", None)
    assert call_llm_attributes0.pop("llm.input_messages.0.message.content", None)
    assert call_llm_attributes0.pop("llm.input_messages.0.message.role", None) == "system"
    assert (
        call_llm_attributes0.pop(
            "llm.input_messages.1.message.contents.0.message_content.text", None
        )
        == "What is the weather in New York?"
    )
    assert (
        call_llm_attributes0.pop(
            "llm.input_messages.1.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes0.pop("llm.input_messages.1.message.role", None) == "user"
    assert call_llm_attributes0.pop("llm.invocation_parameters", None)
    assert call_llm_attributes0.pop("llm.model_name", None) == "gemini-2.0-flash"
    assert call_llm_attributes0.pop("llm.output_messages.0.message.role", None) == "model"
    assert (
        call_llm_attributes0.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"agent_name": "weather_agent"}'
    )
    assert (
        call_llm_attributes0.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name", None
        )
        == "transfer_to_agent"
    )
    assert call_llm_attributes0.pop("llm.token_count.completion", None) == 11
    assert call_llm_attributes0.pop("llm.token_count.prompt", None) == 323
    assert call_llm_attributes0.pop("llm.token_count.total", None) == 334
    assert call_llm_attributes0.pop("llm.tools.0.tool.json_schema", None)
    assert call_llm_attributes0.pop("llm.provider", None) == "google"
    assert call_llm_attributes0.pop("gcp.vertex.agent.event_id", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.invocation_id", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.llm_request", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.llm_response", None)
    assert call_llm_attributes0.pop("gcp.vertex.agent.session_id", None)
    assert call_llm_attributes0.pop("gen_ai.request.model", None) == "gemini-2.0-flash"
    assert call_llm_attributes0.pop("gen_ai.system", None) == "gcp.vertex.agent"
    if _VERSION >= (1, 5, 0):
        assert call_llm_attributes0.pop("gen_ai.usage.input_tokens", None) == 323
        assert call_llm_attributes0.pop("gen_ai.usage.output_tokens", None) == 11
    call_llm_attributes0.pop("gen_ai.response.finish_reasons", None)
    call_llm_attributes0.pop("gen_ai.agent.name", None)
    call_llm_attributes0.pop("gen_ai.conversation.id", None)
    call_llm_attributes0.pop("gen_ai.operation.name", None)
    call_llm_attributes0.pop("gen_ai.agent.version", None)
    assert not call_llm_attributes0

    # 4. execute_tool transfer_to_agent
    transfer_tool_span = spans_by_name["execute_tool transfer_to_agent"][0]
    assert transfer_tool_span.status.is_ok
    assert transfer_tool_span.parent
    assert transfer_tool_span.parent is call_llm_span0.get_span_context()
    transfer_tool_attributes = dict(transfer_tool_span.attributes or {})
    assert transfer_tool_attributes.pop("user.id", None) == user_id
    assert transfer_tool_attributes.pop("session.id", None) == session_id
    assert transfer_tool_attributes.pop("openinference.span.kind", None) == "TOOL"
    assert transfer_tool_attributes.pop("input.mime_type", None) == "application/json"
    assert transfer_tool_attributes.pop("input.value", None) == '{"agent_name": "weather_agent"}'
    assert transfer_tool_attributes.pop("output.mime_type", None) == "application/json"
    assert transfer_tool_attributes.pop("output.value", None)
    assert transfer_tool_attributes.pop("tool.name", None) == "transfer_to_agent"
    assert transfer_tool_attributes.pop("tool.description", None)
    assert (
        transfer_tool_attributes.pop("tool.parameters", None) == '{"agent_name": "weather_agent"}'
    )
    assert transfer_tool_attributes.pop("gcp.vertex.agent.event_id", None)
    assert transfer_tool_attributes.pop("gcp.vertex.agent.llm_request", None) == "{}"
    assert transfer_tool_attributes.pop("gcp.vertex.agent.llm_response", None) == "{}"
    assert (
        transfer_tool_attributes.pop("gcp.vertex.agent.tool_call_args", None)
        == '{"agent_name": "weather_agent"}'
    )
    assert (
        transfer_tool_attributes.pop("gcp.vertex.agent.tool_response", None) == '{"result": null}'
    )
    # GenAI attributes set by google-adk library
    transfer_tool_attributes.pop("gen_ai.operation.name", None)
    transfer_tool_attributes.pop("gen_ai.system", None)
    transfer_tool_attributes.pop("gen_ai.tool.call.id", None)
    transfer_tool_attributes.pop("gen_ai.tool.description", None)
    transfer_tool_attributes.pop("gen_ai.tool.name", None)
    transfer_tool_attributes.pop("gen_ai.tool.type", None)
    assert not transfer_tool_attributes

    # 5. agent_run [weather_agent]
    weather_agent_run_span = spans_by_name[f"agent_run [{weather_agent_name}]"][0]
    assert weather_agent_run_span.status.is_ok
    assert weather_agent_run_span.parent
    assert weather_agent_run_span.parent is call_llm_span0.get_span_context()
    weather_agent_run_attributes = dict(weather_agent_run_span.attributes or {})
    assert weather_agent_run_attributes.pop("user.id", None) == user_id
    assert weather_agent_run_attributes.pop("session.id", None) == session_id
    assert weather_agent_run_attributes.pop("openinference.span.kind", None) == "AGENT"
    assert weather_agent_run_attributes.pop("agent.name", None) == weather_agent_name
    assert weather_agent_run_attributes.pop("output.mime_type", None) == "application/json"
    assert weather_agent_run_attributes.pop("output.value", None)
    # GenAI attributes set by google-adk library
    weather_agent_run_attributes.pop("gen_ai.agent.description", None)
    weather_agent_run_attributes.pop("gen_ai.agent.name", None)
    weather_agent_run_attributes.pop("gen_ai.conversation.id", None)
    weather_agent_run_attributes.pop("gen_ai.operation.name", None)
    weather_agent_run_attributes.pop("gen_ai.agent.version", None)
    assert not weather_agent_run_attributes

    # 6. call_llm (weather agent - get_weather)
    call_llm_span1 = spans_by_name["call_llm"][1]
    assert call_llm_span1.status.is_ok
    assert call_llm_span1.parent
    assert call_llm_span1.parent is weather_agent_run_span.get_span_context()
    call_llm_attributes1 = dict(call_llm_span1.attributes or {})
    assert call_llm_attributes1.pop("user.id", None) == user_id
    assert call_llm_attributes1.pop("session.id", None) == session_id
    assert call_llm_attributes1.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes1.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes1.pop("output.value", None)
    assert call_llm_attributes1.pop("input.mime_type", None) == "application/json"
    assert call_llm_attributes1.pop("input.value", None)
    assert call_llm_attributes1.pop("llm.input_messages.0.message.content", None)
    assert call_llm_attributes1.pop("llm.input_messages.0.message.role", None) == "system"
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.1.message.contents.0.message_content.text", None
        )
        == "What is the weather in New York?"
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.1.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes1.pop("llm.input_messages.1.message.role", None) == "user"
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.2.message.contents.0.message_content.text", None
        )
        == "For context:"
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.2.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.2.message.contents.1.message_content.text", None
        )
        == "[root_agent] called tool `transfer_to_agent` with parameters: {'agent_name': 'weather_agent'}"
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.2.message.contents.1.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes1.pop("llm.input_messages.2.message.role", None) == "user"
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.3.message.contents.0.message_content.text", None
        )
        == "For context:"
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.3.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.3.message.contents.1.message_content.text", None
        )
        == "[root_agent] `transfer_to_agent` tool returned result: {'result': None}"
    )
    assert (
        call_llm_attributes1.pop(
            "llm.input_messages.3.message.contents.1.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes1.pop("llm.input_messages.3.message.role", None) == "user"
    assert call_llm_attributes1.pop("llm.invocation_parameters", None)
    assert call_llm_attributes1.pop("llm.model_name", None) == "gemini-2.0-flash"
    assert call_llm_attributes1.pop("llm.output_messages.0.message.role", None) == "model"
    assert (
        call_llm_attributes1.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"city": "New York"}'
    )
    assert (
        call_llm_attributes1.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name", None
        )
        == "get_weather"
    )
    assert call_llm_attributes1.pop("llm.token_count.completion", None) == 6
    assert call_llm_attributes1.pop("llm.token_count.prompt", None) == 453
    assert call_llm_attributes1.pop("llm.token_count.total", None) == 459
    assert call_llm_attributes1.pop("llm.tools.0.tool.json_schema", None)
    assert call_llm_attributes1.pop("llm.tools.1.tool.json_schema", None)
    assert call_llm_attributes1.pop("llm.provider", None) == "google"
    assert call_llm_attributes1.pop("gcp.vertex.agent.event_id", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.invocation_id", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.llm_request", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.llm_response", None)
    assert call_llm_attributes1.pop("gcp.vertex.agent.session_id", None)
    assert call_llm_attributes1.pop("gen_ai.request.model", None) == "gemini-2.0-flash"
    assert call_llm_attributes1.pop("gen_ai.system", None) == "gcp.vertex.agent"
    if _VERSION >= (1, 5, 0):
        assert call_llm_attributes1.pop("gen_ai.usage.input_tokens", None) == 453
        assert call_llm_attributes1.pop("gen_ai.usage.output_tokens", None) == 6
    call_llm_attributes1.pop("gen_ai.response.finish_reasons", None)
    call_llm_attributes1.pop("gen_ai.agent.name", None)
    call_llm_attributes1.pop("gen_ai.conversation.id", None)
    call_llm_attributes1.pop("gen_ai.operation.name", None)
    call_llm_attributes1.pop("gen_ai.agent.version", None)
    assert not call_llm_attributes1

    # 7. execute_tool get_weather
    get_weather_tool_span = spans_by_name["execute_tool get_weather"][0]
    assert get_weather_tool_span.status.is_ok
    assert get_weather_tool_span.parent
    assert get_weather_tool_span.parent is call_llm_span1.get_span_context()
    get_weather_tool_attributes = dict(get_weather_tool_span.attributes or {})
    assert get_weather_tool_attributes.pop("user.id", None) == user_id
    assert get_weather_tool_attributes.pop("session.id", None) == session_id
    assert get_weather_tool_attributes.pop("openinference.span.kind", None) == "TOOL"
    assert get_weather_tool_attributes.pop("input.mime_type", None) == "application/json"
    assert get_weather_tool_attributes.pop("input.value", None) == '{"city": "New York"}'
    assert get_weather_tool_attributes.pop("output.mime_type", None) == "application/json"
    assert get_weather_tool_attributes.pop("output.value", None)
    assert (
        get_weather_tool_attributes.pop("tool.description", None)
        == "Retrieves the current weather report for a specified city.\n\nArgs:\n    city (str): The name of the city for which to retrieve the weather report.\n\nReturns:\n    dict: status and result or error msg."
    )
    assert get_weather_tool_attributes.pop("tool.name", None) == "get_weather"
    assert get_weather_tool_attributes.pop("tool.parameters", None) == '{"city": "New York"}'
    assert get_weather_tool_attributes.pop("gcp.vertex.agent.event_id", None)
    assert get_weather_tool_attributes.pop("gcp.vertex.agent.llm_request", None) == "{}"
    assert get_weather_tool_attributes.pop("gcp.vertex.agent.llm_response", None) == "{}"
    assert (
        get_weather_tool_attributes.pop("gcp.vertex.agent.tool_call_args", None)
        == '{"city": "New York"}'
    )
    assert (
        get_weather_tool_attributes.pop("gcp.vertex.agent.tool_response", None)
        == '{"status": "success", "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}'
    )
    # GenAI attributes set by google-adk library
    get_weather_tool_attributes.pop("gen_ai.operation.name", None)
    get_weather_tool_attributes.pop("gen_ai.system", None)
    get_weather_tool_attributes.pop("gen_ai.tool.call.id", None)
    get_weather_tool_attributes.pop("gen_ai.tool.description", None)
    get_weather_tool_attributes.pop("gen_ai.tool.name", None)
    get_weather_tool_attributes.pop("gen_ai.tool.type", None)
    assert not get_weather_tool_attributes

    # 8. call_llm (weather agent - final response)
    call_llm_span2 = spans_by_name["call_llm"][2]
    assert call_llm_span2.status.is_ok
    assert call_llm_span2.parent
    assert call_llm_span2.parent is weather_agent_run_span.get_span_context()
    call_llm_attributes2 = dict(call_llm_span2.attributes or {})
    assert call_llm_attributes2.pop("user.id", None) == user_id
    assert call_llm_attributes2.pop("session.id", None) == session_id
    assert call_llm_attributes2.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes2.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes2.pop("output.value", None)
    assert call_llm_attributes2.pop("input.mime_type", None) == "application/json"
    assert call_llm_attributes2.pop("input.value", None)
    assert call_llm_attributes2.pop("llm.input_messages.0.message.content", None)
    assert call_llm_attributes2.pop("llm.input_messages.0.message.role", None) == "system"
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.1.message.contents.0.message_content.text", None
        )
        == "What is the weather in New York?"
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.1.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes2.pop("llm.input_messages.1.message.role", None) == "user"
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.2.message.contents.0.message_content.text", None
        )
        == "For context:"
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.2.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.2.message.contents.1.message_content.text", None
        )
        == "[root_agent] called tool `transfer_to_agent` with parameters: {'agent_name': 'weather_agent'}"
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.2.message.contents.1.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes2.pop("llm.input_messages.2.message.role", None) == "user"
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.3.message.contents.0.message_content.text", None
        )
        == "For context:"
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.3.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.3.message.contents.1.message_content.text", None
        )
        == "[root_agent] `transfer_to_agent` tool returned result: {'result': None}"
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.3.message.contents.1.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes2.pop("llm.input_messages.3.message.role", None) == "user"
    assert call_llm_attributes2.pop("llm.input_messages.4.message.role", None) == "model"
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.4.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"city": "New York"}'
    )
    assert (
        call_llm_attributes2.pop(
            "llm.input_messages.4.message.tool_calls.0.tool_call.function.name", None
        )
        == "get_weather"
    )
    assert call_llm_attributes2.pop("llm.input_messages.5.message.role", None) == "tool"
    assert call_llm_attributes2.pop("llm.input_messages.5.message.name", None) == "get_weather"
    assert (
        call_llm_attributes2.pop("llm.input_messages.5.message.content", None)
        == '{"status": "success", "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}'
    )
    assert call_llm_attributes2.pop("llm.invocation_parameters", None)
    assert call_llm_attributes2.pop("llm.model_name", None) == "gemini-2.0-flash"
    assert (
        call_llm_attributes2.pop(
            "llm.output_messages.0.message.contents.0.message_content.text", None
        )
        == "OK. The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit).\n"
    )
    assert (
        call_llm_attributes2.pop(
            "llm.output_messages.0.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes2.pop("llm.output_messages.0.message.role", None) == "model"
    assert call_llm_attributes2.pop("llm.token_count.completion", None) == 25
    assert call_llm_attributes2.pop("llm.token_count.prompt", None) == 487
    assert call_llm_attributes2.pop("llm.token_count.total", None) == 512
    assert call_llm_attributes2.pop("llm.tools.0.tool.json_schema", None)
    assert call_llm_attributes2.pop("llm.tools.1.tool.json_schema", None)
    assert call_llm_attributes2.pop("llm.provider", None) == "google"
    assert call_llm_attributes2.pop("gcp.vertex.agent.event_id", None)
    assert call_llm_attributes2.pop("gcp.vertex.agent.invocation_id", None)
    assert call_llm_attributes2.pop("gcp.vertex.agent.llm_request", None)
    assert call_llm_attributes2.pop("gcp.vertex.agent.llm_response", None)
    assert call_llm_attributes2.pop("gcp.vertex.agent.session_id", None)
    assert call_llm_attributes2.pop("gen_ai.request.model", None) == "gemini-2.0-flash"
    assert call_llm_attributes2.pop("gen_ai.system", None) == "gcp.vertex.agent"
    if _VERSION >= (1, 5, 0):
        assert call_llm_attributes2.pop("gen_ai.usage.input_tokens", None) == 487
        assert call_llm_attributes2.pop("gen_ai.usage.output_tokens", None) == 25
    call_llm_attributes2.pop("gen_ai.response.finish_reasons", None)
    call_llm_attributes2.pop("gen_ai.agent.name", None)
    call_llm_attributes2.pop("gen_ai.conversation.id", None)
    call_llm_attributes2.pop("gen_ai.operation.name", None)
    call_llm_attributes2.pop("gen_ai.agent.version", None)
    assert not call_llm_attributes2


@pytest.mark.vcr
async def test_google_adk_instrumentor_image_artifacts(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async def load_remote_image(file_path: str, tool_context: ToolContext) -> str:
        """
        Reads a local image file and registers it as an ADK artifact.
        Args:
            tool_context:
            file_path: Remote location of file.
        """
        image_bytes = base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII="
        )
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        filename = "sample.png"
        await tool_context.save_artifact(filename=filename, artifact=image_part)
        return f"Success! Image '{filename}' is now available as an artifact."

    agent_name = "poet_agent"
    agent = Agent(
        name=agent_name,
        model="gemini-2.0-flash",
        instruction=(
            "You are a creative poet and visual analyst. "
            "1. First, use 'load_local_image_artifact' to get the file into the system. "
            "2. Then, use 'load_artifacts' to see the image content. "
            "3. Describe the image in detail. "
            "4. Write a beautiful poem based on that description."
        ),
        tools=[load_remote_image, load_artifacts],
    )

    app_name = token_hex(4)
    user_id = token_hex(4)
    session_id = token_hex(4)

    runner = InMemoryRunner(
        agent=agent,
        app_name=app_name,
    )
    session_service = runner.session_service
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    local_path = "sample.png"
    user_query = f"Please process the image at '{local_path}', describe it."
    content = types.Content(role="user", parts=[types.Part(text=user_query)])
    async for _ in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        ...

    spans = sorted(in_memory_span_exporter.get_finished_spans(), key=lambda s: s.start_time or 0)
    spans_by_name: dict[str, list[ReadableSpan]] = defaultdict(list)
    for span in spans:
        spans_by_name[span.name].append(span)
    assert len(spans) == 7

    invocation_span = spans_by_name[f"invocation [{app_name}]"][0]
    assert invocation_span.status.is_ok
    assert not invocation_span.parent
    invocation_attributes = dict(invocation_span.attributes or {})
    assert invocation_attributes.pop("user.id", None) == user_id
    assert invocation_attributes.pop("session.id", None) == session_id
    assert invocation_attributes.pop("openinference.span.kind", None) == "CHAIN"
    assert invocation_attributes.pop("output.mime_type", None) == "application/json"
    assert invocation_attributes.pop("output.value", None)
    assert invocation_attributes.pop("input.mime_type", None) == "application/json"
    assert invocation_attributes.pop("input.value", None)
    assert not invocation_attributes

    agent_run_span = spans_by_name[f"agent_run [{agent_name}]"][0]
    assert agent_run_span.status.is_ok
    assert agent_run_span.parent
    assert agent_run_span.parent is invocation_span.get_span_context()
    agent_run_attributes = dict(agent_run_span.attributes or {})
    assert agent_run_attributes.pop("user.id", None) == user_id
    assert agent_run_attributes.pop("session.id", None) == session_id
    assert agent_run_attributes.pop("openinference.span.kind", None) == "AGENT"
    assert agent_run_attributes.pop("agent.name", None) == agent_name
    assert agent_run_attributes.pop("output.mime_type", None) == "application/json"
    assert agent_run_attributes.pop("output.value", None)
    # GenAI attributes set by google-adk library
    agent_run_attributes.pop("gen_ai.agent.description", None)
    agent_run_attributes.pop("gen_ai.agent.name", None)
    agent_run_attributes.pop("gen_ai.conversation.id", None)
    agent_run_attributes.pop("gen_ai.operation.name", None)
    agent_run_attributes.pop("gen_ai.agent.version", None)
    assert not agent_run_attributes

    call_llm_span = spans_by_name["call_llm"][-1]
    assert call_llm_span.status.is_ok
    assert call_llm_span.parent
    assert call_llm_span.parent is agent_run_span.get_span_context()
    call_llm_attributes = dict(call_llm_span.attributes or {})
    assert call_llm_attributes.pop("user.id", None) == user_id
    assert call_llm_attributes.pop("session.id", None) == session_id
    assert call_llm_attributes.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes.pop("output.value", None)
    assert call_llm_attributes.pop("input.mime_type", None) == "application/json"
    assert call_llm_attributes.pop("input.value", None)
    assert call_llm_attributes.pop("llm.input_messages.0.message.content", None)
    assert call_llm_attributes.pop("llm.input_messages.0.message.role", None) == "system"
    assert (
        call_llm_attributes.pop(
            "llm.input_messages.1.message.contents.0.message_content.text", None
        )
        == "Please process the image at 'sample.png', describe it."
    )
    assert (
        call_llm_attributes.pop(
            "llm.input_messages.1.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes.pop("llm.input_messages.1.message.role", None) == "user"

    assert call_llm_attributes.pop(
        "llm.input_messages.2.message.tool_calls.0.tool_call.function.name", "load_remote_image"
    )
    assert call_llm_attributes.pop(
        "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments",
        '{"file_url": "https://picsum.photos/200/300"}',
    )
    assert call_llm_attributes.pop("llm.input_messages.2.message.role", None) == "model"

    assert call_llm_attributes.pop("llm.input_messages.3.message.name", None) == "load_remote_image"
    assert call_llm_attributes.pop("llm.input_messages.3.message.content", None)
    assert call_llm_attributes.pop("llm.input_messages.3.message.role", None) == "tool"

    assert call_llm_attributes.pop("llm.input_messages.4.message.role", None) == "model"
    assert (
        call_llm_attributes.pop(
            "llm.input_messages.4.message.tool_calls.0.tool_call.function.name", None
        )
        == "load_artifacts"
    )
    assert (
        call_llm_attributes.pop(
            "llm.input_messages.4.message.tool_calls.0.tool_call.function.arguments", None
        )
        == '{"artifact_names": ["sample.png"]}'
    )

    assert call_llm_attributes.pop("llm.input_messages.5.message.name", None) == "load_artifacts"
    assert "artifact_names" in str(
        call_llm_attributes.pop("llm.input_messages.5.message.content", "")
    )
    assert call_llm_attributes.pop("llm.input_messages.5.message.role", None) == "tool"
    assert call_llm_attributes.pop("llm.input_messages.6.message.role", None) == "user"
    assert (
        call_llm_attributes.pop(
            "llm.input_messages.6.message.contents.0.message_content.text", None
        )
        == "Artifact sample.png is:"
    )
    assert (
        call_llm_attributes.pop(
            "llm.input_messages.6.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes.pop(
        "llm.input_messages.6.message.contents.1.message_content.image.image.url", None
    )
    assert (
        call_llm_attributes.pop(
            "llm.input_messages.6.message.contents.1.message_content.type", None
        )
        == "image"
    )

    assert call_llm_attributes.pop("llm.invocation_parameters", None)
    assert call_llm_attributes.pop("llm.model_name", None) == "gemini-2.0-flash"
    assert call_llm_attributes.pop("llm.output_messages.0.message.role", None) == "model"
    assert call_llm_attributes.pop(
        "llm.output_messages.0.message.contents.0.message_content.text", None
    ) == (
        "\nThe image is a simple, blurred cross shape. Two squares are white, and the other two are a light gray. "
        "The blurring makes the edges soft and indistinct.\n\n"
        "Here's a poem inspired by it:\n\n"
        "A muted cross, a gentle sign,\n"
        "Where light and shadow intertwine.\n"
        "Two squares of white, a pristine gleam,\n"
        "Two shades of gray, a fading dream.\n\n"
        "No sharp divide, no stark contrast,\n"
        "But soft transitions, fading fast.\n"
        "A quiet symbol, undefined,\n"
        "A peaceful solace for the mind.\n"
    )
    assert (
        call_llm_attributes.pop(
            "llm.output_messages.0.message.contents.0.message_content.type", None
        )
        == "text"
    )
    assert call_llm_attributes.pop("llm.token_count.completion", None) == 115
    assert call_llm_attributes.pop("llm.token_count.prompt", None) == 608
    assert call_llm_attributes.pop("llm.token_count.total", None) == 723
    assert call_llm_attributes.pop("llm.tools.0.tool.json_schema", None)
    assert call_llm_attributes.pop("llm.provider", None) == "google"
    assert call_llm_attributes.pop("gcp.vertex.agent.event_id", None)
    assert call_llm_attributes.pop("gcp.vertex.agent.invocation_id", None)
    assert call_llm_attributes.pop("gcp.vertex.agent.llm_request", None)
    assert call_llm_attributes.pop("gcp.vertex.agent.llm_response", None)
    assert call_llm_attributes.pop("gcp.vertex.agent.session_id", None)
    assert call_llm_attributes.pop("gen_ai.request.model", None) == "gemini-2.0-flash"
    assert call_llm_attributes.pop("gen_ai.system", None) == "gcp.vertex.agent"
    if _VERSION >= (1, 5, 0):
        assert call_llm_attributes.pop("gen_ai.usage.input_tokens", None) == 608
        assert call_llm_attributes.pop("gen_ai.usage.output_tokens", None) == 115
    call_llm_attributes.pop("gen_ai.response.finish_reasons", None)
    call_llm_attributes.pop("llm.tools.1.tool.json_schema", None)
    call_llm_attributes.pop("gen_ai.agent.name", None)
    call_llm_attributes.pop("gen_ai.conversation.id", None)
    call_llm_attributes.pop("gen_ai.operation.name", None)
    call_llm_attributes.pop("gen_ai.agent.version", None)
    assert not call_llm_attributes

    tool_span = spans_by_name["execute_tool load_remote_image"][0]
    assert tool_span.status.is_ok
    assert tool_span.parent
    tool_attributes = dict(tool_span.attributes or {})
    assert tool_attributes.pop("user.id", None) == user_id
    assert tool_attributes.pop("session.id", None) == session_id
    assert tool_attributes.pop("openinference.span.kind", None) == "TOOL"
    assert tool_attributes.pop("input.mime_type", None) == "application/json"
    assert tool_attributes.pop("input.value", None) == '{"file_path": "sample.png"}'
    assert tool_attributes.pop("output.mime_type", None) == "application/json"
    assert tool_attributes.pop("output.value", None)
    assert tool_attributes.pop("tool.description", None)
    assert tool_attributes.pop("tool.name", None) == "load_remote_image"
    assert tool_attributes.pop("tool.parameters", None) == '{"file_path": "sample.png"}'
    assert tool_attributes.pop("gcp.vertex.agent.event_id", None)
    assert tool_attributes.pop("gcp.vertex.agent.llm_request", None) == "{}"
    assert tool_attributes.pop("gcp.vertex.agent.llm_response", None) == "{}"
    assert (
        tool_attributes.pop("gcp.vertex.agent.tool_call_args", None)
        == '{"file_path": "sample.png"}'
    )
    assert tool_attributes.pop("gcp.vertex.agent.tool_response", None)
    # GenAI attributes set by google-adk library
    tool_attributes.pop("gen_ai.operation.name", None)
    tool_attributes.pop("gen_ai.system", None)
    tool_attributes.pop("gen_ai.tool.call.id", None)
    tool_attributes.pop("gen_ai.tool.description", None)
    tool_attributes.pop("gen_ai.tool.name", None)
    tool_attributes.pop("gen_ai.tool.type", None)
    assert not tool_attributes

    tool_span1 = spans_by_name["execute_tool load_artifacts"][0]
    assert tool_span1.status.is_ok
    assert tool_span1.parent
    tool_attributes1 = dict(tool_span1.attributes or {})
    assert tool_attributes1.pop("user.id", None) == user_id
    assert tool_attributes1.pop("session.id", None) == session_id
    assert tool_attributes1.pop("openinference.span.kind", None) == "TOOL"
    assert tool_attributes1.pop("input.mime_type", None) == "application/json"
    assert tool_attributes1.pop("input.value", None) == '{"artifact_names": ["sample.png"]}'
    assert tool_attributes1.pop("output.mime_type", None) == "application/json"
    assert tool_attributes1.pop("output.value", None)
    assert tool_attributes1.pop("tool.description", None)
    assert tool_attributes1.pop("tool.name", None) == "load_artifacts"
    assert tool_attributes1.pop("tool.parameters", None) == '{"artifact_names": ["sample.png"]}'
    assert tool_attributes1.pop("gcp.vertex.agent.event_id", None)
    assert tool_attributes1.pop("gcp.vertex.agent.llm_request", None) == "{}"
    assert tool_attributes1.pop("gcp.vertex.agent.llm_response", None) == "{}"
    assert (
        tool_attributes1.pop("gcp.vertex.agent.tool_call_args", None)
        == '{"artifact_names": ["sample.png"]}'
    )
    assert tool_attributes1.pop("gcp.vertex.agent.tool_response", None)
    # GenAI attributes set by google-adk library
    tool_attributes1.pop("gen_ai.operation.name", None)
    tool_attributes1.pop("gen_ai.system", None)
    tool_attributes1.pop("gen_ai.tool.call.id", None)
    tool_attributes1.pop("gen_ai.tool.description", None)
    tool_attributes1.pop("gen_ai.tool.name", None)
    tool_attributes1.pop("gen_ai.tool.type", None)
    assert not tool_attributes1


async def test_google_adk_instrumentor_parallel_tool_calls(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Exercise the parallel-call path that emits an `execute_tool (merged)` span.

    This path is gated on `len(function_response_events) > 1` in
    `google.adk.flows.llm_flows.functions`, which uses the module-local `tracer`
    binding. We use a stubbed `BaseLlm` to deterministically produce two parallel
    function calls in a single response and assert the merged span is attributed
    to our OITracer (not the original ADK tracer).
    """
    from typing import AsyncGenerator

    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse

    def get_weather(city: str) -> dict[str, str]:
        return {
            "status": "success",
            "report": f"The weather in {city} is sunny.",
        }

    parallel_calls_response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="call_1",
                        name="get_weather",
                        args={"city": "New York"},
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="call_2",
                        name="get_weather",
                        args={"city": "London"},
                    )
                ),
            ],
        )
    )
    final_text_response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="Both cities are sunny.")],
        )
    )

    class _StubLlm(BaseLlm):
        model: str = "stub"
        _index: int = 0

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            responses = [parallel_calls_response, final_text_response]
            response = responses[min(self._index, len(responses) - 1)]
            object.__setattr__(self, "_index", self._index + 1)
            yield response

    agent_name = f"_{token_hex(4)}"
    agent = Agent(
        name=agent_name,
        model=_StubLlm(),
        description="Agent that calls tools in parallel.",
        instruction="Call get_weather for both cities in parallel.",
        tools=[get_weather],
    )

    app_name = token_hex(4)
    user_id = token_hex(4)
    session_id = token_hex(4)
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session_service = runner.session_service
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What is the weather in New York and London?")],
        ),
    ):
        ...

    spans = sorted(in_memory_span_exporter.get_finished_spans(), key=lambda s: s.start_time or 0)
    spans_by_name: dict[str, list[ReadableSpan]] = defaultdict(list)
    for span in spans:
        spans_by_name[span.name].append(span)

    # Per-tool spans for both parallel calls
    assert len(spans_by_name["execute_tool get_weather"]) == 2
    for tool_span in spans_by_name["execute_tool get_weather"]:
        assert tool_span.instrumentation_scope is not None
        assert tool_span.instrumentation_scope.name == "openinference.instrumentation.google_adk"
        assert (tool_span.attributes or {}).get("openinference.span.kind") == "TOOL"

    # Merged span for the parallel batch — attributed to our OITracer, not ADK's
    merged_spans = spans_by_name["execute_tool (merged)"]
    assert len(merged_spans) == 1
    merged_span = merged_spans[0]
    assert merged_span.instrumentation_scope is not None
    assert merged_span.instrumentation_scope.name == "openinference.instrumentation.google_adk"
