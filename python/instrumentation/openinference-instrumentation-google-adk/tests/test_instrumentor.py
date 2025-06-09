from collections import defaultdict
from secrets import token_hex
from typing import Any

import pytest
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr(
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
    decode_compressed_response=True,
)
async def test_google_adk_instrumentor(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def get_weather(city: str) -> dict:
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

    agent_name = "_" + token_hex(4)
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

    spans = in_memory_span_exporter.get_finished_spans()
    spans = sorted(spans, key=lambda s: s.start_time or 0)
    spans_by_name: dict[str, list[ReadableSpan]] = defaultdict(list)
    for span in spans:
        spans_by_name[span.name].append(span)
    assert len(spans) == 5

    invocation_span = spans_by_name[f"invocation [{app_name}]"][0]
    assert not invocation_span.parent
    invocation_attributes = dict(invocation_span.attributes or {})
    assert invocation_attributes.pop("user.id", None) == user_id
    assert invocation_attributes.pop("session.id", None) == session_id
    assert invocation_attributes.pop("openinference.span.kind", None) == "CHAIN"
    assert invocation_attributes.pop("input.mime_type", None) == "application/json"
    assert invocation_attributes.pop("input.value", None)
    assert not invocation_attributes

    agent_run_span = spans_by_name[f"agent_run [{agent_name}]"][0]
    assert agent_run_span.parent
    assert agent_run_span.parent is invocation_span.get_span_context()
    agent_run_attributes = dict(agent_run_span.attributes or {})
    assert agent_run_attributes.pop("user.id", None) == user_id
    assert agent_run_attributes.pop("session.id", None) == session_id
    assert agent_run_attributes.pop("openinference.span.kind", None) == "AGENT"
    assert agent_run_attributes.pop("output.mime_type", None) == "application/json"
    assert agent_run_attributes.pop("output.value", None)
    assert not agent_run_attributes

    call_llm_span0 = spans_by_name["call_llm"][0]
    assert call_llm_span0.parent
    assert call_llm_span0.parent is agent_run_span.get_span_context()
    call_llm_attributes0 = dict(call_llm_span0.attributes or {})
    assert call_llm_attributes0.pop("user.id", None) == user_id
    assert call_llm_attributes0.pop("session.id", None) == session_id
    assert call_llm_attributes0.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes0.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes0.pop("output.value", None)

    tool_span = spans_by_name["execute_tool get_weather"][0]
    assert tool_span.parent
    assert tool_span.parent is call_llm_span0.get_span_context()
    tool_attributes = dict(tool_span.attributes or {})
    assert tool_attributes.pop("user.id", None) == user_id
    assert tool_attributes.pop("session.id", None) == session_id
    assert tool_attributes.pop("openinference.span.kind", None) == "TOOL"

    call_llm_span1 = spans_by_name["call_llm"][1]
    assert call_llm_span1.parent
    assert call_llm_span1.parent is agent_run_span.get_span_context()
    call_llm_attributes1 = dict(call_llm_span1.attributes or {})
    assert call_llm_attributes1.pop("user.id", None) == user_id
    assert call_llm_attributes1.pop("session.id", None) == session_id
    assert call_llm_attributes1.pop("openinference.span.kind", None) == "LLM"
    assert call_llm_attributes1.pop("output.mime_type", None) == "application/json"
    assert call_llm_attributes1.pop("output.value", None)
