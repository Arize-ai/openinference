import json
from typing import Any, Dict, Generator, Mapping, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import OITracer
from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor


def remove_all_vcr_request_headers(request: Any) -> Any:
    """
    Removes all request headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_request_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    request.headers.clear()
    return request


def remove_all_vcr_response_headers(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes all response headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_response_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    response["headers"] = {}
    return response


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)

    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    AutogenAgentChatInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AutogenAgentChatInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(
            group="opentelemetry_instrumentor", name="autogen_agentchat"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, AutogenAgentChatInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self) -> None:
        assert isinstance(AutogenAgentChatInstrumentor()._tracer, OITracer)


class TestAssistantAgent:
    @pytest.mark.asyncio
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_response_headers,
        before_record_request=remove_all_vcr_request_headers,
    )
    async def test_agent_run(
        self,
        tracer_provider: trace_api.TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        model_client = OpenAIChatCompletionClient(
            model="gpt-3.5-turbo",
            api_key="fake-key",
        )

        # Define a simple function tool that the agent can use
        def get_weather(city: str) -> str:
            """Get the weather for a given city."""
            return f"The weather in {city} is 73 degrees and Sunny."

        # Define an AssistantAgent with the model, tool, system message, and reflection enabled
        agent = AssistantAgent(
            name="weather_agent",
            model_client=model_client,
            tools=[get_weather],
            system_message="You are a helpful assistant that can check the weather.",
            reflect_on_tool_use=True,
            model_client_stream=True,
        )

        # Run the agent and stream the messages to the console
        _ = await agent.run(task="What is the weather in New York?")
        await model_client.close()

        # Verify that spans were created
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 2
        final_span = spans[-1]
        assert final_span.status.is_ok

        attributes = dict(cast(Mapping[str, AttributeValue], final_span.attributes))
        assert attributes["agent_name"] == "weather_agent"
        assert (
            attributes["agent_description"]
            == "An agent that provides assistance with ability to use tools."
        )
        assert json.loads(attributes["input.value"]) == {"task": "What is the weather in New York?"}
        assert attributes["openinference.span.kind"] == "AGENT"
        assert attributes["output.mime_type"] == "application/json"

        expected_output = {
            "messages": [
                {
                    "source": "user",
                    "models_usage": None,
                    "metadata": {},
                    "content": "What is the weather in New York?",
                    "type": "TextMessage",
                },
                {
                    "source": "weather_agent",
                    "models_usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "metadata": {},
                    "content": [
                        {
                            "id": "call_rOzPHDXJlI50BqN305vbzckW",
                            "arguments": '{"city":"New York"}',
                            "name": "get_weather",
                        }
                    ],
                    "type": "ToolCallRequestEvent",
                },
                {
                    "source": "weather_agent",
                    "models_usage": None,
                    "metadata": {},
                    "content": [
                        {
                            "content": "The weather in New York is 73 degrees and Sunny.",
                            "name": "get_weather",
                            "call_id": "call_rOzPHDXJlI50BqN305vbzckW",
                            "is_error": False,
                        }
                    ],
                    "type": "ToolCallExecutionEvent",
                },
                {
                    "source": "weather_agent",
                    "models_usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "metadata": {},
                    "content": "The weather in New York is 73 degrees and sunny.",
                    "type": "TextMessage",
                },
            ],
            "stop_reason": None,
        }
        assert json.loads(attributes["output.value"]) == expected_output


class TestTeam:
    @pytest.mark.asyncio
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_response_headers,
        before_record_request=remove_all_vcr_request_headers,
        decode_compressed_response=True,
    )
    async def test_team_run(
        self,
        tracer_provider: trace_api.TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import TextMentionTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
        )

        # Create two agents: a primary and a critic
        primary_agent = AssistantAgent(
            "primary",
            model_client=model_client,
            system_message="You are a helpful AI assistant.",
        )

        critic_agent = AssistantAgent(
            "critic",
            model_client=model_client,
            system_message="""
            Provide constructive feedback.
            Respond with 'APPROVE' when your feedbacks are addressed.
            """,
        )

        # Termination condition: stop when the critic says "APPROVE"
        text_termination = TextMentionTermination("APPROVE")

        # Create a team with both agents
        team = RoundRobinGroupChat(
            [primary_agent, critic_agent], termination_condition=text_termination
        )

        # Run the team on a task
        _ = await team.run(task="Write a short poem about the fall season.")
        await model_client.close()

        # Verify that spans were created
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 17
        final_span = spans[-1]
        assert final_span.name == "RoundRobinGroupChat.run"
        assert final_span.status.is_ok

        attributes = dict(cast(Mapping[str, AttributeValue], final_span.attributes))
        # Verify input value
        assert attributes["input.value"] == '{"task": "Write a short poem about the fall season."}'

        # Verify team information
        assert "team_key" in attributes
        assert "team_id" in attributes
        assert attributes["team_key"] == attributes["team_id"]  # team_key and team_id should match

        # Verify participant information
        assert attributes["participant_names"] == ("primary", "critic")
        assert len(attributes["participant_descriptions"]) == 2
        assert attributes["openinference.span.kind"] == "AGENT"
