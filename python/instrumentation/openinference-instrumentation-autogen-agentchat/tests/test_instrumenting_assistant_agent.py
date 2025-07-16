import json
from typing import Mapping, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue


class TestAssistantAgent:
    @pytest.mark.asyncio
    @pytest.mark.vcr(
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
        decode_compressed_response=True,
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

        assert attributes.pop("llm.input_messages.0.message.role") == "developer"
        assert (
            attributes.pop("llm.input_messages.0.message.content")
            == "You are a helpful assistant that can check the weather."
        )
        assert attributes.pop("llm.input_messages.1.message.role") == "user"
        assert (
            attributes.pop("llm.input_messages.1.message.content")
            == "What is the weather in New York?"
        )
        assert attributes.pop("llm.input_messages.2.message.role") == "assistant"
        assert "FunctionCall" in str(attributes.pop("llm.input_messages.2.message.content"))
        assert attributes.pop("llm.input_messages.3.message.role") == "function"
        assert "get_weather" in str(attributes.pop("llm.input_messages.3.function.0"))
        assert attributes.pop("output.mime_type") == "application/json"
        assert attributes.pop("openinference.span.kind") == "LLM"
        output_value = json.loads(str(attributes.pop("output.value")))
        content_lower = output_value["content"].lower()
        assert "the weather in new york" in content_lower
        assert "73 degrees" in content_lower
        assert "sunny" in content_lower
        assert not attributes
