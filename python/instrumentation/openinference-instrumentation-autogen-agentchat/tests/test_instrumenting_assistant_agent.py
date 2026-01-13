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
        assert len(spans) == 3  # LLM + TOOL + LLM spans
        final_span = spans[-1]
        assert final_span.status.is_ok

        # Find and verify the tool execution span
        tool_span = next((s for s in spans if "get_weather" in s.name), None)
        assert tool_span is not None, "Expected to find tool execution span"
        assert tool_span.status.is_ok

        tool_attrs = dict(tool_span.attributes or {})
        assert tool_attrs.get("openinference.span.kind") == "TOOL"
        assert tool_attrs.get("tool.name") == "get_weather"
        assert tool_attrs.get("tool.description") == "Get the weather for a given city."
        assert "New York" in tool_attrs.get("input.value", "")
        assert "73 degrees and Sunny" in tool_attrs.get("output.value", "")

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

        # Check for tool call attributes instead of serialized FunctionCall content
        tool_call_id = attributes.pop(
            "llm.input_messages.2.message.tool_calls.0.tool_call.id", None
        )
        assert tool_call_id is not None, "Expected tool call ID to be present"

        tool_call_function_name = attributes.pop(
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.name", None
        )
        assert tool_call_function_name == "get_weather", (
            f"Expected function name 'get_weather', got {tool_call_function_name}"
        )

        tool_call_arguments = attributes.pop(
            "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments", None
        )
        assert tool_call_arguments is not None, "Expected tool call arguments to be present"

        # Verify the arguments are valid JSON and contain the expected city parameter
        import json

        args_dict = json.loads(tool_call_arguments)  # Should always be valid JSON
        assert "city" in args_dict, f"Expected 'city' in arguments, got {args_dict}"
        assert args_dict["city"] == "New York", (
            f"Expected city to be 'New York', got {args_dict['city']}"
        )
        assert attributes.pop("llm.input_messages.3.message.role") == "function"
        assert "get_weather" in str(attributes.pop("llm.input_messages.3.function.0"))

        # Check for tool calls in OUTPUT messages (LLM response)
        output_tool_call_id = attributes.pop(
            "llm.output_messages.0.message.tool_calls.0.tool_call.id", None
        )
        if output_tool_call_id is not None:
            # If output tool calls are present, verify they match our expectations
            output_tool_call_function_name = attributes.pop(
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name", None
            )
            assert output_tool_call_function_name == "get_weather", (
                f"Expected function name 'get_weather' in output, "
                f"got {output_tool_call_function_name}"
            )

            output_tool_call_arguments = attributes.pop(
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments", None
            )
            assert output_tool_call_arguments is not None, "Expected tool call arguments in output"

            output_args_dict = json.loads(output_tool_call_arguments)
            assert "city" in output_args_dict, (
                f"Expected 'city' in output arguments, got {output_args_dict}"
            )
            assert output_args_dict["city"] == "New York", (
                f"Expected city to be 'New York' in output, got {output_args_dict['city']}"
            )

            output_message_role = attributes.pop("llm.output_messages.0.message.role", None)
            assert output_message_role == "assistant", (
                f"Expected output message role to be 'assistant', got {output_message_role}"
            )

        assert attributes.pop("output.mime_type") == "application/json"
        assert attributes.pop("openinference.span.kind") == "LLM"
        output_value = json.loads(str(attributes.pop("output.value")))
        content_lower = output_value["content"].lower()
        assert "the weather in new york" in content_lower
        assert "73 degrees" in content_lower
        assert "sunny" in content_lower
        assert attributes.pop("llm.model_name", None) == "gpt-3.5-turbo-0125"
        assert attributes.pop("llm.provider", None) == "openai"
        assert attributes.pop("llm.system", None) == "openai"
        expected_content = "The weather in New York is currently 73 degrees and sunny."
        assert attributes.pop("llm.output_messages.0.message.content") == expected_content
        assert attributes.pop("llm.output_messages.0.message.role") == "assistant"
        assert attributes.pop("llm.token_count.completion") == 0
        assert attributes.pop("llm.token_count.prompt") == 0
        assert attributes.pop("llm.token_count.total") == 0
        assert not attributes
