"""Tests for AgentFrameworkToOpenInferenceProcessor."""

import json
from typing import Any, Dict
from unittest.mock import MagicMock

from openinference.instrumentation.agent_framework import (
    AgentFrameworkToOpenInferenceProcessor,
)


def create_mock_span(name: str, attributes: Dict[str, Any]) -> MagicMock:
    """Create a mock span with the given attributes."""
    span = MagicMock()
    span.name = name
    span._attributes = attributes
    span._events = []
    span.get_span_context.return_value.span_id = 12345
    return span


class TestSpanKindDetermination:
    """Tests for span kind determination."""

    def test_chat_operation_returns_llm(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "chat gpt-4", {"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4"}
        )

        processor.on_end(span)

        assert span._attributes["openinference.span.kind"] == "LLM"

    def test_execute_tool_operation_returns_tool(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "execute_tool get_weather",
            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_weather"},
        )

        processor.on_end(span)

        assert span._attributes["openinference.span.kind"] == "TOOL"

    def test_invoke_agent_operation_returns_agent(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "invoke_agent Assistant",
            {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "Assistant"},
        )

        processor.on_end(span)

        assert span._attributes["openinference.span.kind"] == "AGENT"

    def test_workflow_span_returns_chain(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "workflow.run my_workflow", {"workflow.id": "wf-123", "workflow.name": "my_workflow"}
        )

        processor.on_end(span)

        assert span._attributes["openinference.span.kind"] == "CHAIN"

    def test_unknown_operation_defaults_to_chain(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span("unknown_operation", {"some.attribute": "value"})

        processor.on_end(span)

        assert span._attributes["openinference.span.kind"] == "CHAIN"


class TestModelInfoMapping:
    """Tests for model and provider info mapping."""

    def test_model_name_mapped(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "chat gpt-4",
            {"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4-turbo"},
        )

        processor.on_end(span)

        assert span._attributes["llm.model_name"] == "gpt-4-turbo"

    def test_provider_mapped(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "chat gpt-4",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "gpt-4",
                "gen_ai.provider.name": "openai",
            },
        )

        processor.on_end(span)

        assert span._attributes["llm.provider"] == "openai"


class TestTokenUsageMapping:
    """Tests for token usage mapping."""

    def test_input_tokens_mapped_to_prompt(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "chat gpt-4",
            {"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 100},
        )

        processor.on_end(span)

        assert span._attributes["llm.token_count.prompt"] == 100

    def test_output_tokens_mapped_to_completion(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "chat gpt-4",
            {"gen_ai.operation.name": "chat", "gen_ai.usage.output_tokens": 50},
        )

        processor.on_end(span)

        assert span._attributes["llm.token_count.completion"] == 50

    def test_total_tokens_calculated(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "chat gpt-4",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
            },
        )

        processor.on_end(span)

        assert span._attributes["llm.token_count.total"] == 150


class TestMessageExtraction:
    """Tests for message extraction and transformation."""

    def test_simple_user_message(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        messages = json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": "Hello, world!"}]}]
        )
        span = create_mock_span(
            "chat gpt-4",
            {"gen_ai.operation.name": "chat", "gen_ai.input.messages": messages},
        )

        processor.on_end(span)

        assert span._attributes["llm.input_messages.0.message.role"] == "user"
        assert span._attributes["llm.input_messages.0.message.content"] == "Hello, world!"

    def test_assistant_message_with_tool_call(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        messages = json.dumps(
            [
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": "call_123",
                            "name": "get_weather",
                            "arguments": {"location": "Seattle"},
                        }
                    ],
                }
            ]
        )
        span = create_mock_span(
            "chat gpt-4",
            {"gen_ai.operation.name": "chat", "gen_ai.output.messages": messages},
        )

        processor.on_end(span)

        assert span._attributes["llm.output_messages.0.message.role"] == "assistant"
        assert (
            span._attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"]
            == "get_weather"
        )
        assert (
            span._attributes["llm.output_messages.0.message.tool_calls.0.tool_call.id"]
            == "call_123"
        )


class TestToolSpanHandling:
    """Tests for tool span handling."""

    def test_tool_attributes_mapped(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "execute_tool get_weather",
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "get_weather",
                "gen_ai.tool.call.id": "call_123",
                "gen_ai.tool.call.arguments": json.dumps({"location": "Seattle"}),
                "gen_ai.tool.call.result": "Sunny, 72F",
            },
        )

        processor.on_end(span)

        assert span._attributes["tool.name"] == "get_weather"
        assert span._attributes["tool.call_id"] == "call_123"
        assert span._attributes["output.value"] == "Sunny, 72F"


class TestGraphNodeAttributes:
    """Tests for graph node attribute generation."""

    def test_agent_span_graph_node(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "invoke_agent Assistant",
            {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.id": "agent-001",
                "gen_ai.agent.name": "Assistant",
            },
        )

        processor.on_end(span)

        assert span._attributes["graph.node.id"] == "agent_agent-001"
        assert span._attributes["graph.node.name"] == "Assistant"

    def test_tool_span_graph_node(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "execute_tool calculator",
            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "calculator"},
        )

        processor.on_end(span)

        assert "tool_calculator_" in span._attributes["graph.node.id"]
        assert span._attributes["graph.node.name"] == "calculator"


class TestSessionMapping:
    """Tests for session/conversation ID mapping."""

    def test_conversation_id_mapped_to_session(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "invoke_agent Assistant",
            {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.conversation.id": "conv-12345",
            },
        )

        processor.on_end(span)

        assert span._attributes["session.id"] == "conv-12345"


class TestInputOutputValues:
    """Tests for input/output value creation."""

    def test_simple_input_as_plain_text(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        messages = json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": "What is 2+2?"}]}]
        )
        span = create_mock_span(
            "chat gpt-4",
            {"gen_ai.operation.name": "chat", "gen_ai.input.messages": messages},
        )

        processor.on_end(span)

        assert span._attributes["input.value"] == "What is 2+2?"
        assert span._attributes["input.mime_type"] == "text/plain"

    def test_llm_output_as_json_structure(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        output_messages = json.dumps(
            [{"role": "assistant", "parts": [{"type": "text", "content": "The answer is 4."}]}]
        )
        span = create_mock_span(
            "chat gpt-4",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.output.messages": output_messages,
                "gen_ai.response.finish_reasons": json.dumps(["stop"]),
            },
        )

        processor.on_end(span)

        output_value = json.loads(span._attributes["output.value"])
        assert "choices" in output_value
        assert output_value["choices"][0]["message"]["content"] == "The answer is 4."
        assert output_value["choices"][0]["finish_reason"] == "stop"


class TestProcessorInfo:
    """Tests for processor info method."""

    def test_get_processor_info(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor(debug=True)

        info = processor.get_processor_info()

        assert info["processor_name"] == "AgentFrameworkToOpenInferenceProcessor"
        assert info["debug_enabled"] is True
        assert "LLM" in info["supported_span_kinds"]
        assert "AGENT" in info["supported_span_kinds"]
        assert "TOOL" in info["supported_span_kinds"]
        assert "CHAIN" in info["supported_span_kinds"]


class TestErrorHandling:
    """Tests for error handling."""

    def test_empty_span_attributes_handled(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = MagicMock()
        span._attributes = None

        # Should not raise
        processor.on_end(span)

    def test_invalid_json_in_messages_handled(self) -> None:
        processor = AgentFrameworkToOpenInferenceProcessor()
        span = create_mock_span(
            "chat gpt-4",
            {"gen_ai.operation.name": "chat", "gen_ai.input.messages": "not valid json{"},
        )

        # Should not raise
        processor.on_end(span)

        assert span._attributes["openinference.span.kind"] == "LLM"
