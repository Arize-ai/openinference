"""Tests for Strands to OpenInference processor."""

import json
from typing import Any, Dict, List, Optional

import pytest
from opentelemetry.trace import SpanKind, Status, StatusCode

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.config import REDACTED_VALUE
from openinference.instrumentation.strands_agents.processor import (
    StrandsAgentsToOpenInferenceProcessor,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


class MockEvent:
    """Mock event for testing."""

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.attributes: Dict[str, Any] = attributes or {}


class MockReadableSpan:
    """Mock ReadableSpan for testing."""

    def __init__(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        events: Optional[List[Any]] = None,
    ) -> None:
        self.name = name
        self._attributes = attributes or {}
        self._events = events or []
        self._status = Status(status_code=StatusCode.OK)
        self.kind = SpanKind.INTERNAL
        self.parent = None

    @property
    def status(self) -> Status:
        return self._status

    def get_span_context(self) -> Any:
        """Mock get_span_context."""

        class MockSpanContext:
            def __init__(self) -> None:
                self.span_id = 12345

        return MockSpanContext()

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON dict."""
        return {
            "name": self.name,
            "attributes": self._attributes,
            "events": self._events,
        }


class TestStrandsAgentsToOpenInferenceProcessor:
    """Test cases for StrandsAgentsToOpenInferenceProcessor."""

    def test_processor_can_be_instantiated(self) -> None:
        """Test that the processor can be instantiated."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        assert processor is not None

    def test_processor_transforms_llm_span(self) -> None:
        """Test that the processor transforms LLM spans correctly."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "gen_ai.system": "strands-agents",
            },
        )

        # Call on_end to process the span
        processor.on_end(span)  # type: ignore[arg-type]

        # Check that attributes were transformed
        assert span._attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4"
        assert span._attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 100
        assert span._attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 50
        assert (
            span._attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.LLM.value
        )

    def _llm_span(
        self, events: Optional[List[Any]] = None, attributes: Optional[Dict[str, Any]] = None
    ) -> MockReadableSpan:
        return MockReadableSpan(
            name="chat",
            attributes={"gen_ai.request.model": "gpt-4", "gen_ai.system": "strands-agents"}
            | (attributes or {}),
            events=events,
        )

    @staticmethod
    def _flat(prefix: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Flatten messages into the span attribute keys they are expected to produce."""
        flat: Dict[str, Any] = {}
        for i, message in enumerate(messages):
            for key, value in message.items():
                if key == "message.tool_calls":
                    for j, call in enumerate(value):
                        for call_key, call_value in call.items():
                            flat[f"{prefix}.{i}.message.tool_calls.{j}.{call_key}"] = call_value
                else:
                    flat[f"{prefix}.{i}.{key}"] = value
        return flat

    @staticmethod
    def _pop_all(attributes: Dict[str, Any], expected: Dict[str, Any]) -> None:
        for key, value in expected.items():
            assert attributes.pop(key) == value

    @staticmethod
    def _chat_attributes(
        span: MockReadableSpan, raw: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Copy the span's attributes, popping the ones every `chat` span test shares.

        `raw` is whatever extra attributes the test put on the span. They stay on the span
        as-is, but the content ones must not be copied into `metadata`.
        """
        raw = raw or {}
        attributes = dict(span._attributes)
        assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
        assert attributes.pop("graph.node.id") == "llm_12345"
        assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4"
        assert attributes.pop("gen_ai.request.model") == "gpt-4"
        assert attributes.pop("gen_ai.system") == "strands-agents"
        for key, value in raw.items():
            assert attributes.pop(key) == value
        assert json.loads(attributes.pop(SpanAttributes.METADATA)) == {
            "gen_ai.system": "strands-agents"
        }
        return attributes

    SYSTEM_BRIEF = {"message.role": "system", "message.content": "Be brief."}
    WEATHER_CALL = {
        "tool_call.id": "call_1",
        "tool_call.function.name": "weather",
        "tool_call.function.arguments": '{"c": "Paris"}',
    }
    WEATHER_TOOL_USE = (
        '[{"toolUse": {"toolUseId": "call_1", "name": "weather", "input": {"c": "Paris"}}}]'
    )
    WEATHER_TOOL_RESULT = (
        '[{"toolResult": {"status": "success", "content": [{"text": "sunny"}], '
        '"toolUseId": "call_1"}}]'
    )

    def test_system_message_event_is_first_input_message(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = self._llm_span(
            events=[
                MockEvent(
                    "gen_ai.system.message",
                    {"content": '[{"text": "You are a helpful weather assistant."}]'},
                ),
                MockEvent("gen_ai.user.message", {"content": '[{"text": "What\'s the weather?"}]'}),
            ]
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = self._chat_attributes(span)
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_INPUT_MESSAGES,
                [
                    {
                        "message.role": "system",
                        "message.content": "You are a helpful weather assistant.",
                    },
                    {"message.role": "user", "message.content": "What's the weather?"},
                ],
            ),
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "What's the weather?"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert not attributes

    def test_messages_are_only_flattened_not_a_json_string(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = self._llm_span(
            events=[
                MockEvent("gen_ai.user.message", {"content": "Hi"}),
                MockEvent(
                    "gen_ai.choice",
                    {"message": '[{"text": "Hello"}]', "finish_reason": "end_turn"},
                ),
            ]
        )
        processor.on_end(span)  # type: ignore[arg-type]

        # The bare llm.input_messages / llm.output_messages keys must not be left behind.
        attributes = self._chat_attributes(span)
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_INPUT_MESSAGES,
                [{"message.role": "user", "message.content": "Hi"}],
            ),
        )
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_OUTPUT_MESSAGES,
                [
                    {
                        "message.role": "assistant",
                        "message.content": "Hello",
                        "message.finish_reason": "end_turn",
                    }
                ],
            ),
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hi"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Hello"
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_tool_span_input_messages_are_only_flattened(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        raw = {
            "gen_ai.system": "strands-agents",
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "weather",
            "gen_ai.tool.call.id": "call_1",
        }
        span = MockReadableSpan(
            name="execute_tool weather",
            attributes=dict(raw),
            events=[
                MockEvent(
                    "gen_ai.tool.message",
                    {"content": '{"city": "Paris"}', "role": "tool", "id": "call_1"},
                ),
            ],
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = dict(span._attributes)
        for key, value in raw.items():
            assert attributes.pop(key) == value
        assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "TOOL"
        assert attributes.pop("graph.node.id") == "tool_weather_12345"
        assert attributes.pop(SpanAttributes.TOOL_NAME) == "weather"
        assert attributes.pop("tool.call_id") == "call_1"
        assert json.loads(attributes.pop(SpanAttributes.TOOL_PARAMETERS)) == {"city": "Paris"}
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_INPUT_MESSAGES,
                [
                    {
                        "message.role": "assistant",
                        "message.content": "",
                        "message.tool_calls": [
                            {
                                "tool_call.id": "call_1",
                                "tool_call.function.name": "weather",
                                "tool_call.function.arguments": '{"city": "Paris"}',
                            }
                        ],
                    }
                ],
            ),
        )
        assert json.loads(attributes.pop(SpanAttributes.INPUT_VALUE)) == {"city": "Paris"}
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
        assert json.loads(attributes.pop(SpanAttributes.METADATA)) == raw
        assert not attributes

    def test_system_message_event_accepts_plain_string(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = self._llm_span(
            events=[
                MockEvent("gen_ai.system.message", {"content": "Be brief."}),
                MockEvent("gen_ai.user.message", {"content": "Hi"}),
            ]
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = self._chat_attributes(span)
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_INPUT_MESSAGES,
                [self.SYSTEM_BRIEF, {"message.role": "user", "message.content": "Hi"}],
            ),
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hi"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert not attributes

    def test_system_message_does_not_change_plain_input_value(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = self._llm_span(
            events=[
                MockEvent("gen_ai.system.message", {"content": '[{"text": "Be brief."}]'}),
                MockEvent("gen_ai.user.message", {"content": '[{"text": "Hi"}]'}),
            ]
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = self._chat_attributes(span)
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_INPUT_MESSAGES,
                [self.SYSTEM_BRIEF, {"message.role": "user", "message.content": "Hi"}],
            ),
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hi"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert not attributes

    def test_system_instructions_attribute_becomes_system_message(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        raw = {
            "gen_ai.system_instructions": (
                '[{"type": "text", "content": "You are terse."},'
                ' {"type": "text", "content": "Answer in one line."}]'
            )
        }
        span = self._llm_span(
            events=[MockEvent("gen_ai.user.message", {"content": "Hello"})], attributes=raw
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = self._chat_attributes(span, raw)
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_INPUT_MESSAGES,
                [
                    {
                        "message.role": "system",
                        "message.content": "You are terse. Answer in one line.",
                    },
                    {"message.role": "user", "message.content": "Hello"},
                ],
            ),
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hello"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert not attributes

    def test_system_instructions_attribute_without_events(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        raw = {"gen_ai.system_instructions": "Plain text prompt"}
        span = self._llm_span(attributes=raw)
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = self._chat_attributes(span, raw)
        system = {"message.role": "system", "message.content": "Plain text prompt"}
        self._pop_all(attributes, self._flat(SpanAttributes.LLM_INPUT_MESSAGES, [system]))
        assert json.loads(attributes.pop(SpanAttributes.INPUT_VALUE)) == {
            "messages": [system],
            "model": "gpt-4",
        }
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
        assert not attributes

    def test_system_instructions_on_operation_details_event(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = self._llm_span(
            events=[
                self._details(
                    **{"gen_ai.system_instructions": '[{"type": "text", "content": "Be kind."}]'}
                ),
                MockEvent("gen_ai.user.message", {"content": "Hello"}),
            ]
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = self._chat_attributes(span)
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_INPUT_MESSAGES,
                [
                    {"message.role": "system", "message.content": "Be kind."},
                    {"message.role": "user", "message.content": "Hello"},
                ],
            ),
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hello"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert not attributes

    def test_system_prompt_is_not_duplicated_across_sources(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        raw = {"gen_ai.system_instructions": '[{"type": "text", "content": "From attr."}]'}
        span = self._llm_span(
            events=[
                MockEvent("gen_ai.system.message", {"content": '[{"text": "From event."}]'}),
                MockEvent("gen_ai.user.message", {"content": "Hello"}),
            ],
            attributes=raw,
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = self._chat_attributes(span, raw)
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_INPUT_MESSAGES,
                [
                    {"message.role": "system", "message.content": "From event."},
                    {"message.role": "user", "message.content": "Hello"},
                ],
            ),
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hello"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert not attributes

    def test_empty_system_instructions_are_ignored(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        raw = {"gen_ai.system_instructions": "[]"}
        span = self._llm_span(
            events=[MockEvent("gen_ai.user.message", {"content": "Hello"})], attributes=raw
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = self._chat_attributes(span, raw)
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_INPUT_MESSAGES,
                [{"message.role": "user", "message.content": "Hello"}],
            ),
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hello"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert not attributes

    def test_tool_loop_history_stays_in_input_messages(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = self._llm_span(
            events=[
                MockEvent("gen_ai.system.message", {"content": '[{"text": "Be brief."}]'}),
                MockEvent("gen_ai.user.message", {"content": '[{"text": "Weather?"}]'}),
                MockEvent("gen_ai.assistant.message", {"content": self.WEATHER_TOOL_USE}),
                MockEvent("gen_ai.tool.message", {"content": self.WEATHER_TOOL_RESULT}),
                MockEvent(
                    "gen_ai.choice",
                    {"message": '[{"text": "Sunny."}]', "finish_reason": "end_turn"},
                ),
            ]
        )
        processor.on_end(span)  # type: ignore[arg-type]

        history: List[Dict[str, Any]] = [
            self.SYSTEM_BRIEF,
            {"message.role": "user", "message.content": "Weather?"},
            {"message.role": "assistant", "message.tool_calls": [self.WEATHER_CALL]},
            {
                "message.role": "tool",
                "message.content": "sunny",
                "message.tool_call_id": "call_1",
                "message.name": "weather",
            },
        ]
        answer = {
            "message.role": "assistant",
            "message.content": "Sunny.",
            "message.finish_reason": "end_turn",
        }
        attributes = self._chat_attributes(span)
        self._pop_all(attributes, self._flat(SpanAttributes.LLM_INPUT_MESSAGES, history))
        self._pop_all(attributes, self._flat(SpanAttributes.LLM_OUTPUT_MESSAGES, [answer]))
        assert json.loads(attributes.pop(SpanAttributes.INPUT_VALUE)) == {
            "messages": history,
            "model": "gpt-4",
        }
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Sunny."
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_llm_span_values_for_tool_call_and_text_turns(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        call_span = self._llm_span(
            events=[
                MockEvent("gen_ai.user.message", {"content": '[{"text": "Weather?"}]'}),
                MockEvent(
                    "gen_ai.choice",
                    {"message": self.WEATHER_TOOL_USE, "finish_reason": "tool_use"},
                ),
            ]
        )
        answer_span = self._llm_span(
            events=[
                MockEvent("gen_ai.user.message", {"content": '[{"text": "Weather?"}]'}),
                MockEvent("gen_ai.assistant.message", {"content": self.WEATHER_TOOL_USE}),
                MockEvent("gen_ai.tool.message", {"content": self.WEATHER_TOOL_RESULT}),
                MockEvent(
                    "gen_ai.choice",
                    {"message": '[{"text": "Sunny."}]', "finish_reason": "end_turn"},
                ),
            ]
        )
        processor.on_end(call_span)  # type: ignore[arg-type]
        processor.on_end(answer_span)  # type: ignore[arg-type]

        user = {"message.role": "user", "message.content": "Weather?"}
        call_attributes = self._chat_attributes(call_span)
        self._pop_all(call_attributes, self._flat(SpanAttributes.LLM_INPUT_MESSAGES, [user]))
        self._pop_all(
            call_attributes,
            self._flat(
                SpanAttributes.LLM_OUTPUT_MESSAGES,
                [
                    {
                        "message.role": "assistant",
                        "message.tool_calls": [self.WEATHER_CALL],
                        "message.finish_reason": "tool_use",
                    }
                ],
            ),
        )
        assert call_attributes.pop(SpanAttributes.INPUT_VALUE) == "Weather?"
        assert call_attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert json.loads(call_attributes.pop(SpanAttributes.OUTPUT_VALUE)) == [
            {"id": "call_1", "name": "weather", "arguments": '{"c": "Paris"}'}
        ]
        assert call_attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
        assert call_attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "tool_use"
        assert not call_attributes

        history: List[Dict[str, Any]] = [
            user,
            {"message.role": "assistant", "message.tool_calls": [self.WEATHER_CALL]},
            {
                "message.role": "tool",
                "message.content": "sunny",
                "message.tool_call_id": "call_1",
                "message.name": "weather",
            },
        ]
        answer_attributes = self._chat_attributes(answer_span)
        self._pop_all(answer_attributes, self._flat(SpanAttributes.LLM_INPUT_MESSAGES, history))
        self._pop_all(
            answer_attributes,
            self._flat(
                SpanAttributes.LLM_OUTPUT_MESSAGES,
                [
                    {
                        "message.role": "assistant",
                        "message.content": "Sunny.",
                        "message.finish_reason": "end_turn",
                    }
                ],
            ),
        )
        # The model saw the whole tool loop, so input.value is the full request.
        assert json.loads(answer_attributes.pop(SpanAttributes.INPUT_VALUE)) == {
            "messages": history,
            "model": "gpt-4",
        }
        assert answer_attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
        assert answer_attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Sunny."
        assert answer_attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
        assert answer_attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not answer_attributes

    # Payloads below are copied from spans emitted by a real Strands agent with
    # OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental.
    LATEST_SYSTEM = '[{"type": "text", "content": "You are a weather bot."}]'
    LATEST_USER = (
        '[{"role": "user", "parts": '
        '[{"type": "text", "content": "What is the weather in Paris?"}]}]'
    )
    LATEST_TOOL_CALL = (
        '[{"role": "assistant", "parts": [{"type": "tool_call", "name": "get_weather", '
        '"id": "t1", "arguments": {"city": "Paris"}}], "finish_reason": "tool_use"}]'
    )
    LATEST_HISTORY = (
        '[{"role": "user", "parts": '
        '[{"type": "text", "content": "What is the weather in Paris?"}]}, '
        '{"role": "assistant", "parts": [{"type": "tool_call", "name": "get_weather", "id": "t1", '
        '"arguments": {"city": "Paris"}}]}, '
        '{"role": "user", "parts": [{"type": "tool_call_response", "id": "t1", '
        '"response": [{"text": "Sunny in Paris"}]}]}]'
    )
    LATEST_ANSWER = (
        '[{"role": "assistant", "parts": [{"type": "text", "content": "It is sunny in Paris."}], '
        '"finish_reason": "end_turn"}]'
    )
    PARIS_SYSTEM = {"message.role": "system", "message.content": "You are a weather bot."}
    PARIS_USER = {"message.role": "user", "message.content": "What is the weather in Paris?"}
    PARIS_CALL = {
        "tool_call.id": "t1",
        "tool_call.function.name": "get_weather",
        "tool_call.function.arguments": '{"city": "Paris"}',
    }
    PARIS_ANSWER = {
        "message.role": "assistant",
        "message.content": "It is sunny in Paris.",
        "message.finish_reason": "end_turn",
    }

    @staticmethod
    def _details(**attributes: str) -> MockEvent:
        return MockEvent("gen_ai.client.inference.operation.details", dict(attributes))

    def test_latest_conventions_tool_call_turn(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = self._llm_span(
            events=[
                self._details(**{"gen_ai.system_instructions": self.LATEST_SYSTEM}),
                self._details(**{"gen_ai.input.messages": self.LATEST_USER}),
                self._details(**{"gen_ai.output.messages": self.LATEST_TOOL_CALL}),
            ]
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = self._chat_attributes(span)
        self._pop_all(
            attributes,
            self._flat(SpanAttributes.LLM_INPUT_MESSAGES, [self.PARIS_SYSTEM, self.PARIS_USER]),
        )
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_OUTPUT_MESSAGES,
                [
                    {
                        "message.role": "assistant",
                        "message.tool_calls": [self.PARIS_CALL],
                        "message.finish_reason": "tool_use",
                    }
                ],
            ),
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "What is the weather in Paris?"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert json.loads(attributes.pop(SpanAttributes.OUTPUT_VALUE)) == [
            {"id": "t1", "name": "get_weather", "arguments": '{"city": "Paris"}'}
        ]
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "tool_use"
        assert not attributes

    def test_latest_conventions_tool_loop_turn(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = self._llm_span(
            events=[
                self._details(**{"gen_ai.system_instructions": self.LATEST_SYSTEM}),
                self._details(**{"gen_ai.input.messages": self.LATEST_HISTORY}),
                self._details(**{"gen_ai.output.messages": self.LATEST_ANSWER}),
            ]
        )
        processor.on_end(span)  # type: ignore[arg-type]

        history: List[Dict[str, Any]] = [
            self.PARIS_SYSTEM,
            self.PARIS_USER,
            {"message.role": "assistant", "message.tool_calls": [self.PARIS_CALL]},
            {
                "message.role": "tool",
                "message.tool_call_id": "t1",
                "message.content": "Sunny in Paris",
                "message.name": "get_weather",
            },
        ]
        attributes = self._chat_attributes(span)
        self._pop_all(attributes, self._flat(SpanAttributes.LLM_INPUT_MESSAGES, history))
        self._pop_all(
            attributes, self._flat(SpanAttributes.LLM_OUTPUT_MESSAGES, [self.PARIS_ANSWER])
        )
        assert json.loads(attributes.pop(SpanAttributes.INPUT_VALUE)) == {
            "messages": history,
            "model": "gpt-4",
        }
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "It is sunny in Paris."
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_latest_conventions_recorded_as_span_attributes(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        raw = {
            "gen_ai.system_instructions": self.LATEST_SYSTEM,
            "gen_ai.input.messages": self.LATEST_USER,
            "gen_ai.output.messages": self.LATEST_ANSWER,
        }
        span = self._llm_span(attributes=raw)
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = self._chat_attributes(span, raw)
        self._pop_all(
            attributes,
            self._flat(SpanAttributes.LLM_INPUT_MESSAGES, [self.PARIS_SYSTEM, self.PARIS_USER]),
        )
        self._pop_all(
            attributes, self._flat(SpanAttributes.LLM_OUTPUT_MESSAGES, [self.PARIS_ANSWER])
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "What is the weather in Paris?"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "It is sunny in Paris."
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_latest_conventions_ignore_malformed_messages(self) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = self._llm_span(
            events=[
                self._details(**{"gen_ai.input.messages": "not json"}),
                self._details(
                    **{
                        "gen_ai.output.messages": (
                            '[{"role": "assistant", '
                            '"parts": [{"type": "reasoning", "content": "x"}]}, '
                            '"oops", {"role": "assistant", "parts": ["oops"]}]'
                        )
                    }
                ),
            ]
        )
        processor.on_end(span)  # type: ignore[arg-type]

        assert not self._chat_attributes(span)

    def test_agent_span_system_prompt_attribute(self) -> None:
        """Strands 1.19-1.33 record the system prompt only as `system_prompt` on the agent span."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        raw = {
            "gen_ai.system": "strands-agents",
            "gen_ai.agent.name": "A",
            "system_prompt": '["keep this literal"]',
        }
        span = MockReadableSpan(
            name="invoke_agent A",
            attributes=dict(raw),
            events=[MockEvent("gen_ai.user.message", {"content": '[{"text": "Hi"}]'})],
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = dict(span._attributes)
        for key, value in raw.items():
            assert attributes.pop(key) == value
        assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "AGENT"
        assert attributes.pop("graph.node.id") == "strands_agent"
        assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "strands-agents"
        assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "strands-agents"
        self._pop_all(
            attributes,
            self._flat(
                SpanAttributes.LLM_INPUT_MESSAGES,
                [
                    {"message.role": "system", "message.content": '["keep this literal"]'},
                    {"message.role": "user", "message.content": "Hi"},
                ],
            ),
        )
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hi"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert json.loads(attributes.pop(SpanAttributes.METADATA)) == {
            "gen_ai.system": "strands-agents",
            "gen_ai.agent.name": "A",
        }
        assert not attributes

    def test_processor_maps_cache_token_counts(self) -> None:
        """Cache read/write tokens map to prompt_details and roll up into the prompt count."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "gen_ai.usage.input_tokens": 4299,
                "gen_ai.usage.output_tokens": 631,
                "gen_ai.usage.cache_read_input_tokens": 35574,
                "gen_ai.usage.cache_write_input_tokens": 17787,
                "gen_ai.usage.total_tokens": 58291,
                "gen_ai.system": "strands-agents",
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        attributes = span._attributes
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 35574
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE) == 17787
        # Prompt aggregate includes cached tokens so prompt + completion == total
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 4299 + 35574 + 17787
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 631
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 58291

    def test_processor_omits_cache_details_when_zero(self) -> None:
        """Zero cache counts (Strands emits 0 when caching is unused) add no attributes."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "gen_ai.usage.total_tokens": 150,
                "gen_ai.usage.cache_read_input_tokens": 0,
                "gen_ai.usage.cache_write_input_tokens": 0,
                "gen_ai.system": "strands-agents",
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        attributes = span._attributes
        assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ not in attributes
        assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE not in attributes
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 100
        assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 150

    def test_processor_transforms_agent_span(self) -> None:
        """Test that the processor transforms agent spans correctly."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="invoke_agent test_agent",
            attributes={
                "agent.name": "test_agent",
                "gen_ai.provider.name": "strands-agents",
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        assert (
            span._attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.AGENT.value
        )

    def test_processor_transforms_tool_span(self) -> None:
        """Test that the processor transforms tool spans correctly."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="execute_tool calculator",
            attributes={
                "gen_ai.tool.name": "calculator",
                "gen_ai.provider.name": "strands-agents",
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._attributes.get(SpanAttributes.TOOL_NAME) == "calculator"
        assert (
            span._attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.TOOL.value
        )

    def test_processor_transforms_chain_span(self) -> None:
        """Test that the processor transforms chain spans correctly."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="execute_event_loop_cycle",
            attributes={
                "event_loop.cycle_id": "cycle-123",
            },
        )

        processor.on_end(span)  # type: ignore[arg-type]

        assert (
            span._attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.CHAIN.value
        )

    @pytest.mark.parametrize(
        ("span_name", "attributes"),
        [
            # HTTP span
            (
                "http.request",
                {
                    "http.method": "GET",
                    "http.url": "https://example.com/api",
                    "http.status_code": 200,
                },
            ),
            # RPC span
            (
                "rpc.call",
                {
                    "rpc.system": "grpc",
                    "rpc.service": "my.Service",
                    "rpc.method": "DoWork",
                },
            ),
            # AWS span
            (
                "aws.request",
                {
                    "rpc.system": "aws-api",
                    "rpc.service": "S3",
                    "rpc.method": "GetObject",
                },
            ),
        ],
    )
    def test_processor_leaves_non_strands_spans_unchanged(
        self,
        span_name: str,
        attributes: Dict[str, Any],
    ) -> None:
        """Test that non-Strands spans are not modified by the processor."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(name=span_name, attributes=dict(attributes))

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._attributes == attributes

    def test_mixed_trace_only_transforms_strands_spans(self) -> None:
        """Test that only Strands spans are transformed in mixed traces."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        http_span = MockReadableSpan(
            name="http.request",
            attributes={
                "http.method": "GET",
                "http.url": "https://example.com",
            },
        )
        strands_span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.system": "strands-agents",
            },
        )

        processor.on_end(http_span)  # type: ignore[arg-type]
        processor.on_end(strands_span)  # type: ignore[arg-type]

        assert http_span._attributes.get("http.method") == "GET"
        assert SpanAttributes.LLM_MODEL_NAME in strands_span._attributes

    def test_processor_ignores_non_strands_genai_span(self) -> None:
        """Test that spans from other GenAI SDKs remain unchanged."""
        processor = StrandsAgentsToOpenInferenceProcessor()

        event = MockEvent("gen_ai.request")

        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.system": "openai",
            },
            events=[event],
        )

        original_attrs = dict(span._attributes)
        original_events = list(span._events)
        original_status = span._status

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._attributes == original_attrs
        assert span._events == original_events
        assert span._status is original_status

    def test_processor_ignores_event_loop_span_without_cycle_id(self) -> None:
        """Test that event loop spans without a cycle ID are ignored."""
        processor = StrandsAgentsToOpenInferenceProcessor()

        span = MockReadableSpan(
            name="execute_event_loop_cycle",
            attributes={},
        )

        processor.on_end(span)  # type: ignore[arg-type]

        assert SpanAttributes.OPENINFERENCE_SPAN_KIND not in span._attributes

    def test_processor_does_not_overwrite_error_status(self) -> None:
        """Test that processor does not overwrite ERROR spans."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
            },
        )
        span._status = Status(StatusCode.ERROR)

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._status.status_code == StatusCode.ERROR

    def test_processor_sets_ok_status_when_not_error(self) -> None:
        """Test that spans without errors are normalized to OK."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.system": "strands-agents",
            },
        )
        span._status = Status(StatusCode.UNSET)

        processor.on_end(span)  # type: ignore[arg-type]

        assert span._status.status_code == StatusCode.OK

    def test_processor_handles_empty_attributes(self) -> None:
        """Test that the processor handles spans with no attributes."""
        processor = StrandsAgentsToOpenInferenceProcessor()
        span = MockReadableSpan(name="test_span", attributes={})

        # Should not raise an exception
        processor.on_end(span)  # type: ignore[arg-type]

    def test_processor_debug_mode(self) -> None:
        """Test that debug mode works."""
        processor = StrandsAgentsToOpenInferenceProcessor(debug=True)
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
            },
        )

        # Should not raise an exception
        processor.on_end(span)  # type: ignore[arg-type]


class TestTraceConfigMasking:
    """The processor honors TraceConfig for everything it copies out of Strands spans."""

    LATEST_SYSTEM = '[{"type": "text", "content": "You are a weather bot."}]'
    LATEST_USER = (
        '[{"role": "user", "parts": '
        '[{"type": "text", "content": "What is the weather in Paris?"}]}]'
    )
    LATEST_ANSWER = (
        '[{"role": "assistant", "parts": [{"type": "text", "content": "It is sunny in Paris."}], '
        '"finish_reason": "end_turn"}]'
    )
    RAW_LATEST = {
        "gen_ai.system_instructions": LATEST_SYSTEM,
        "gen_ai.input.messages": LATEST_USER,
        "gen_ai.output.messages": LATEST_ANSWER,
    }
    SYSTEM = {"message.role": "system", "message.content": "You are a weather bot."}
    USER = {"message.role": "user", "message.content": "What is the weather in Paris?"}
    ANSWER = {
        "message.role": "assistant",
        "message.content": "It is sunny in Paris.",
        "message.finish_reason": "end_turn",
    }

    def _processed_chat_span(self, config: Optional[TraceConfig]) -> Dict[str, Any]:
        """Run a `chat` span that records the latest conventions as span attributes."""
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.request.model": "gpt-4",
                "gen_ai.system": "strands-agents",
                **self.RAW_LATEST,
            },
        )
        StrandsAgentsToOpenInferenceProcessor(config=config).on_end(span)  # type: ignore[arg-type]
        attributes = dict(span._attributes)
        assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
        assert attributes.pop("graph.node.id") == "llm_12345"
        assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4"
        assert attributes.pop("gen_ai.request.model") == "gpt-4"
        assert attributes.pop("gen_ai.system") == "strands-agents"
        assert json.loads(attributes.pop(SpanAttributes.METADATA)) == {
            "gen_ai.system": "strands-agents"
        }
        return attributes

    @staticmethod
    def _pop_messages(attributes: Dict[str, Any], prefix: str, *messages: Dict[str, Any]) -> None:
        for i, message in enumerate(messages):
            for key, value in message.items():
                assert attributes.pop(f"{prefix}.{i}.{key}") == value

    def test_nothing_is_hidden_by_default(self) -> None:
        attributes = self._processed_chat_span(TraceConfig())

        for key, value in self.RAW_LATEST.items():
            assert attributes.pop(key) == value
        self._pop_messages(attributes, SpanAttributes.LLM_INPUT_MESSAGES, self.SYSTEM, self.USER)
        self._pop_messages(attributes, SpanAttributes.LLM_OUTPUT_MESSAGES, self.ANSWER)
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "What is the weather in Paris?"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "It is sunny in Paris."
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_hide_inputs_removes_system_prompt_and_input_messages(self) -> None:
        attributes = self._processed_chat_span(TraceConfig(hide_inputs=True))

        # Only the output side of the raw attributes is left.
        assert attributes.pop("gen_ai.output.messages") == self.LATEST_ANSWER
        self._pop_messages(attributes, SpanAttributes.LLM_OUTPUT_MESSAGES, self.ANSWER)
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == REDACTED_VALUE
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "It is sunny in Paris."
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_hide_input_messages_keeps_input_value(self) -> None:
        attributes = self._processed_chat_span(TraceConfig(hide_input_messages=True))

        assert attributes.pop("gen_ai.output.messages") == self.LATEST_ANSWER
        self._pop_messages(attributes, SpanAttributes.LLM_OUTPUT_MESSAGES, self.ANSWER)
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "What is the weather in Paris?"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "It is sunny in Paris."
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_hide_input_text_redacts_message_content_only(self) -> None:
        attributes = self._processed_chat_span(TraceConfig(hide_input_text=True))

        assert attributes.pop("gen_ai.output.messages") == self.LATEST_ANSWER
        self._pop_messages(
            attributes,
            SpanAttributes.LLM_INPUT_MESSAGES,
            {"message.role": "system", "message.content": REDACTED_VALUE},
            {"message.role": "user", "message.content": REDACTED_VALUE},
        )
        self._pop_messages(attributes, SpanAttributes.LLM_OUTPUT_MESSAGES, self.ANSWER)
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "What is the weather in Paris?"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "It is sunny in Paris."
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_hide_outputs_removes_output_messages(self) -> None:
        attributes = self._processed_chat_span(TraceConfig(hide_outputs=True))

        assert attributes.pop("gen_ai.system_instructions") == self.LATEST_SYSTEM
        assert attributes.pop("gen_ai.input.messages") == self.LATEST_USER
        self._pop_messages(attributes, SpanAttributes.LLM_INPUT_MESSAGES, self.SYSTEM, self.USER)
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "What is the weather in Paris?"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == REDACTED_VALUE
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_hide_inputs_and_outputs_leave_no_content(self) -> None:
        attributes = self._processed_chat_span(TraceConfig(hide_inputs=True, hide_outputs=True))

        assert attributes.pop(SpanAttributes.INPUT_VALUE) == REDACTED_VALUE
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == REDACTED_VALUE
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_legacy_events_are_masked_too(self) -> None:
        span = MockReadableSpan(
            name="chat",
            attributes={"gen_ai.request.model": "gpt-4", "gen_ai.system": "strands-agents"},
            events=[
                MockEvent("gen_ai.system.message", {"content": '[{"text": "Be brief."}]'}),
                MockEvent("gen_ai.user.message", {"content": '[{"text": "Hi"}]'}),
            ],
        )
        processor = StrandsAgentsToOpenInferenceProcessor(config=TraceConfig(hide_inputs=True))
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = dict(span._attributes)
        assert attributes.pop("gen_ai.request.model") == "gpt-4"
        assert attributes.pop("gen_ai.system") == "strands-agents"
        assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4"
        assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
        assert attributes.pop("graph.node.id") == "llm_12345"
        assert json.loads(attributes.pop(SpanAttributes.METADATA)) == {
            "gen_ai.system": "strands-agents"
        }
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == REDACTED_VALUE
        assert not attributes

    def test_legacy_system_prompt_attribute_is_hidden(self) -> None:
        """Strands 1.19-1.33 keep the prompt in `system_prompt` on the agent span."""
        span = MockReadableSpan(
            name="invoke_agent A",
            attributes={
                "gen_ai.system": "strands-agents",
                "gen_ai.agent.name": "A",
                "system_prompt": "You are a weather bot.",
            },
            events=[MockEvent("gen_ai.user.message", {"content": '[{"text": "Hi"}]'})],
        )
        processor = StrandsAgentsToOpenInferenceProcessor(
            config=TraceConfig(hide_input_messages=True)
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = dict(span._attributes)
        assert attributes.pop("gen_ai.system") == "strands-agents"
        assert attributes.pop("gen_ai.agent.name") == "A"
        assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "AGENT"
        assert attributes.pop("graph.node.id") == "strands_agent"
        assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "strands-agents"
        assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "strands-agents"
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hi"
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
        assert json.loads(attributes.pop(SpanAttributes.METADATA)) == {
            "gen_ai.system": "strands-agents",
            "gen_ai.agent.name": "A",
        }
        assert not attributes

    def test_tool_arguments_and_result_are_hidden(self) -> None:
        raw = {
            "gen_ai.system": "strands-agents",
            "gen_ai.tool.name": "weather",
            "gen_ai.tool.call.id": "call_1",
            "gen_ai.tool.call.arguments": '{"city": "Paris"}',
            "gen_ai.tool.call.result": '[{"text": "Sunny"}]',
        }
        span = MockReadableSpan(name="execute_tool weather", attributes=dict(raw))
        processor = StrandsAgentsToOpenInferenceProcessor(
            config=TraceConfig(hide_inputs=True, hide_outputs=True)
        )
        processor.on_end(span)  # type: ignore[arg-type]

        attributes = dict(span._attributes)
        for key in ("gen_ai.system", "gen_ai.tool.name", "gen_ai.tool.call.id"):
            assert attributes.pop(key) == raw[key]
        assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "TOOL"
        assert attributes.pop("graph.node.id") == "tool_weather_12345"
        assert attributes.pop(SpanAttributes.TOOL_NAME) == "weather"
        assert attributes.pop("tool.call_id") == "call_1"
        assert json.loads(attributes.pop(SpanAttributes.METADATA)) == {
            "gen_ai.system": "strands-agents",
            "gen_ai.tool.name": "weather",
            "gen_ai.tool.call.id": "call_1",
        }
        assert not attributes

    def test_environment_variables_are_honored_without_a_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENINFERENCE_HIDE_INPUTS", "true")
        attributes = self._processed_chat_span(None)

        assert attributes.pop("gen_ai.output.messages") == self.LATEST_ANSWER
        self._pop_messages(attributes, SpanAttributes.LLM_OUTPUT_MESSAGES, self.ANSWER)
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == REDACTED_VALUE
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "It is sunny in Paris."
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
        assert not attributes

    def test_hidden_content_is_not_exported_when_the_transform_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        processor = StrandsAgentsToOpenInferenceProcessor(config=TraceConfig(hide_inputs=True))

        def boom(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("boom")

        monkeypatch.setattr(processor, "_transform_attributes", boom)
        span = MockReadableSpan(
            name="chat",
            attributes={
                "gen_ai.system": "strands-agents",
                "gen_ai.system_instructions": self.LATEST_SYSTEM,
                "gen_ai.input.messages": self.LATEST_USER,
                "gen_ai.output.messages": self.LATEST_ANSWER,
            },
        )
        processor.on_end(span)  # type: ignore[arg-type]

        assert span._attributes == {
            "gen_ai.system": "strands-agents",
            "gen_ai.output.messages": self.LATEST_ANSWER,
        }
