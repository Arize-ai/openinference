from types import SimpleNamespace
from typing import Any

from openinference.instrumentation.agno._model_wrapper import (
    _llm_input_messages,
    _parse_model_output_stream,
    _stream_output_messages,
)


def _chunk(content: str = "", tool_calls: Any = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(
    *,
    index: Any = None,
    id: Any = None,
    type: Any = None,
    name: Any = None,
    arguments: Any = None,
) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, type=type, function=function)


class TestOpenAIIndexedStreaming:
    def test_accumulates_real_openai_argument_fragments(self) -> None:
        # OpenAI sends args as fragments — real sequence recorded from live API
        chunks = [
            _chunk(
                tool_calls=[
                    _tool_call(
                        index=0,
                        id="call_7hIJuqbG8ixthENF1uc9NLZ1",
                        type="function",
                        name="get_weather",
                        arguments="",
                    )
                ]
            ),
            _chunk(tool_calls=[_tool_call(index=0, arguments='{"')]),
            _chunk(tool_calls=[_tool_call(index=0, arguments="city")]),
            _chunk(tool_calls=[_tool_call(index=0, arguments='":"')]),
            _chunk(tool_calls=[_tool_call(index=0, arguments="Paris")]),
            _chunk(tool_calls=[_tool_call(index=0, arguments='"}')]),
            _chunk(tool_calls=None),
        ]

        result = _parse_model_output_stream(chunks)

        messages = result["messages"]
        assert len(messages) == 1
        tc = messages[0]["tool_calls"][0]
        assert tc["id"] == "call_7hIJuqbG8ixthENF1uc9NLZ1"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city":"Paris"}'

    def test_accumulates_multiple_parallel_tool_calls(self) -> None:
        chunks = [
            _chunk(
                tool_calls=[
                    _tool_call(
                        index=0, id="call_1", type="function", name="get_weather", arguments=""
                    ),
                    _tool_call(
                        index=1, id="call_2", type="function", name="get_time", arguments=""
                    ),
                ]
            ),
            _chunk(
                tool_calls=[
                    _tool_call(index=0, arguments='{"city":"Paris"}'),
                    _tool_call(index=1, arguments='{"tz":"UTC"}'),
                ]
            ),
        ]
        result = _parse_model_output_stream(chunks)

        tool_calls = result["messages"][0]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["arguments"] == '{"city":"Paris"}'
        assert tool_calls[1]["function"]["arguments"] == '{"tz":"UTC"}'

    def test_preserves_content_alongside_tool_calls(self) -> None:
        chunks = [
            _chunk(
                content="Sure, ",
                tool_calls=[
                    _tool_call(index=0, id="call_x", type="function", name="search", arguments="")
                ],
            ),
            _chunk(
                content="let me check.", tool_calls=[_tool_call(index=0, arguments='{"q":"foo"}')]
            ),
        ]
        result = _parse_model_output_stream(chunks)

        msg = result["messages"][0]
        assert msg["content"] == "Sure, let me check."
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"q":"foo"}'


class TestAnthropicNonIndexedStreaming:
    def test_captures_real_anthropic_tool_call(self) -> None:
        # Anthropic SDK fires one complete dict per tool call — no index field
        chunks = [
            _chunk(),
            _chunk(),
            _chunk(),
            _chunk(
                tool_calls=[
                    _tool_call(
                        id="toolu_01BtvBNNZFr68ucnjFmepsf8",
                        type="function",
                        name="get_weather",
                        arguments='{"city": "Paris"}',
                    )
                ]
            ),
            _chunk(),
        ]

        result = _parse_model_output_stream(chunks)

        messages = result["messages"]
        assert len(messages) == 1
        tc = messages[0]["tool_calls"][0]
        assert tc["id"] == "toolu_01BtvBNNZFr68ucnjFmepsf8"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "Paris"}'


class TestGeminiNonIndexedStreaming:
    def test_captures_real_gemini_tool_call(self) -> None:
        # Gemini sends complete functionCall in single chunk — no index field
        chunks = [
            _chunk(
                tool_calls=[
                    _tool_call(
                        id="get_weather_0",
                        type="function",
                        name="get_weather",
                        arguments='{"city": "Paris"}',
                    )
                ]
            ),
        ]

        result = _parse_model_output_stream(chunks)

        messages = result["messages"]
        assert len(messages) == 1
        tc = messages[0]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "Paris"}'


class TestEmptyStream:
    def test_returns_empty_messages_for_empty_stream(self) -> None:
        result = _parse_model_output_stream([])
        assert result == {"messages": []}

    def test_returns_empty_messages_when_no_content_or_tool_calls(self) -> None:
        chunks = [_chunk(content="", tool_calls=None)]
        result = _parse_model_output_stream(chunks)
        assert result == {"messages": []}


class TestStreamOutputMessages:
    def test_emits_role_and_content_attributes(self) -> None:
        output = {"messages": [{"role": "assistant", "content": "Hello!"}]}
        attrs = dict(_stream_output_messages(output))

        assert attrs["llm.output_messages.0.message.role"] == "assistant"
        assert attrs["llm.output_messages.0.message.content"] == "Hello!"

    def test_emits_tool_call_attributes_for_streaming_span(self) -> None:
        output = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                }
            ]
        }
        attrs = dict(_stream_output_messages(output))

        assert attrs["llm.output_messages.0.message.role"] == "assistant"
        assert attrs["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == "call_abc123"
        assert (
            attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"]
            == "get_weather"
        )
        assert (
            attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]
            == '{"city": "Paris"}'
        )

    def test_emits_attributes_for_parallel_tool_calls(self) -> None:
        # Two parallel tool calls (Paris + New Delhi) — each must have its own indexed attributes.
        output = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        },
                        {
                            "id": "call_2",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"New Delhi"}',
                            },
                        },
                    ],
                }
            ]
        }
        attrs = dict(_stream_output_messages(output))

        assert attrs["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == "call_1"
        assert (
            attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]
            == '{"city":"Paris"}'
        )
        assert attrs["llm.output_messages.0.message.tool_calls.1.tool_call.id"] == "call_2"
        assert (
            attrs["llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments"]
            == '{"city":"New Delhi"}'
        )

    def test_no_attributes_for_empty_stream(self) -> None:
        attrs = dict(_stream_output_messages({"messages": []}))
        assert attrs == {}

    def test_content_captured_from_empty_string_start_chunks(self) -> None:
        chunks = [
            _chunk(content=""),  # OpenAI first-chunk empty content
            _chunk(content="The weather"),
            _chunk(content=" is sunny."),
            _chunk(),  # finish chunk — content="" is excluded by the is not None and != "" guard
        ]
        result = _parse_model_output_stream(chunks)
        assert result["messages"][0]["content"] == "The weather is sunny."

    def test_content_in_output_messages_attributes_after_streaming(self) -> None:
        chunks = [
            _chunk(content="The weather is sunny in Paris."),
        ]
        output = _parse_model_output_stream(chunks)
        attrs = dict(_stream_output_messages(output))

        assert attrs["llm.output_messages.0.message.role"] == "assistant"
        assert attrs["llm.output_messages.0.message.content"] == "The weather is sunny in Paris."


class TestToolCallIdInputMessages:
    def _make_tool_message(self, tool_call_id: str, content: str) -> SimpleNamespace:
        return SimpleNamespace(
            role="tool",
            tool_call_id=tool_call_id,
            content=content,
            tool_calls=None,
            get_content_string=lambda: content,
        )

    def _make_arguments(self, messages: list) -> dict:  # type: ignore[type-arg]
        return {"messages": messages, "tools": []}

    def test_tool_call_id_emitted_for_tool_result_message(self) -> None:
        # Regression: role=tool messages carry tool_call_id to link back to the original
        # tool call. This was never emitted — MESSAGE_TOOL_CALL_ID was missing entirely.
        msg = self._make_tool_message("call_abc123", "Sunny, 22°C in Paris.")
        attrs = dict(_llm_input_messages(self._make_arguments([msg])))

        assert attrs["llm.input_messages.0.message.role"] == "tool"
        assert attrs["llm.input_messages.0.message.tool_call_id"] == "call_abc123"
        assert attrs["llm.input_messages.0.message.content"] == "Sunny, 22°C in Paris."

    def test_tool_call_id_absent_for_non_tool_messages(self) -> None:
        msg = SimpleNamespace(
            role="user",
            tool_call_id=None,
            content="What is the weather?",
            tool_calls=None,
            get_content_string=lambda: "What is the weather?",
        )
        attrs = dict(_llm_input_messages(self._make_arguments([msg])))

        assert "llm.input_messages.0.message.tool_call_id" not in attrs
        assert attrs["llm.input_messages.0.message.content"] == "What is the weather?"
