from types import SimpleNamespace
from typing import Any

from openinference.instrumentation.agno._model_wrapper import _parse_model_output_stream


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
            _chunk(tool_calls=[_tool_call(index=0, id="call_7hIJuqbG8ixthENF1uc9NLZ1", type="function", name="get_weather", arguments="")]),
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
            _chunk(tool_calls=[
                _tool_call(index=0, id="call_1", type="function", name="get_weather", arguments=""),
                _tool_call(index=1, id="call_2", type="function", name="get_time", arguments=""),
            ]),
            _chunk(tool_calls=[
                _tool_call(index=0, arguments='{"city":"Paris"}'),
                _tool_call(index=1, arguments='{"tz":"UTC"}'),
            ]),
        ]
        result = _parse_model_output_stream(chunks)

        tool_calls = result["messages"][0]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["arguments"] == '{"city":"Paris"}'
        assert tool_calls[1]["function"]["arguments"] == '{"tz":"UTC"}'

    def test_preserves_content_alongside_tool_calls(self) -> None:
        chunks = [
            _chunk(content="Sure, ", tool_calls=[_tool_call(index=0, id="call_x", type="function", name="search", arguments="")]),
            _chunk(content="let me check.", tool_calls=[_tool_call(index=0, arguments='{"q":"foo"}')]),
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
            _chunk(tool_calls=[
                _tool_call(
                    id="toolu_01BtvBNNZFr68ucnjFmepsf8",
                    type="function",
                    name="get_weather",
                    arguments='{"city": "Paris"}',
                )
            ]),
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
            _chunk(tool_calls=[
                _tool_call(
                    id="get_weather_0",
                    type="function",
                    name="get_weather",
                    arguments='{"city": "Paris"}',
                )
            ]),
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
