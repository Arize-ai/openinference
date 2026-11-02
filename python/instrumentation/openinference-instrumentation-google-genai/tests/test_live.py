"""Tests for Gemini Live session tracing, with a scripted websocket in place of the server."""

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Union

import pytest
import websockets
from google import genai
from google.genai import live, types
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import (
    TraceConfig,
    suppress_tracing,
    using_attributes,
    using_session,
)
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

MODEL = "gemini-2.5-flash-native-audio-latest"
AUDIO_DATA = "UklGRiQAAABXQVZF"  # base64 placeholder; must never show up on a span

TOOLS: types.ToolListUnion = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_weather",
                description="Get weather for a city",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"city": types.Schema(type=types.Type.STRING)},
                    required=["city"],
                ),
            )
        ]
    )
]


class FakeWebSocket:
    """Plays server frames pushed by the test; frames are mldev-format JSON."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.send_error: Optional[BaseException] = None
        self.close_error: Optional[BaseException] = None
        self._frames: "asyncio.Queue[Union[dict[str, Any], BaseException]]" = asyncio.Queue()
        self._frames.put_nowait({"setupComplete": {}})

    def push(self, *frames: Union[dict[str, Any], BaseException]) -> None:
        for frame in frames:
            self._frames.put_nowait(frame)

    async def send(self, data: str) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(json.loads(data))

    async def recv(self, decode: bool = True) -> bytes:
        frame = await self._frames.get()
        if isinstance(frame, BaseException):
            raise frame
        return json.dumps(frame).encode()

    async def close(self) -> None:
        if self.close_error is not None:
            raise self.close_error
        # a pending recv fails once the socket is closed
        self.push(websockets.exceptions.ConnectionClosedOK(None, None))


@pytest.fixture
def websocket(monkeypatch: pytest.MonkeyPatch) -> FakeWebSocket:
    ws = FakeWebSocket()

    @asynccontextmanager
    async def fake_connect(*args: Any, **kwargs: Any) -> AsyncIterator[FakeWebSocket]:
        yield ws

    monkeypatch.setattr(live, "ws_connect", fake_connect)
    return ws


@pytest.fixture
def client() -> genai.Client:
    return genai.Client(api_key="test-key")


@pytest.fixture
def instrument(tracer_provider: TracerProvider) -> Iterator[GoogleGenAIInstrumentor]:
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield instrumentor
    instrumentor.uninstrument()


def _text(text: str, thought: bool = False) -> dict[str, Any]:
    part: dict[str, Any] = {"text": text}
    if thought:
        part["thought"] = True
    return {"serverContent": {"modelTurn": {"parts": [part]}}}


def _user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def _audio() -> dict[str, Any]:
    return {
        "serverContent": {
            "modelTurn": {
                "parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": AUDIO_DATA}}]
            }
        }
    }


def _turn_complete(**usage: Any) -> dict[str, Any]:
    frame: dict[str, Any] = {"serverContent": {"turnComplete": True}}
    if usage:
        frame["usageMetadata"] = usage
    return frame


def _tool_call(call_id: str = "call-1", city: str = "Paris") -> dict[str, Any]:
    return {
        "toolCall": {
            "functionCalls": [{"id": call_id, "name": "get_weather", "args": {"city": city}}]
        }
    }


async def _drain(session: live.AsyncSession) -> list[types.LiveServerMessage]:
    return [message async for message in session.receive()]


def _spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    # a CHAIN and its first LLM can share a start time; the parent goes first
    return sorted(
        exporter.get_finished_spans(),
        key=lambda span: (span.start_time or 0, span.parent is not None),
    )


def _kind(span: ReadableSpan) -> Optional[str]:
    return (span.attributes or {}).get(SpanAttributes.OPENINFERENCE_SPAN_KIND)  # type: ignore[return-value]


def _is_child(child: ReadableSpan, parent: ReadableSpan) -> bool:
    return child.parent is not None and child.parent.span_id == parent.context.span_id


async def test_text_interaction(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.TEXT],
        system_instruction="You are terse.",
        temperature=0.2,
        session_resumption=types.SessionResumptionConfig(handle="resume-handle"),
        tools=TOOLS,
    )
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text="Say hi.")])
        )
        websocket.push(
            _text("Thinking about it.", thought=True),
            _text("Hi"),
            _text(" there."),
            _turn_complete(
                promptTokenCount=10,
                responseTokenCount=4,
                thoughtsTokenCount=3,
                totalTokenCount=14,
                promptTokensDetails=[{"modality": "TEXT", "tokenCount": 10}],
                responseTokensDetails=[{"modality": "TEXT", "tokenCount": 4}],
            ),
        )
        await _drain(session)

    chain, llm = _spans(in_memory_span_exporter)
    assert _kind(chain) == OpenInferenceSpanKindValues.CHAIN.value
    assert chain.name == "AsyncSession"
    assert chain.parent is None
    assert chain.status.status_code is trace_api.StatusCode.OK
    assert dict(chain.attributes or {}) == {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "CHAIN",
        SpanAttributes.INPUT_VALUE: "Say hi.",
        SpanAttributes.INPUT_MIME_TYPE: "text/plain",
        SpanAttributes.OUTPUT_VALUE: "Hi there.",
        SpanAttributes.OUTPUT_MIME_TYPE: "text/plain",
    }

    assert _is_child(llm, chain)
    assert llm.name == "AsyncSession.generate"
    assert llm.status.status_code is trace_api.StatusCode.OK
    attributes = dict(llm.attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "google"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == MODEL
    assert json.loads(str(attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS))) == {
        "response_modalities": ["TEXT"],
        "temperature": 0.2,
    }
    assert json.loads(str(attributes.pop(f"{SpanAttributes.LLM_TOOLS}.0.tool.json_schema"))) == {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "OBJECT",
            "properties": {"city": {"type": "STRING"}},
            "required": ["city"],
        },
    }
    assert json.loads(str(attributes.pop(SpanAttributes.INPUT_VALUE))) == {
        "contents": [{"parts": [{"text": "Say hi."}], "role": "user"}]
    }
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    input_prefix = SpanAttributes.LLM_INPUT_MESSAGES
    assert attributes.pop(f"{input_prefix}.0.{MessageAttributes.MESSAGE_ROLE}") == "system"
    assert attributes.pop(f"{input_prefix}.0.{MessageAttributes.MESSAGE_CONTENT}") == (
        "You are terse."
    )
    assert attributes.pop(f"{input_prefix}.1.{MessageAttributes.MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{input_prefix}.1.{MessageAttributes.MESSAGE_CONTENT}") == "Say hi."
    assert json.loads(str(attributes.pop(SpanAttributes.OUTPUT_VALUE))) == {
        "parts": [{"text": "Thinking about it.", "thought": True}, {"text": "Hi there."}],
        "role": "model",
    }
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    output_prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    contents = f"{output_prefix}.{MessageAttributes.MESSAGE_CONTENTS}"
    assert attributes.pop(f"{output_prefix}.{MessageAttributes.MESSAGE_ROLE}") == "model"
    assert attributes.pop(f"{contents}.0.message_content.type") == "reasoning"
    assert attributes.pop(f"{contents}.0.message_content.text") == "Thinking about it."
    assert attributes.pop(f"{contents}.1.message_content.type") == "text"
    assert attributes.pop(f"{contents}.1.message_content.text") == "Hi there."
    # Live calls output tokens "response" tokens; thoughts count toward completion
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 7
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING) == 3
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 17
    assert attributes == {}


async def test_tool_round_trip(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(
        model=MODEL,
        config=types.LiveConnectConfig(response_modalities=[types.Modality.TEXT], tools=TOOLS),
    ) as session:
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": "Weather in Paris?"}]}
        )
        websocket.push(_tool_call())
        async for message in session.receive():
            if message.tool_call:
                await session.send_tool_response(
                    function_responses={
                        "id": "call-1",
                        "name": "get_weather",
                        "response": {"result": "sunny"},
                    }
                )
                websocket.push(
                    _text("It is sunny."),
                    _turn_complete(promptTokenCount=20, responseTokenCount=5, totalTokenCount=25),
                )

    chain, first, tool, second = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in (chain, first, tool, second)] == ["CHAIN", "LLM", "TOOL", "LLM"]
    assert all(_is_child(span, chain) for span in (first, tool, second))
    assert all(
        span.status.status_code is trace_api.StatusCode.OK for span in (chain, first, tool, second)
    )
    assert (chain.attributes or {})[SpanAttributes.OUTPUT_VALUE] == "It is sunny."

    first_attributes = dict(first.attributes or {})
    tool_call = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert first_attributes[f"{tool_call}.{ToolCallAttributes.TOOL_CALL_ID}"] == "call-1"
    assert first_attributes[f"{tool_call}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == (
        "get_weather"
    )
    assert json.loads(
        str(first_attributes[f"{tool_call}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"])
    ) == {"city": "Paris"}
    # usage arrives once, with the end of the model turn, so the tokens land on the
    # generation that finished the turn
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in first_attributes

    assert tool.name == "get_weather"
    assert dict(tool.attributes or {}) == {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
        SpanAttributes.TOOL_NAME: "get_weather",
        SpanAttributes.TOOL_ID: "call-1",
        SpanAttributes.INPUT_VALUE: '{"city": "Paris"}',
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.OUTPUT_VALUE: '{"result": "sunny"}',
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
    }

    second_attributes = dict(second.attributes or {})
    messages = SpanAttributes.LLM_INPUT_MESSAGES
    assert second_attributes[f"{messages}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    assert second_attributes[f"{messages}.1.{MessageAttributes.MESSAGE_ROLE}"] == "model"
    assert (
        second_attributes[
            f"{messages}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_ID}"
        ]
        == "call-1"
    )
    assert second_attributes[f"{messages}.2.{MessageAttributes.MESSAGE_ROLE}"] == "tool"
    assert second_attributes[f"{messages}.2.{MessageAttributes.MESSAGE_TOOL_CALL_ID}"] == "call-1"
    assert second_attributes[f"{messages}.2.{MessageAttributes.MESSAGE_CONTENT}"] == (
        '{"result": "sunny"}'
    )
    assert (
        second_attributes[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
        ]
        == "It is sunny."
    )
    assert second_attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 25


async def test_audio_is_recorded_as_transcripts_only(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(
        model=MODEL, config=types.LiveConnectConfig(response_modalities=[types.Modality.AUDIO])
    ) as session:
        await session.send_realtime_input(
            audio=types.Blob(data=b"\x00\x01" * 8, mime_type="audio/pcm;rate=16000")
        )
        await session.send_realtime_input(audio_stream_end=True)
        websocket.push(
            {"serverContent": {"inputTranscription": {"text": "What time"}}},
            {"serverContent": {"inputTranscription": {"text": " is it?"}}},
            _audio(),
            {"serverContent": {"outputTranscription": {"text": "It is noon."}}},
            _audio(),
            _turn_complete(
                promptTokenCount=30,
                responseTokenCount=12,
                totalTokenCount=42,
                promptTokensDetails=[{"modality": "AUDIO", "tokenCount": 30}],
                responseTokensDetails=[{"modality": "AUDIO", "tokenCount": 12}],
            ),
        )
        await _drain(session)

    chain, llm = _spans(in_memory_span_exporter)
    assert (chain.attributes or {})[SpanAttributes.INPUT_VALUE] == "What time is it?"
    assert (chain.attributes or {})[SpanAttributes.OUTPUT_VALUE] == "It is noon."
    attributes = dict(llm.attributes or {})
    messages = SpanAttributes.LLM_INPUT_MESSAGES
    assert attributes[f"{messages}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    assert attributes[f"{messages}.0.{MessageAttributes.MESSAGE_CONTENT}"] == "What time is it?"
    output = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes[f"{output}.{MessageAttributes.MESSAGE_CONTENT}"] == "It is noon."
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO] == 30
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO] == 12
    for span in (chain, llm):
        for value in (span.attributes or {}).values():
            assert AUDIO_DATA not in str(value)
            assert "audio/pcm" not in str(value)


async def test_each_interaction_is_its_own_chain(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with using_session("my-session"):
        async with client.aio.live.connect(model=MODEL) as session:
            for prompt in ("one", "two"):
                await session.send_client_content(turns=_user(prompt))
                websocket.push(_text(f"reply to {prompt}"), _turn_complete())
                await _drain(session)

    spans = _spans(in_memory_span_exporter)
    chains = [span for span in spans if _kind(span) == "CHAIN"]
    assert [(chain.attributes or {})[SpanAttributes.INPUT_VALUE] for chain in chains] == [
        "one",
        "two",
    ]
    assert all(chain.parent is None for chain in chains)
    assert all((span.attributes or {})[SpanAttributes.SESSION_ID] == "my-session" for span in spans)
    second_llm = spans[-1]
    assert _is_child(second_llm, chains[1])
    # only the current interaction is replayed as input; the server keeps the rest
    messages = SpanAttributes.LLM_INPUT_MESSAGES
    assert (second_llm.attributes or {})[f"{messages}.0.{MessageAttributes.MESSAGE_CONTENT}"] == (
        "two"
    )
    assert f"{messages}.1.{MessageAttributes.MESSAGE_ROLE}" not in (second_llm.attributes or {})


async def test_provider_session_id_and_parent_from_connect_context(
    instrument: GoogleGenAIInstrumentor,
    tracer_provider: TracerProvider,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    websocket._frames.get_nowait()
    websocket.push({"setupComplete": {"sessionId": "provider-session"}})
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("app"):
        async with client.aio.live.connect(model=MODEL) as session:

            async def receive_in_another_task() -> None:
                await _drain(session)

            receiver = asyncio.create_task(receive_in_another_task())
            await session.send_client_content(turns=_user("hi"))
            websocket.push(_text("hello"), _turn_complete())
            await receiver

    app, chain, llm = _spans(in_memory_span_exporter)
    assert app.name == "app"
    assert _is_child(chain, app)
    assert _is_child(llm, chain)
    assert (chain.attributes or {})[SpanAttributes.SESSION_ID] == "provider-session"
    assert (llm.attributes or {})[SpanAttributes.SESSION_ID] == "provider-session"


@pytest.mark.skipif(
    "interaction_status" not in types.LiveServerContent.model_fields,
    reason="interaction_status is not in this google-genai version",
)
async def test_interaction_status_keeps_chain_open_across_model_turns(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("plan a trip"))
        websocket.push(
            _text("Step one."),
            {"serverContent": {"turnComplete": True, "interactionStatus": "IN_PROGRESS"}},
            _text("Step two."),
            {"serverContent": {"turnComplete": True, "interactionStatus": "IDLE"}},
        )
        async for message in session.receive():
            pass

    chain, first, second = _spans(in_memory_span_exporter)
    assert _kind(chain) == "CHAIN"
    assert _is_child(first, chain) and _is_child(second, chain)
    assert (chain.attributes or {})[SpanAttributes.OUTPUT_VALUE] == "Step two."


async def test_tool_call_cancellation_ends_tool_span(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(
            _tool_call(),
            {"toolCallCancellation": {"ids": ["call-1"]}},
            _text("Never mind."),
            _turn_complete(),
        )
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    (tool,) = [span for span in spans if _kind(span) == "TOOL"]
    assert tool.status.status_code is trace_api.StatusCode.UNSET
    assert [event.name for event in tool.events] == ["cancelled"]
    assert SpanAttributes.OUTPUT_VALUE not in (tool.attributes or {})


async def test_close_mid_interaction_ends_spans_without_status(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("long story"))
        websocket.push(_text("Once upon"))
        await session._receive()
        await session.close()

    chain, llm = _spans(in_memory_span_exporter)
    assert chain.status.status_code is trace_api.StatusCode.UNSET
    assert llm.status.status_code is trace_api.StatusCode.UNSET
    output = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
    assert (llm.attributes or {})[output] == "Once upon"


async def test_exit_without_close_ends_open_spans(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("hello"))

    chain, llm = _spans(in_memory_span_exporter)
    assert chain.status.status_code is trace_api.StatusCode.UNSET
    assert llm.status.status_code is trace_api.StatusCode.UNSET


async def test_receive_error_marks_spans_as_errors(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with pytest.raises(RuntimeError, match="socket dropped"):
        async with client.aio.live.connect(model=MODEL) as session:
            await session.send_client_content(turns=_user("hello"))
            websocket.push(_text("Hel"), RuntimeError("socket dropped"))
            await _drain(session)

    chain, llm = _spans(in_memory_span_exporter)
    for span in (chain, llm):
        assert span.status.status_code is trace_api.StatusCode.ERROR
        assert span.status.description == "RuntimeError: socket dropped"
        assert [event.name for event in span.events] == ["exception"]


async def test_suppress_tracing(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with suppress_tracing():
        async with client.aio.live.connect(model=MODEL) as session:
            await session.send_client_content(turns=_user("hello"))
            websocket.push(_text("hi"), _turn_complete())
            await _drain(session)

    assert not in_memory_span_exporter.get_finished_spans()


async def test_hide_inputs_and_outputs(
    tracer_provider: TracerProvider,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
    )
    try:
        async with client.aio.live.connect(
            model=MODEL, config={"system_instruction": "secret system"}
        ) as session:
            await session.send_client_content(turns=_user("secret question"))
            websocket.push(_text("secret answer"), _turn_complete())
            await _drain(session)
    finally:
        instrumentor.uninstrument()

    spans = _spans(in_memory_span_exporter)
    assert len(spans) == 2
    for span in spans:
        for key, value in (span.attributes or {}).items():
            assert not key.startswith(SpanAttributes.LLM_INPUT_MESSAGES)
            assert not key.startswith(SpanAttributes.LLM_OUTPUT_MESSAGES)
            assert "secret" not in str(value) or value == "__REDACTED__", key


async def test_uninstrument_restores_live_methods(
    tracer_provider: TracerProvider,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    originals = {
        name: live.AsyncSession.__dict__[name]
        for name in (
            "send_client_content",
            "send_realtime_input",
            "send_tool_response",
            "send",
            "_receive",
            "close",
        )
    }
    original_connect = live.AsyncLive.__dict__["connect"]
    original_connect_config = live._t_live_connect_config
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    assert live.AsyncLive.__dict__["connect"] is not original_connect
    instrumentor.uninstrument()

    assert live.AsyncLive.__dict__["connect"] is original_connect
    assert live._t_live_connect_config is original_connect_config
    for name, original in originals.items():
        assert live.AsyncSession.__dict__[name] is original
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("hello"))
        websocket.push(_turn_complete())
        await _drain(session)
    assert not in_memory_span_exporter.get_finished_spans()


async def test_audio_in_client_content_is_not_recorded(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[
                    types.Part(text="Transcribe this."),
                    types.Part(inline_data=types.Blob(data=b"\x00\x01" * 8, mime_type="audio/wav")),
                ],
            )
        )
        websocket.push(_text("Done."), _turn_complete())
        await _drain(session)

    chain, llm = _spans(in_memory_span_exporter)
    assert json.loads(str((llm.attributes or {})[SpanAttributes.INPUT_VALUE])) == {
        "contents": [{"parts": [{"text": "Transcribe this."}], "role": "user"}]
    }
    assert (chain.attributes or {})[SpanAttributes.INPUT_VALUE] == "Transcribe this."
    for span in (chain, llm):
        assert not any("audio/wav" in str(value) for value in (span.attributes or {}).values())


async def test_usage_without_open_generation_does_not_start_a_span(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(
            _tool_call(),
            {"usageMetadata": {"promptTokenCount": 7, "totalTokenCount": 7}},
            # nothing is left to attach this one to, so it waits for the next generation
            {"usageMetadata": {"promptTokenCount": 8, "totalTokenCount": 8}},
        )
        await session._receive()
        await session._receive()
        await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1", name="get_weather", response={"result": "sunny"}
            )
        )
        websocket.push(
            _text("Sunny."),
            _turn_complete(),
            {"usageMetadata": {"promptTokenCount": 9, "totalTokenCount": 9}},
        )
        await _drain(session)
        await session._receive()

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL", "LLM"]
    first, second = spans[1], spans[3]
    assert (first.attributes or {})[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 7
    assert (second.attributes or {})[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 8


async def test_tool_calls_without_ids_match_by_name(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    # Vertex AI does not require ids on function calls and responses
    client = genai.Client(vertexai=True, api_key="test-key")
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(
            {"toolCall": {"functionCalls": [{"name": "get_weather", "args": {"city": "Rome"}}]}}
        )
        await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                name="get_weather", response={"result": "warm"}
            )
        )
        websocket.push(_text("Warm."), _turn_complete())
        await _drain(session)

    (tool,) = [span for span in _spans(in_memory_span_exporter) if _kind(span) == "TOOL"]
    assert tool.status.status_code is trace_api.StatusCode.OK
    assert (tool.attributes or {})[SpanAttributes.OUTPUT_VALUE] == '{"result": "warm"}'
    assert SpanAttributes.TOOL_ID not in (tool.attributes or {})


async def test_failed_tool_response_marks_tool_span_as_error(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with pytest.raises(RuntimeError, match="send failed"):
        async with client.aio.live.connect(model=MODEL) as session:
            await session.send_client_content(turns=_user("weather?"))
            websocket.push(_tool_call())
            await session._receive()
            websocket.send_error = RuntimeError("send failed")
            await session.send_tool_response(
                function_responses=types.FunctionResponse(
                    id="call-1", name="get_weather", response={"result": "sunny"}
                )
            )

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL", "LLM"]
    tool = spans[2]
    assert tool.status.status_code is trace_api.StatusCode.ERROR
    assert tool.status.description == "RuntimeError: send failed"
    assert all(
        span.status.status_code is trace_api.StatusCode.ERROR for span in (spans[0], spans[3])
    )
    assert spans[1].status.status_code is trace_api.StatusCode.OK


async def test_close_while_another_task_receives_is_not_an_error(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:

        async def receive_forever() -> None:
            while True:
                await session._receive()

        await session.send_client_content(turns=_user("long story"))
        receiver = asyncio.create_task(receive_forever())
        websocket.push(_text("Once upon"))
        await asyncio.sleep(0)
        await session.close()
        with pytest.raises(Exception):
            await receiver

    chain, llm = _spans(in_memory_span_exporter)
    assert chain.status.status_code is trace_api.StatusCode.UNSET
    assert llm.status.status_code is trace_api.StatusCode.UNSET


async def test_close_failure_marks_spans_as_errors(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("hello"))
        websocket.close_error = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="close failed"):
            await session.close()

    chain, llm = _spans(in_memory_span_exporter)
    for span in (chain, llm):
        assert span.status.status_code is trace_api.StatusCode.ERROR
        assert span.status.description == "RuntimeError: close failed"


async def test_response_with_unknown_id_does_not_end_another_call(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call("call-1"))
        await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="unknown", name="get_weather", response={"result": "?"}
            )
        )
        assert not [
            span for span in in_memory_span_exporter.get_finished_spans() if _kind(span) == "TOOL"
        ]

    (tool,) = [span for span in _spans(in_memory_span_exporter) if _kind(span) == "TOOL"]
    assert tool.status.status_code is trace_api.StatusCode.UNSET
    assert SpanAttributes.OUTPUT_VALUE not in (tool.attributes or {})


async def test_close_failure_after_concurrent_receive_failure_is_an_error(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("hello"))
        receiver = asyncio.create_task(session._receive())
        await asyncio.sleep(0)
        original_close = websocket.close

        async def close_then_fail() -> None:
            await original_close()  # fails the pending receive
            await asyncio.sleep(0)
            raise RuntimeError("close failed")

        websocket.close = close_then_fail  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="close failed"):
            await session.close()
        with pytest.raises(Exception):
            await receiver

    chain, llm = _spans(in_memory_span_exporter)
    for span in (chain, llm):
        assert span.status.status_code is trace_api.StatusCode.ERROR
        assert span.status.description == "RuntimeError: close failed"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_deprecated_send(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send(input="Weather in Paris?", end_of_turn=True)
        websocket.push(_tool_call())
        await session._receive()
        await session.send(
            input=types.FunctionResponse(
                id="call-1", name="get_weather", response={"result": "sunny"}
            )
        )
        websocket.push(_text("Sunny."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL", "LLM"]
    chain, first, tool, _ = spans
    assert (chain.attributes or {})[SpanAttributes.INPUT_VALUE] == "Weather in Paris?"
    messages = SpanAttributes.LLM_INPUT_MESSAGES
    assert (first.attributes or {})[f"{messages}.0.{MessageAttributes.MESSAGE_CONTENT}"] == (
        "Weather in Paris?"
    )
    assert tool.status.status_code is trace_api.StatusCode.OK
    assert (tool.attributes or {})[SpanAttributes.OUTPUT_VALUE] == '{"result": "sunny"}'


async def test_partial_tool_responses_keep_tool_span_open(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("search"))
        websocket.push(_tool_call())
        await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1", name="get_weather", response={"partial": 1}, will_continue=True
            )
        )
        assert not [
            span for span in in_memory_span_exporter.get_finished_spans() if _kind(span) == "TOOL"
        ]
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1", name="get_weather", response={"final": 2}
            )
        )
        websocket.push(_text("Done."), _turn_complete())
        await _drain(session)

    (tool,) = [span for span in _spans(in_memory_span_exporter) if _kind(span) == "TOOL"]
    assert tool.status.status_code is trace_api.StatusCode.OK
    assert (tool.attributes or {})[SpanAttributes.OUTPUT_VALUE] == '{"final": 2}'


async def test_close_ends_tool_span_while_its_response_is_still_sending(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call())
        await session._receive()
        stuck = asyncio.Event()
        original_send = websocket.send

        async def blocked_send(data: str) -> None:
            await stuck.wait()
            await original_send(data)

        websocket.send = blocked_send  # type: ignore[method-assign]
        sender = asyncio.create_task(
            session.send_tool_response(
                function_responses=types.FunctionResponse(
                    id="call-1", name="get_weather", response={"result": "sunny"}
                )
            )
        )
        await asyncio.sleep(0)
        await session.close()
        assert [_kind(span) for span in _spans(in_memory_span_exporter)] == [
            "CHAIN",
            "LLM",
            "TOOL",
            "LLM",
        ]
        sender.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sender

    tool = _spans(in_memory_span_exporter)[2]
    assert tool.status.status_code is trace_api.StatusCode.UNSET
    assert (tool.attributes or {})[SpanAttributes.OUTPUT_VALUE] == '{"result": "sunny"}'


async def test_session_is_traced_normally_after_a_failed_close(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with pytest.raises(RuntimeError, match="socket dropped"):
        async with client.aio.live.connect(model=MODEL) as session:
            websocket.close_error = RuntimeError("close failed")
            with pytest.raises(RuntimeError, match="close failed"):
                await session.close()
            await session.send_client_content(turns=_user("still there?"))
            websocket.push(RuntimeError("socket dropped"))
            await session._receive()

    chain, llm = _spans(in_memory_span_exporter)
    for span in (chain, llm):
        assert span.status.status_code is trace_api.StatusCode.ERROR
        assert span.status.description == "RuntimeError: socket dropped"


async def test_interaction_completing_during_tool_response_send_ends_tool_ok(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call())
        await session._receive()
        release = asyncio.Event()
        original_send = websocket.send

        async def slow_send(data: str) -> None:
            # the server gets the response and answers before send() returns
            await original_send(data)
            websocket.push(_text("Sunny."), _turn_complete())
            await release.wait()

        websocket.send = slow_send  # type: ignore[method-assign]
        sender = asyncio.create_task(
            session.send_tool_response(
                function_responses=types.FunctionResponse(
                    id="call-1", name="get_weather", response={"result": "sunny"}
                )
            )
        )
        await asyncio.sleep(0)
        await _drain(session)
        release.set()
        await sender

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL", "LLM"]
    assert all(span.status.status_code is trace_api.StatusCode.OK for span in spans)


async def test_receive_returning_after_exit_records_nothing(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("hello"))
        websocket.push(_text("hi"), _turn_complete())
        await _drain(session)
        receiver = asyncio.create_task(session._receive())
        await asyncio.sleep(0)
    websocket.push(_text("late"), _turn_complete())
    await receiver

    assert [_kind(span) for span in _spans(in_memory_span_exporter)] == ["CHAIN", "LLM"]


async def test_late_send_failure_does_not_fail_the_next_interaction(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call())
        await session._receive()
        release = asyncio.Event()
        original_send = websocket.send

        async def send_then_fail(data: str) -> None:
            await original_send(data)
            websocket.push(_text("Sunny."), _turn_complete())
            await release.wait()
            raise RuntimeError("late send failure")

        websocket.send = send_then_fail  # type: ignore[method-assign]
        sender = asyncio.create_task(
            session.send_tool_response(
                function_responses=types.FunctionResponse(
                    id="call-1", name="get_weather", response={"result": "sunny"}
                )
            )
        )
        await asyncio.sleep(0)
        await _drain(session)
        websocket.send = original_send  # type: ignore[method-assign]
        await session.send_client_content(turns=_user("and tomorrow?"))
        release.set()
        with pytest.raises(RuntimeError, match="late send failure"):
            await sender
        websocket.push(_text("Rain."), _turn_complete())
        await _drain(session)

    chains = [span for span in _spans(in_memory_span_exporter) if _kind(span) == "CHAIN"]
    assert [chain.status.status_code for chain in chains] == [
        trace_api.StatusCode.OK,
        trace_api.StatusCode.OK,
    ]


async def test_non_blocking_tool_outlives_its_interaction(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    config = types.LiveConnectConfig(
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="get_weather", behavior=types.Behavior.NON_BLOCKING
                    )
                ]
            )
        ]
    )
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call(), _text("Checking, one moment."), _turn_complete())
        await _drain(session)
        # the model kept generating after the call, so it is one generation
        assert [_kind(span) for span in in_memory_span_exporter.get_finished_spans()] == [
            "LLM",
            "CHAIN",
        ]
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1",
                name="get_weather",
                response={"result": "sunny"},
                scheduling=types.FunctionResponseScheduling.SILENT,
            )
        )

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL"]
    chain, llm, tool = spans
    assert _is_child(tool, chain)
    assert tool.status.status_code is trace_api.StatusCode.OK
    assert (tool.attributes or {})[SpanAttributes.OUTPUT_VALUE] == '{"result": "sunny"}'
    output = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    llm_attributes = llm.attributes or {}
    assert (
        llm_attributes[
            f"{output}.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_ID}"
        ]
        == "call-1"
    )
    assert llm_attributes[
        f"{output}.{MessageAttributes.MESSAGE_CONTENTS}.0.message_content.text"
    ] == ("Checking, one moment.")


async def test_streamed_function_call_frames_make_one_tool_span(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(
            {
                "toolCall": {
                    "functionCalls": [{"id": "call-1", "name": "get_weather", "willContinue": True}]
                }
            },
            _tool_call(),
        )
        await session._receive()
        await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1", name="get_weather", response={"result": "sunny"}
            )
        )
        websocket.push(_text("Sunny."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL", "LLM"]
    assert (spans[2].attributes or {})[SpanAttributes.INPUT_VALUE] == '{"city": "Paris"}'
    assert spans[2].status.status_code is trace_api.StatusCode.OK


def get_time(city: str) -> str:
    """Returns the current time in a city."""
    return "noon"


async def test_python_function_tools_are_recorded(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(
        model=MODEL, config=types.LiveConnectConfig(tools=[get_time])
    ) as session:
        await session.send_client_content(turns=_user("time?"))
        websocket.push(_text("Noon."), _turn_complete())
        await _drain(session)

    llm = _spans(in_memory_span_exporter)[1]
    schema = json.loads(
        str((llm.attributes or {})[f"{SpanAttributes.LLM_TOOLS}.0.tool.json_schema"])
    )
    assert schema["name"] == "get_time"
    assert schema["description"] == "Returns the current time in a city."
    assert "city" in json.dumps(schema["parameters"])


NON_BLOCKING_CONFIG = types.LiveConnectConfig(
    tools=[
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(name="get_weather", behavior=types.Behavior.NON_BLOCKING)
            ]
        )
    ]
)


async def test_non_blocking_response_that_prompts_the_model_starts_an_interaction(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL, config=NON_BLOCKING_CONFIG) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call(), _text("Checking."), _turn_complete())
        await _drain(session)
        # WHEN_IDLE (the default for NON_BLOCKING responses) prompts the model
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1", name="get_weather", response={"result": "sunny"}
            )
        )
        websocket.push(_text("It is sunny."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL", "CHAIN", "LLM"]
    second_llm = spans[4]
    assert _is_child(second_llm, spans[3])
    messages = SpanAttributes.LLM_INPUT_MESSAGES
    attributes = second_llm.attributes or {}
    assert attributes[f"{messages}.0.{MessageAttributes.MESSAGE_ROLE}"] == "tool"
    assert attributes[f"{messages}.0.{MessageAttributes.MESSAGE_TOOL_CALL_ID}"] == "call-1"
    assert (spans[3].attributes or {})[SpanAttributes.OUTPUT_VALUE] == "It is sunny."


async def test_cancellation_while_tool_response_is_sending(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call())
        await session._receive()
        release = asyncio.Event()
        original_send = websocket.send

        async def slow_send(data: str) -> None:
            await original_send(data)
            websocket.push({"toolCallCancellation": {"ids": ["call-1"]}})
            await release.wait()

        websocket.send = slow_send  # type: ignore[method-assign]
        sender = asyncio.create_task(
            session.send_tool_response(
                function_responses=types.FunctionResponse(
                    id="call-1", name="get_weather", response={"result": "sunny"}
                )
            )
        )
        await asyncio.sleep(0)
        await session._receive()
        release.set()
        await sender

    (tool,) = [span for span in _spans(in_memory_span_exporter) if _kind(span) == "TOOL"]
    assert [event.name for event in tool.events] == ["cancelled"]
    assert tool.status.status_code is trace_api.StatusCode.UNSET


async def test_failed_silent_background_response_does_not_fail_current_interaction(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL, config=NON_BLOCKING_CONFIG) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call(), _text("Checking."), _turn_complete())
        await _drain(session)
        await session.send_client_content(turns=_user("tell me a joke"))
        websocket.send_error = RuntimeError("send failed")
        with pytest.raises(RuntimeError, match="send failed"):
            await session.send_tool_response(
                function_responses=types.FunctionResponse(
                    id="call-1",
                    name="get_weather",
                    response={"result": "sunny"},
                    scheduling=types.FunctionResponseScheduling.SILENT,
                )
            )
        websocket.send_error = None
        websocket.push(_text("Why did..."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    (tool,) = [span for span in spans if _kind(span) == "TOOL"]
    assert tool.status.status_code is trace_api.StatusCode.ERROR
    chains = [span for span in spans if _kind(span) == "CHAIN"]
    assert [chain.status.status_code for chain in chains] == [
        trace_api.StatusCode.OK,
        trace_api.StatusCode.OK,
    ]


async def test_audio_in_tool_response_parts_is_not_recorded(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("record a sound"))
        websocket.push(_tool_call())
        await session._receive()
        # older SDKs cannot serialize raw bytes in a Live tool response and raise, but the
        # message is recorded before it is sent
        with suppress(TypeError):
            await session.send_tool_response(
                function_responses=types.FunctionResponse(
                    id="call-1",
                    name="get_weather",
                    response={"result": "recorded"},
                    parts=[
                        types.FunctionResponsePart.from_bytes(
                            data=b"RIFFxxxx", mime_type="audio/wav"
                        )
                    ],
                )
            )

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL", "LLM"]
    assert (spans[2].attributes or {})[SpanAttributes.OUTPUT_VALUE] == '{"result": "recorded"}'
    encoded = base64.b64encode(b"RIFFxxxx").decode()
    for span in spans:
        for value in (span.attributes or {}).values():
            assert encoded not in str(value)
            assert "audio/wav" not in str(value)


async def test_tool_response_parts_are_recorded_as_tool_output(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("draw a chart"))
        websocket.push(_tool_call())
        await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1",
                name="get_weather",
                parts=[
                    types.FunctionResponsePart.from_uri(
                        file_uri="gs://bucket/chart.png", mime_type="image/png"
                    )
                ],
            )
        )
        websocket.push(_text("Here it is."), _turn_complete())
        await _drain(session)

    (tool,) = [span for span in _spans(in_memory_span_exporter) if _kind(span) == "TOOL"]
    assert tool.status.status_code is trace_api.StatusCode.OK
    assert json.loads(str((tool.attributes or {})[SpanAttributes.OUTPUT_VALUE])) == {
        "parts": [{"file_data": {"file_uri": "gs://bucket/chart.png", "mime_type": "image/png"}}]
    }


EMBEDDED_PAYLOAD = base64.b64encode(b"RIFFxxxx").decode()


def _assert_payload_not_recorded(spans: Sequence[ReadableSpan]) -> None:
    for span in spans:
        for value in (span.attributes or {}).values():
            assert EMBEDDED_PAYLOAD not in str(value)


@pytest.mark.parametrize(
    "file_uri, mime_type",
    [
        pytest.param(f"data:audio/wav;base64,{EMBEDDED_PAYLOAD}", "audio/wav", id="audio"),
        pytest.param(f"data:video/mp4;base64,{EMBEDDED_PAYLOAD}", "video/mp4", id="video"),
        pytest.param(f"DATA:audio/wav;base64,{EMBEDDED_PAYLOAD}", "audio/wav", id="upper-case"),
        pytest.param(f"data:image/png;base64,{EMBEDDED_PAYLOAD}", "audio/wav", id="wrong-mime"),
        pytest.param(f"data:;base64,{EMBEDDED_PAYLOAD}", "image/png", id="no-media-type"),
    ],
)
async def test_embedded_media_in_tool_response_is_not_recorded(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
    file_uri: str,
    mime_type: str,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("record a sound"))
        websocket.push(_tool_call())
        await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1",
                name="get_weather",
                response={"result": "recorded"},
                parts=[types.FunctionResponsePart.from_uri(file_uri=file_uri, mime_type=mime_type)],
            )
        )
        websocket.push(_text("Done."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    (tool,) = [span for span in spans if _kind(span) == "TOOL"]
    assert (tool.attributes or {})[SpanAttributes.OUTPUT_VALUE] == '{"result": "recorded"}'
    _assert_payload_not_recorded(spans)


async def test_remote_media_uri_in_tool_response_is_recorded(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("record a sound"))
        websocket.push(_tool_call())
        await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1",
                name="get_weather",
                parts=[
                    types.FunctionResponsePart.from_uri(
                        file_uri="gs://bucket/clip.wav", mime_type="audio/wav"
                    )
                ],
            )
        )
        websocket.push(_text("Done."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    (tool,) = [span for span in spans if _kind(span) == "TOOL"]
    assert json.loads(str((tool.attributes or {})[SpanAttributes.OUTPUT_VALUE])) == {
        "parts": [{"file_data": {"file_uri": "gs://bucket/clip.wav", "mime_type": "audio/wav"}}]
    }
    llm = [span for span in spans if _kind(span) == "LLM"][-1]
    assert "gs://bucket/clip.wav" in str((llm.attributes or {})[SpanAttributes.INPUT_VALUE])


async def test_client_content_keeps_remote_media_uri_and_drops_embedded_media(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[
                    types.Part(text="compare these"),
                    types.Part.from_uri(
                        file_uri=f"data:audio/wav;base64,{EMBEDDED_PAYLOAD}",
                        mime_type="audio/wav",
                    ),
                    types.Part.from_uri(file_uri="gs://bucket/clip.mp4", mime_type="video/mp4"),
                ],
            )
        )
        websocket.push(_text("They differ."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    _assert_payload_not_recorded(spans)
    (llm,) = [span for span in spans if _kind(span) == "LLM"]
    assert "gs://bucket/clip.mp4" in str((llm.attributes or {})[SpanAttributes.INPUT_VALUE])


async def test_deprecated_send_drops_embedded_media(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send(
            input=types.LiveClientContent(
                turns=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text="listen"),
                            types.Part.from_uri(
                                file_uri=f"data:audio/wav;base64,{EMBEDDED_PAYLOAD}",
                                mime_type="audio/wav",
                            ),
                        ],
                    )
                ],
                turn_complete=True,
            )
        )
        websocket.push(_text("Heard."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM"]
    _assert_payload_not_recorded(spans)


async def test_audio_interaction_without_transcription(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    config = types.LiveConnectConfig(response_modalities=[types.Modality.AUDIO])
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send_realtime_input(
            audio=types.Blob(data=b"RIFFxxxx", mime_type="audio/pcm;rate=16000")
        )
        # audio input alone does not start a span; the user may just be silent
        assert not in_memory_span_exporter.get_finished_spans()
        websocket.push(
            _audio(),
            _audio(),
            _turn_complete(promptTokenCount=4, responseTokenCount=3, totalTokenCount=7),
        )
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM"]
    chain, llm = spans
    assert _is_child(llm, chain)
    assert chain.status.status_code is trace_api.StatusCode.OK
    assert llm.status.status_code is trace_api.StatusCode.OK
    attributes = llm.attributes or {}
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 7
    assert SpanAttributes.INPUT_VALUE not in attributes
    assert SpanAttributes.OUTPUT_VALUE not in attributes
    for span in spans:
        for value in (span.attributes or {}).values():
            assert AUDIO_DATA not in str(value)


@pytest.mark.parametrize(
    "mime_type",
    ["audio/x-image", "video/x-image", "x-image/png", "IMAGE/PNG", None],
)
async def test_inline_media_is_kept_only_with_an_image_mime_type(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
    mime_type: Optional[str],
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[
                    types.Part(text="listen"),
                    types.Part(inline_data=types.Blob(data=b"RIFFxxxx", mime_type=mime_type)),
                ],
            )
        )
        websocket.push(_text("Heard."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM"]
    _assert_payload_not_recorded(spans)


async def test_inline_image_is_still_recorded(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part(inline_data=types.Blob(data=b"\x89PNG", mime_type="image/png"))],
            )
        )
        websocket.push(_text("A picture."), _turn_complete())
        await _drain(session)

    (llm,) = [span for span in _spans(in_memory_span_exporter) if _kind(span) == "LLM"]
    assert "image/png" in str((llm.attributes or {})[SpanAttributes.INPUT_VALUE])


async def test_audio_interaction_starts_when_the_input_began(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        before_input = time.time_ns()
        await session.send_realtime_input(
            audio=types.Blob(data=b"\x00\x01", mime_type="audio/pcm;rate=16000")
        )
        after_input = time.time_ns()
        # later chunks of the same input do not move the start
        await session.send_realtime_input(
            audio=types.Blob(data=b"\x00\x01", mime_type="audio/pcm;rate=16000")
        )
        websocket.push(_audio(), _turn_complete())
        await _drain(session)

    chain, llm = _spans(in_memory_span_exporter)
    assert (_kind(chain), _kind(llm)) == ("CHAIN", "LLM")
    assert _is_child(llm, chain)
    for span in (chain, llm):
        assert before_input <= (span.start_time or 0) <= after_input


async def test_activity_start_sets_when_the_input_began(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_realtime_input(
            audio=types.Blob(data=b"\x00\x01", mime_type="audio/pcm;rate=16000")
        )
        before_activity = time.time_ns()
        await session.send_realtime_input(activity_start=types.ActivityStart())
        await session.send_realtime_input(activity_end=types.ActivityEnd())
        websocket.push(_audio(), _turn_complete())
        await _drain(session)

    chain, _ = _spans(in_memory_span_exporter)
    assert (chain.start_time or 0) >= before_activity


async def test_failed_audio_send_before_any_response_is_an_error_span(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        websocket.send_error = RuntimeError("send failed")
        with pytest.raises(RuntimeError, match="send failed"):
            await session.send_realtime_input(
                audio=types.Blob(data=b"\x00\x01", mime_type="audio/pcm;rate=16000")
            )
        websocket.send_error = None

    (chain,) = _spans(in_memory_span_exporter)
    assert _kind(chain) == "CHAIN"
    assert chain.status.status_code is trace_api.StatusCode.ERROR
    assert "send failed" in (chain.status.description or "")


async def test_audio_input_that_gets_no_response_makes_no_span(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_realtime_input(
            audio=types.Blob(data=b"\x00\x01", mime_type="audio/pcm;rate=16000")
        )

    assert not in_memory_span_exporter.get_finished_spans()


async def test_text_interaction_ignores_earlier_audio_input(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_realtime_input(
            audio=types.Blob(data=b"\x00\x01", mime_type="audio/pcm;rate=16000")
        )
        before_text = time.time_ns()
        await session.send_client_content(turns=_user("hello"))
        websocket.push(_text("Hi."), _turn_complete())
        await _drain(session)

    chain, llm = _spans(in_memory_span_exporter)
    assert (chain.start_time or 0) >= before_text
    assert (llm.start_time or 0) >= before_text


async def test_reply_to_suppressed_send_is_not_traced(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        with suppress_tracing():
            await session.send_client_content(turns=_user("secret"))
        websocket.push(_text("Secret reply."), _turn_complete(totalTokenCount=5))
        await _drain(session)
        assert not in_memory_span_exporter.get_finished_spans()
        # the next turn is traced as usual
        await session.send_client_content(turns=_user("hello"))
        websocket.push(_text("Hi."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM"]
    assert (spans[0].attributes or {})[SpanAttributes.INPUT_VALUE] == "hello"
    for span in spans:
        assert "Secret" not in str(dict(span.attributes or {}))


async def test_tool_round_trip_of_suppressed_turn_is_not_traced(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        with suppress_tracing():
            await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call())
        await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1", name="get_weather", response={"result": "sunny"}
            )
        )
        websocket.push(_text("It is sunny."), _turn_complete())
        await _drain(session)

    assert not in_memory_span_exporter.get_finished_spans()


async def test_suppressed_receive_finishes_interaction_that_already_started(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("hello"))
        websocket.push(_text("Hi."), _turn_complete(totalTokenCount=5))
        with suppress_tracing():
            await _drain(session)
        spans = _spans(in_memory_span_exporter)
        assert [_kind(span) for span in spans] == ["CHAIN", "LLM"]
        assert spans[0].status.status_code is trace_api.StatusCode.OK
        assert (spans[1].attributes or {})[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 5


async def test_turn_first_received_under_suppression_is_not_traced(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        websocket.push(_audio(), _audio(), _turn_complete())
        with suppress_tracing():
            await session._receive()
        await _drain(session)

    assert not in_memory_span_exporter.get_finished_spans()


async def test_failed_suppressed_send_does_not_suppress_later_turns(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        websocket.send_error = RuntimeError("send failed")
        with suppress_tracing(), pytest.raises(RuntimeError, match="send failed"):
            await session.send_client_content(turns=_user("secret"))
        websocket.send_error = None
        websocket.push(_audio(), _turn_complete())
        await _drain(session)

    assert [_kind(span) for span in _spans(in_memory_span_exporter)] == ["CHAIN", "LLM"]


async def test_silent_suppressed_tool_response_does_not_suppress_later_turns(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL, config=NON_BLOCKING_CONFIG) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call(), _text("Checking."), _turn_complete())
        await _drain(session)
        with suppress_tracing():
            await session.send_tool_response(
                function_responses=types.FunctionResponse(
                    id="call-1",
                    name="get_weather",
                    response={"result": "sunny"},
                    scheduling=types.FunctionResponseScheduling.SILENT,
                )
            )
        websocket.push(_audio(), _turn_complete())
        await _drain(session)

    chains = [span for span in _spans(in_memory_span_exporter) if _kind(span) == "CHAIN"]
    assert len(chains) == 2


_needs_context_update = pytest.mark.skipif(
    not hasattr(live.AsyncSession, "send_context_update"),
    reason="send_context_update is not in this google-genai version",
)


async def _send_context_update(session: Any, **kwargs: Any) -> None:
    await session.send_context_update(**kwargs)


@_needs_context_update
async def test_context_update_applies_to_later_spans(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    config = types.LiveConnectConfig(system_instruction="old", tools=TOOLS)
    new_tools: types.ToolListUnion = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="get_time",
                    description="Get the time",
                    behavior=types.Behavior.NON_BLOCKING,
                )
            ]
        )
    ]
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send_client_content(turns=_user("one"))
        # the generation in progress keeps what it started with
        await _send_context_update(session, system_instruction="new", tools=new_tools)
        websocket.push(_text("1"), _turn_complete())
        await _drain(session)
        await session.send_client_content(turns=_user("two"))
        websocket.push(_text("2"), _turn_complete())
        await _drain(session)
        await _send_context_update(session, tools=[])
        await session.send_client_content(turns=_user("three"))
        websocket.push(_text("3"), _turn_complete())
        await _drain(session)

    first, second, third = [
        dict(span.attributes or {})
        for span in _spans(in_memory_span_exporter)
        if _kind(span) == "LLM"
    ]
    system = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
    schema = f"{SpanAttributes.LLM_TOOLS}.0.tool.json_schema"
    assert first[system] == "old"
    assert "get_weather" in str(first[schema])
    assert second[system] == "new"
    assert "get_time" in str(second[schema])
    assert third[system] == "new"
    assert schema not in third


@_needs_context_update
async def test_context_update_changes_only_the_fields_that_are_set(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    config = types.LiveConnectConfig(system_instruction="old", tools=TOOLS)
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await _send_context_update(session, system_instruction="new")
        await session.send_client_content(turns=_user("one"))
        websocket.push(_text("1"), _turn_complete())
        await _drain(session)

    (llm,) = [span for span in _spans(in_memory_span_exporter) if _kind(span) == "LLM"]
    attributes = llm.attributes or {}
    assert (
        attributes[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "new"
    )
    assert "get_weather" in str(attributes[f"{SpanAttributes.LLM_TOOLS}.0.tool.json_schema"])


@_needs_context_update
async def test_failed_context_update_keeps_the_previous_context(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    config = types.LiveConnectConfig(system_instruction="old")
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        websocket.send_error = RuntimeError("send failed")
        with pytest.raises(RuntimeError, match="send failed"):
            await _send_context_update(session, system_instruction="new")
        websocket.send_error = None
        await session.send_client_content(turns=_user("one"))
        websocket.push(_text("1"), _turn_complete())
        await _drain(session)

    (llm,) = [span for span in _spans(in_memory_span_exporter) if _kind(span) == "LLM"]
    system = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
    assert (llm.attributes or {})[system] == "old"


async def test_generation_ends_at_generation_complete(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("hello"))
        websocket.push(_audio(), {"serverContent": {"generationComplete": True}})
        await session._receive()
        await session._receive()
        # audio is still playing; usage comes with turn_complete
        before_turn_complete = time.time_ns()
        websocket.push(_turn_complete(totalTokenCount=5))
        await _drain(session)

    chain, llm = _spans(in_memory_span_exporter)
    assert (llm.end_time or 0) <= before_turn_complete
    assert (llm.attributes or {})[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 5
    assert (chain.end_time or 0) >= before_turn_complete


def _with_usage(frame: dict[str, Any], **usage: Any) -> dict[str, Any]:
    return {**frame, "usageMetadata": usage}


@pytest.mark.parametrize(
    "frames",
    [
        pytest.param(
            [_with_usage(_tool_call(), totalTokenCount=7)],
            id="same-frame",
        ),
        pytest.param(
            [_tool_call(), {"usageMetadata": {"totalTokenCount": 7}}],
            id="next-frame",
        ),
    ],
)
async def test_usage_after_tool_call_belongs_to_the_generation_that_made_it(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
    frames: list[dict[str, Any]],
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(*frames)
        for _ in frames:
            await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1", name="get_weather", response={"result": "sunny"}
            )
        )
        websocket.push(_text("It is sunny."), _turn_complete(totalTokenCount=20))
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL", "LLM"]
    _, first_llm, tool, second_llm = spans
    assert (first_llm.attributes or {})[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 7
    assert (second_llm.attributes or {})[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 20
    # the span's end time is when the model stopped, not when the usage arrived
    assert (first_llm.end_time or 0) <= (tool.start_time or 0)


async def test_generation_before_tool_call_ends_without_usage_when_session_closes(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        await session.send_client_content(turns=_user("weather?"))
        websocket.push(_tool_call())
        await session._receive()

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL"]
    _, llm, tool = spans
    assert llm.status.status_code is trace_api.StatusCode.OK
    assert (llm.end_time or 0) <= (tool.start_time or 0)
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in (llm.attributes or {})


async def test_context_attributes_applied_after_connect(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async with client.aio.live.connect(model=MODEL) as session:
        with using_attributes(
            session_id="late-session",
            user_id="late-user",
            metadata={"key": "value"},
            tags=["late-tag"],
            prompt_template="Weather in {city}?",
            prompt_template_version="v1",
            prompt_template_variables={"city": "Paris"},
        ):
            await session.send_client_content(turns=_user("weather?"))
        # received and answered outside the context the interaction started in
        websocket.push(_tool_call())
        await session._receive()
        await session.send_tool_response(
            function_responses=types.FunctionResponse(
                id="call-1", name="get_weather", response={"result": "sunny"}
            )
        )
        websocket.push(_text("It is sunny."), _turn_complete())
        await _drain(session)

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "TOOL", "LLM"]
    for span in spans:
        attributes = span.attributes or {}
        assert attributes[SpanAttributes.SESSION_ID] == "late-session"
        assert attributes[SpanAttributes.USER_ID] == "late-user"
        assert json.loads(str(attributes[SpanAttributes.METADATA])) == {"key": "value"}
        assert list(attributes[SpanAttributes.TAG_TAGS]) == ["late-tag"]  # type: ignore[arg-type]
        assert attributes[SpanAttributes.LLM_PROMPT_TEMPLATE] == "Weather in {city}?"
        assert attributes[SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION] == "v1"
        assert json.loads(str(attributes[SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES])) == {
            "city": "Paris"
        }


async def test_each_interaction_can_have_its_own_context_attributes(
    instrument: GoogleGenAIInstrumentor,
    websocket: FakeWebSocket,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with using_attributes(session_id="connect-session", user_id="connect-user"):
        async with client.aio.live.connect(model=MODEL) as session:
            with using_session("first-session"):
                await session.send_client_content(turns=_user("one"))
            websocket.push(_text("1"), _turn_complete())
            await _drain(session)
            await session.send_client_content(turns=_user("two"))
            websocket.push(_text("2"), _turn_complete())
            await _drain(session)

    spans = _spans(in_memory_span_exporter)
    assert [_kind(span) for span in spans] == ["CHAIN", "LLM", "CHAIN", "LLM"]
    sessions = [(span.attributes or {})[SpanAttributes.SESSION_ID] for span in spans]
    assert sessions == ["first-session", "first-session", "connect-session", "connect-session"]
    # anything the later context does not set still comes from connect()
    for span in spans:
        assert (span.attributes or {})[SpanAttributes.USER_ID] == "connect-user"
