"""A real AgentSession with OpenAI STT, LLM, and TTS, replayed from a vcrpy cassette.

The WAV fixture stands in for the microphone and an instant sink stands in for the speaker,
so the session runs without a LiveKit server. Only the OpenAI HTTP calls are recorded.
"""

import asyncio
import base64
import io
import json
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
from livekit import rtc
from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.voice import io as lk_io
from livekit.plugins import openai, silero
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import (
    REDACTED_VALUE,
    TraceConfig,
    suppress_tracing,
    using_session,
)
from openinference.instrumentation.livekit_agents import LiveKitAgentsInstrumentor

_FIXTURE = Path(__file__).parent / "fixtures" / "weather_question.wav"
_RATE = 24_000
_FRAME = 240


class _WavMicrophone(lk_io.AudioInput):
    """Plays the fixture at real time, then silence, like a participant's track."""

    def __init__(self, pcm: bytes) -> None:
        super().__init__(label="wav")
        self._pcm = pcm
        self._offset = 0
        self._due: float | None = None

    async def __anext__(self) -> rtc.AudioFrame:
        self._due = (self._due or time.monotonic()) + _FRAME / _RATE
        await asyncio.sleep(max(0.0, self._due - time.monotonic()))
        chunk = self._pcm[self._offset : self._offset + _FRAME * 2]
        self._offset += _FRAME * 2
        if len(chunk) < _FRAME * 2:
            chunk = b"\x00\x00" * _FRAME
        return rtc.AudioFrame(chunk, _RATE, 1, _FRAME)


class _InstantSpeaker(lk_io.AudioOutput):
    """Reports every segment as fully played as soon as it is flushed."""

    def __init__(self) -> None:
        super().__init__(
            label="instant",
            capabilities=lk_io.AudioOutputCapabilities(pause=False),
            sample_rate=_RATE,
        )
        self._captured = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if self._captured == 0.0:
            self.on_playback_started(created_at=time.time())
        await super().capture_frame(frame)
        self._captured += frame.duration

    def flush(self) -> None:
        super().flush()
        played, self._captured = self._captured, 0.0
        asyncio.get_running_loop().call_soon(
            lambda: self.on_playback_finished(playback_position=played, interrupted=False)
        )

    def clear_buffer(self) -> None:
        played, self._captured = self._captured, 0.0
        self.on_playback_finished(playback_position=played, interrupted=True)


@function_tool
async def get_weather(context: RunContext[None], city: str) -> str:
    """Get the current weather for a city."""
    return f"It is 18 degrees and cloudy in {city}."


async def _ask_about_the_weather() -> None:
    with wave.open(str(_FIXTURE)) as wav:
        pcm = wav.readframes(wav.getnframes())
    session: AgentSession[None] = AgentSession(
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts"),
        vad=silero.VAD.load(),
    )
    session.input.audio = _WavMicrophone(pcm)
    session.output.audio = _InstantSpeaker()
    tools_ran = asyncio.Event()
    replied = asyncio.Event()
    session.on("function_tools_executed", lambda _: tools_ran.set())

    def _on_agent_state(event: Any) -> None:
        if event.old_state == "speaking" and event.new_state == "listening" and tools_ran.is_set():
            replied.set()

    session.on("agent_state_changed", _on_agent_state)
    speeches: list[Any] = []
    session.on("speech_created", lambda event: speeches.append(event.speech_handle))
    await session.start(
        Agent(instructions="Answer briefly. Use get_weather for weather.", tools=[get_weather]),
        record=False,
    )
    await asyncio.wait_for(replied.wait(), timeout=60)
    await asyncio.wait_for(asyncio.gather(*speeches), timeout=10)
    await session.aclose()


@contextmanager
def _instrumented(config: TraceConfig | None = None) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = LiveKitAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider, config=config)
    try:
        yield exporter
    finally:
        instrumentor.uninstrument()


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    with _instrumented() as exporter:
        yield exporter


def _wav_seconds(url: Any) -> float:
    assert isinstance(url, str) and url.startswith("data:audio/wav;base64,")
    with wave.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (_RATE, 1, 2)
        frames = wav.readframes(wav.getnframes())
        assert any(frames), "the clip is not silence"
        return float(wav.getnframes() / wav.getframerate())


def _by_name(spans: tuple[ReadableSpan, ...]) -> dict[str, list[ReadableSpan]]:
    names: dict[str, list[ReadableSpan]] = {}
    for span in spans:
        names.setdefault(span.name, []).append(span)
    return names


@pytest.mark.vcr
def test_spoken_tool_turn(exporter: InMemorySpanExporter) -> None:
    asyncio.run(_ask_about_the_weather())

    spans = _by_name(exporter.get_finished_spans())
    for name in ("conversation.turn", "user", "get_weather"):
        assert len(spans[name]) == 1, name
    assert len(spans["assistant"]) == 2
    turn = spans["conversation.turn"][0]
    user = spans["user"][0]
    tool = spans["get_weather"][0]
    calls_tool, answers = sorted(spans["assistant"], key=lambda span: span.start_time or 0)
    assert turn.parent is None
    for child in (user, calls_tool, answers):
        assert child.parent is not None and child.parent.span_id == turn.context.span_id
    assert tool.parent is not None and tool.parent.span_id == calls_tool.context.span_id

    question = "What's the weather in London right now?"
    answer = "The weather in London right now is 18 degrees and cloudy."
    system = "Answer briefly. Use get_weather for weather."

    attributes = dict(turn.attributes or {})
    assert 1.4 < _wav_seconds(attributes.pop("input.audio.url")) < 4.0
    assert 2.0 < _wav_seconds(attributes.pop("output.audio.url")) < 8.0
    assert attributes == {
        "openinference.span.kind": "AUDIO",
        "end_reason": "complete",
        "llm.model_name": "gpt-4.1-mini",
        "input.value": question,
        "input.mime_type": "text/plain",
        "output.value": answer,
        "output.mime_type": "text/plain",
        "input.audio.mime_type": "audio/wav",
        "output.audio.mime_type": "audio/wav",
        "output.audio.transcript": answer,
    }

    assert dict(user.attributes or {}) == {
        "openinference.span.kind": "USER",
        "input.audio.transcript": question,
        "llm.model_name": "whisper-1",
        "llm.provider": "openai",
    }

    asked = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    attributes = dict(calls_tool.attributes or {})
    call_id = attributes.pop("llm.output_messages.0.message.tool_calls.0.tool_call.id")
    assert isinstance(call_id, str) and call_id
    assert isinstance(attributes.pop("time_to_first_token_ms"), int)
    requested = {
        "id": call_id,
        "function": {"name": "get_weather", "arguments": '{"city":"London"}'},
    }
    assert json.loads(str(attributes.pop("output.value"))) == [
        {"role": "assistant", "tool_calls": [requested]}
    ]
    assert json.loads(str(attributes.pop("input.value"))) == asked
    assert attributes == {
        "openinference.span.kind": "LLM",
        "llm.model_name": "gpt-4.1-mini",
        "llm.provider": "openai",
        "input.mime_type": "application/json",
        "output.mime_type": "application/json",
        "llm.input_messages.0.message.role": "system",
        "llm.input_messages.0.message.content": system,
        "llm.input_messages.1.message.role": "user",
        "llm.input_messages.1.message.content": question,
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": (
            '{"city":"London"}'
        ),
        "llm.token_count.prompt": 62,
        "llm.token_count.completion": 14,
    }

    attributes = dict(answers.attributes or {})
    assert isinstance(attributes.pop("time_to_first_token_ms"), int)
    tool_result = "It is 18 degrees and cloudy in London."
    assert json.loads(str(attributes.pop("input.value"))) == [
        *asked,
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": tool_result},
    ]
    assert attributes == {
        "openinference.span.kind": "LLM",
        "llm.model_name": "gpt-4.1-mini",
        "llm.provider": "openai",
        "input.mime_type": "application/json",
        "output.value": answer,
        "output.mime_type": "text/plain",
        "output.audio.transcript": answer,
        "llm.input_messages.0.message.role": "system",
        "llm.input_messages.0.message.content": system,
        "llm.input_messages.1.message.role": "user",
        "llm.input_messages.1.message.content": question,
        "llm.input_messages.2.message.role": "assistant",
        "llm.input_messages.2.message.tool_calls.0.tool_call.id": call_id,
        "llm.input_messages.2.message.tool_calls.0.tool_call.function.name": "get_weather",
        "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments": (
            '{"city": "London"}'
        ),
        "llm.input_messages.3.message.role": "tool",
        "llm.input_messages.3.message.content": tool_result,
        "llm.input_messages.3.message.tool_call_id": call_id,
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": answer,
        "llm.token_count.prompt": 94,
        "llm.token_count.completion": 14,
    }

    assert dict(tool.attributes or {}) == {
        "openinference.span.kind": "TOOL",
        "tool.name": "get_weather",
        "input.value": '{"city": "London"}',
        "input.mime_type": "application/json",
        "output.value": tool_result,
        "output.mime_type": "text/plain",
    }


_CASSETTE = "test_spoken_tool_turn.yaml"


@pytest.mark.vcr
@pytest.mark.default_cassette(_CASSETTE)
def test_caller_session_and_metadata_reach_every_span() -> None:
    with _instrumented() as exporter:

        async def run() -> None:
            with using_session("caller-session"):
                await _ask_about_the_weather()

        asyncio.run(run())
    spans = exporter.get_finished_spans()
    assert len(spans) == 5
    assert {(span.attributes or {}).get("session.id") for span in spans} == {"caller-session"}


@pytest.mark.vcr
@pytest.mark.default_cassette(_CASSETTE)
def test_hide_inputs_drops_what_the_user_said() -> None:
    with _instrumented(TraceConfig(hide_inputs=True)) as exporter:
        asyncio.run(_ask_about_the_weather())
    spans = _by_name(exporter.get_finished_spans())
    turn = dict(spans["conversation.turn"][0].attributes or {})
    assert turn["input.value"] == REDACTED_VALUE
    assert "input.audio.url" not in turn and "output.audio.url" in turn
    assert "input.audio.transcript" not in dict(spans["user"][0].attributes or {})
    for step in spans["assistant"]:
        attributes = dict(step.attributes or {})
        assert attributes["input.value"] == REDACTED_VALUE
        assert not any(key.startswith("llm.input_messages") for key in attributes)


@pytest.mark.vcr
@pytest.mark.default_cassette(_CASSETTE)
def test_suppressed_session_emits_no_spans() -> None:
    with _instrumented() as exporter:

        async def run() -> None:
            with suppress_tracing():
                await _ask_about_the_weather()

        asyncio.run(run())
    assert exporter.get_finished_spans() == ()


@pytest.mark.vcr
@pytest.mark.default_cassette(_CASSETTE)
def test_uninstrumented_session_emits_no_spans() -> None:
    with _instrumented() as exporter:
        pass
    asyncio.run(_ask_about_the_weather())
    assert exporter.get_finished_spans() == ()
