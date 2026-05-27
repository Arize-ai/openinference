"""realtime_with_tools.py — Realtime voice agent with tool calls.

Local mic/speaker setup with two function tools (get_weather, get_current_time).
Tool calls produce TOOL child spans nested under the LLM span in the trace.

Requirements: see examples/requirements.txt
Usage:
    export OPENAI_API_KEY=sk-...
    python realtime_with_tools.py
    # Try: "What's the weather in London?" or "What time is it in Tokyo?"

    # With verbose instrumentation/exporter logging:
    python realtime_with_tools.py --debug
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import os
import signal
import sys
from datetime import datetime

import numpy as np
import sounddevice as sd
from agents import function_tool  # type: ignore[import]
from agents.realtime import RealtimeAgent, RealtimeRunner  # type: ignore[import]
from agents.realtime.events import (  # type: ignore[import]
    RealtimeAudio,
    RealtimeAudioInterrupted,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
CHANNELS = 1
MIC_CHUNK_FRAMES = 1_200


class _LoggingSpanExporter(SpanExporter):
    """Wraps another exporter and logs each batch — to diagnose export issues."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans):  # type: ignore[override]
        names = [s.name for s in spans]
        logger.info("exporter: exporting %d span(s): %s", len(spans), names)
        result = self._inner.export(spans)
        logger.info("exporter: result=%s", result)
        return result

    def shutdown(self) -> None:
        logger.info("exporter: shutdown")
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # type: ignore[override]
        flush = getattr(self._inner, "force_flush", None)
        return bool(flush(timeout_millis)) if flush else True


def _setup_tracing(debug: bool = False) -> TracerProvider:
    provider = TracerProvider()
    endpoint = "http://127.0.0.1:6006/v1/traces"
    project = os.environ.get("PHOENIX_PROJECT_NAME", "realtime-with-tools")
    exporter: SpanExporter = OTLPSpanExporter(endpoint, headers={"project_name": project})
    print(f"Traces → Phoenix ({endpoint}, project={project})")
    if debug:
        exporter = _LoggingSpanExporter(exporter)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
    return provider


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@function_tool
def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather for a location."""
    if unit == "celsius":
        return f"The weather in {location} is 22 °C and sunny."
    return f"The weather in {location} is 72 °F and sunny."


@function_tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time for a timezone."""
    now = datetime.now().strftime("%I:%M %p")
    return f"The current time in {timezone} is {now}."


# ---------------------------------------------------------------------------
# Audio plumbing
# ---------------------------------------------------------------------------


def _make_mic_callback(
    mic_queue: asyncio.Queue[bytes],
    loop: asyncio.AbstractEventLoop,
) -> object:
    def _cb(indata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            logger.warning("Mic: %s", status)
        loop.call_soon_threadsafe(mic_queue.put_nowait, indata.copy().tobytes())

    return _cb


def _make_speaker_callback(speaker_buf: collections.deque[bytes]) -> object:
    def _cb(outdata: np.ndarray, frames: int, _time, status) -> None:
        needed = frames * CHANNELS * 2
        buf = bytearray(needed)
        pos = 0
        while pos < needed and speaker_buf:
            chunk = speaker_buf.popleft()
            take = min(len(chunk), needed - pos)
            buf[pos : pos + take] = chunk[:take]
            if take < len(chunk):
                speaker_buf.appendleft(chunk[take:])
            pos += take
        outdata[:] = np.frombuffer(bytes(buf), dtype=np.int16).reshape(-1, CHANNELS)

    return _cb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("openinference.instrumentation.openai_agents").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        print("Debug logging enabled (instrumentation + exporter)")
    provider = _setup_tracing(debug=args.debug)
    loop = asyncio.get_running_loop()

    mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
    speaker_buf: collections.deque[bytes] = collections.deque()

    agent = RealtimeAgent(
        name="Assistant",
        instructions=(
            "You are a helpful voice assistant. "
            "You have tools to look up weather and the current time — use them when asked. "
            "Keep responses short and conversational."
        ),
        tools=[get_weather, get_current_time],
    )

    stop = asyncio.Event()
    loop.add_signal_handler(signal.SIGINT, stop.set)

    print("Connecting to OpenAI Realtime… (Ctrl-C to stop)")

    runner = RealtimeRunner(agent)
    async with await runner.run() as session:
        print("Connected. Speak now.\n")
        print("  Try: 'What's the weather in London?'")
        print("  Try: 'What time is it in Tokyo?'\n")
        print("  (Type a message and press Enter to send text input.)\n")

        async def _send_mic() -> None:
            while not stop.is_set():
                try:
                    chunk = await asyncio.wait_for(mic_queue.get(), timeout=0.1)
                    await session.send_audio(chunk)
                except asyncio.TimeoutError:
                    continue

        async def _handle_events() -> None:
            async for event in session:
                if stop.is_set():
                    break
                if isinstance(event, RealtimeAudio):
                    speaker_buf.append(event.audio.data)
                elif isinstance(event, RealtimeAudioInterrupted):
                    speaker_buf.clear()
                    print("[interrupted]")

        async def _stdin_reader() -> None:
            """Read typed lines from stdin and inject them as user text messages."""
            while not stop.is_set():
                try:
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                except Exception:
                    return
                if not line:  # EOF
                    return
                text = line.strip()
                if not text:
                    continue
                print(f"[text→] {text}")
                try:
                    await session.send_message(text)  # type: ignore[attr-defined]
                except Exception as exc:
                    logger.warning("send_message failed: %s", exc)

        send_task = asyncio.create_task(_send_mic())
        events_task = asyncio.create_task(_handle_events())
        stdin_task = asyncio.create_task(_stdin_reader())

        async def _wait_and_teardown() -> None:
            await stop.wait()
            print("\nStopping…")
            send_task.cancel()
            events_task.cancel()
            stdin_task.cancel()
            await asyncio.gather(send_task, events_task, stdin_task, return_exceptions=True)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=MIC_CHUNK_FRAMES,
            callback=_make_mic_callback(mic_queue, loop),
        ), sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=MIC_CHUNK_FRAMES,
            callback=_make_speaker_callback(speaker_buf),
        ):
            await _wait_and_teardown()

    provider.force_flush()
    provider.shutdown()
    print("Done. Traces flushed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime voice agent with tool calls and OpenInference tracing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose debug logging for instrumentation and span export.",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
